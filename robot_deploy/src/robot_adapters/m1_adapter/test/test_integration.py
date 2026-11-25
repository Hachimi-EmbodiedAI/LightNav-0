"""End-to-end tests for m1_adapter against a FAKE m1_robot_driver.

Real ROS 2 messages, real DDS, real timers -- but the chassis driver is
replaced by a small stub that publishes RobotStatus and records the services and
twists it receives.  Nothing here can move a robot.

Run with the ROS venv:

    source /home/jetson/Public/LightNav-0/robot_deploy/scripts/m1_env.sh
    .venv_ros/bin/python -m pytest test/test_integration.py -q

Skipped automatically when robot_control's messages are not importable, so this
file does not break a plain ``pytest`` run on a machine without the vendor stack.
"""

from __future__ import annotations

import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")
pytest.importorskip("robot_control.msg")

from geometry_msgs.msg import Twist, TwistStamped  # noqa: E402
from robot_control.msg import RobotStatus  # noqa: E402
from std_srvs.srv import SetBool, Trigger  # noqa: E402

from m1_adapter.adapter_node import M1AdapterNode  # noqa: E402


class FakeRobotControl:
    """Minimal stand-in for m1_robot_driver: state out, services in, cmd_vel in."""

    def __init__(self, node):
        self.node = node
        self.twists = []
        self.posture_calls = []
        self.estop_calls = []
        self.switch_remote_calls = 0
        self.take_control_calls = 0
        self.posture = "walk"
        self.machine_status = "remote"
        self.estop = False
        self.hw_estop = False
        self.take_control_ok = True
        self._lock = threading.Lock()

        self.state_pub = node.create_publisher(RobotStatus, "/robot_control/state", 10)
        self.create_sub = node.create_subscription(Twist, "/cmd_vel", self._on_twist, 10)

        for name in ("stand_up", "lie_down", "crawl", "stair", "locked"):
            node.create_service(
                Trigger, f"/robot_control/{name}", self._make_posture_srv(name)
            )
        node.create_service(
            Trigger, "/robot_control/switch_remote", self._on_switch_remote
        )
        node.create_service(
            SetBool, "/robot_control/soft_emergency_stop", self._on_estop
        )
        node.create_service(Trigger, "/robot_control/take_control", self._on_take)

    # -- inputs from the adapter ------------------------------------------
    def _on_twist(self, msg: Twist) -> None:
        with self._lock:
            self.twists.append((msg.linear.x, msg.angular.z))

    def _make_posture_srv(self, name):
        def handler(_request, response):
            with self._lock:
                self.posture_calls.append(name)
            response.success = True
            response.message = f"{name} accepted"
            return response

        return handler

    def _on_switch_remote(self, _request, response):
        with self._lock:
            self.switch_remote_calls += 1
            self.machine_status = "remote"
        response.success = True
        response.message = "switched to remote"
        return response

    def _on_estop(self, request, response):
        with self._lock:
            self.estop = bool(request.data)
            self.estop_calls.append(bool(request.data))
        response.success = True
        response.message = "ok"
        return response

    def _on_take(self, _request, response):
        with self._lock:
            self.take_control_calls += 1
            ok = self.take_control_ok
        response.success = ok
        response.message = "granted" if ok else "APP has control"
        return response

    # -- test helpers ------------------------------------------------------
    def publish_status(
        self, posture=None, estop=None, hw_estop=None, machine_status=None
    ) -> None:
        with self._lock:
            if posture is not None:
                self.posture = posture
            if estop is not None:
                self.estop = estop
            if hw_estop is not None:
                self.hw_estop = hw_estop
            if machine_status is not None:
                self.machine_status = machine_status
            msg = RobotStatus()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.motion_status_name = self.posture
            msg.machine_status_name = self.machine_status
            # SDK 0.2.3 EmergencyStatus: 1=recover (safe), 2=stop (engaged)
            msg.software_emergency_status = 2 if self.estop else 1
            msg.software_emergency_status_name = "stop" if self.estop else "recover"
            msg.hardware_emergency_status = 2 if self.hw_estop else 1
            msg.hardware_emergency_status_name = "stop" if self.hw_estop else "recover"
            msg.control_source_name = "sdk"
        self.state_pub.publish(msg)

    def snapshot(self):
        with self._lock:
            return list(self.twists), list(self.posture_calls), list(self.estop_calls)

    def clear(self) -> None:
        with self._lock:
            self.twists.clear()
            self.posture_calls.clear()
            self.estop_calls.clear()


@pytest.fixture
def rig():
    from rclpy.executors import MultiThreadedExecutor

    rclpy.init()
    adapter = M1AdapterNode()
    # Tests drive the fake's state, so don't fight over control ownership.
    adapter.auto_take_control = False
    fake_node = rclpy.create_node("fake_robot_control")
    fake = FakeRobotControl(fake_node)
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(adapter)
    executor.add_node(fake_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()
    try:
        time.sleep(0.4)
        yield adapter, fake, executor
    finally:
        executor.shutdown()
        spin_thread.join(timeout=2.0)
        adapter.destroy_node()
        fake_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def drive(
    adapter, fake, executor, *, command, posture, machine_status="remote", seconds=0.4
):
    """Publish one command in a given posture/machine_status and collect output."""
    fake.publish_status(posture=posture, machine_status=machine_status)
    adapter._set_source("auto")
    msg = TwistStamped()
    msg.twist.linear.x = command[0]
    msg.twist.angular.z = command[1]

    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        adapter._on_auto_command(msg)
        time.sleep(0.05)
    twists, _, _ = fake.snapshot()
    return twists


def test_command_reaches_cmd_vel_in_walk_posture(rig):
    adapter, fake, executor = rig
    twists = drive(adapter, fake, executor, command=(0.25, 0.0), posture="walk")
    nonzero = [t for t in twists if abs(t[0]) > 1e-6]
    assert nonzero, f"expected a non-zero /cmd_vel, got {twists[:5]}"
    assert abs(nonzero[-1][0] - 0.25) < 1e-6


def test_no_command_while_lying_down(rig):
    adapter, fake, executor = rig
    twists = drive(adapter, fake, executor, command=(0.25, 0.0), posture="lie_down")
    assert all(abs(t[0]) < 1e-9 for t in twists), (
        f"lie_down must never move, got {twists[:5]}"
    )


def test_no_command_when_machine_status_is_idle(rig):
    adapter, fake, executor = rig
    twists = drive(
        adapter,
        fake,
        executor,
        command=(0.25, 0.0),
        posture="walk",
        machine_status="idle",
    )
    assert all(abs(t[0]) < 1e-9 for t in twists), (
        f"idle machine_status must never move, got {twists[:5]}"
    )


def test_no_command_while_estopped(rig):
    adapter, fake, executor = rig
    fake.publish_status(posture="walk", estop=True)
    adapter._set_source("auto")
    msg = TwistStamped()
    msg.twist.linear.x = 0.3
    for _ in range(10):
        adapter._on_auto_command(msg)
        time.sleep(0.05)
    twists, _, _ = fake.snapshot()
    assert all(abs(t[0]) < 1e-9 for t in twists), f"e-stop must hold zero, got {twists[:5]}"


def test_no_command_while_hardware_estopped(rig):
    adapter, fake, executor = rig
    fake.publish_status(posture="walk", hw_estop=True)
    adapter._set_source("auto")
    msg = TwistStamped()
    msg.twist.linear.x = 0.3
    for _ in range(10):
        adapter._on_auto_command(msg)
        time.sleep(0.05)
    twists, _, _ = fake.snapshot()
    assert all(abs(t[0]) < 1e-9 for t in twists), (
        f"hardware e-stop must hold zero, got {twists[:5]}"
    )


def test_unknown_posture_holds_zero(rig):
    adapter, fake, executor = rig
    adapter._set_source("auto")
    msg = TwistStamped()
    msg.twist.linear.x = 0.3
    for _ in range(10):
        adapter._on_auto_command(msg)
        time.sleep(0.05)
    twists, _, _ = fake.snapshot()
    assert all(abs(t[0]) < 1e-9 for t in twists), "no state => no motion"


def test_stop_service_disables_and_zeroes(rig):
    adapter, fake, executor = rig
    drive(adapter, fake, executor, command=(0.25, 0.0), posture="walk")
    fake.clear()
    response = Trigger.Response()
    adapter._software_stop(Trigger.Request(), response)
    assert response.success
    time.sleep(0.2)
    twists, _, _ = fake.snapshot()
    assert all(abs(t[0]) < 1e-9 for t in twists), f"stop must zero, got {twists[:5]}"


def test_posture_service_maps_to_vendor_service(rig):
    adapter, fake, executor = rig
    response = Trigger.Response()
    adapter._request_posture("sit", response)
    twists, calls, _ = fake.snapshot()
    assert response.success, response.message
    assert "lie_down" in calls, f"sit should call lie_down, got {calls}"
    assert all(abs(t[0]) < 1e-9 for t in twists), "posture change must zero velocity"


def test_switch_remote_service_forwards_to_driver(rig):
    adapter, fake, executor = rig
    response = Trigger.Response()
    adapter._switch_remote(Trigger.Request(), response)
    assert response.success, response.message
    assert fake.switch_remote_calls == 1


def test_walk_and_stand_both_map_to_stand_up(rig):
    adapter, fake, executor = rig
    for requested in ("stand", "walk"):
        fake.clear()
        response = Trigger.Response()
        adapter._request_posture(requested, response)
        twists, calls, _ = fake.snapshot()
        assert response.success, response.message
        assert calls == ["stand_up"], f"{requested} -> {calls}"


def test_emergency_service_drives_setbool(rig):
    adapter, fake, executor = rig
    response = SetBool.Response()
    request = SetBool.Request()
    request.data = True
    adapter._set_emergency(request, response)
    twists, _, estops = fake.snapshot()
    assert response.success, response.message
    assert estops == [True], f"expected one engage, got {estops}"
    assert all(abs(t[0]) < 1e-9 for t in twists)


def test_source_change_emits_zero_frames(rig):
    adapter, fake, executor = rig
    drive(adapter, fake, executor, command=(0.3, 0.0), posture="walk")
    fake.clear()
    adapter._set_source("disabled")
    time.sleep(0.2)
    twists, _, _ = fake.snapshot()
    assert twists, "a source change must emit explicit zeros"
    assert all(abs(t[0]) < 1e-9 for t in twists)


def test_clamping_applies_when_configured(rig):
    adapter, fake, executor = rig
    adapter.max_linear = 0.3
    twists = drive(adapter, fake, executor, command=(5.0, 0.0), posture="walk")
    nonzero = [t for t in twists if abs(t[0]) > 1e-6]
    assert nonzero, "clamping must not silence the command"
    assert abs(nonzero[-1][0] - 0.3) < 1e-6, (
        f"expected clamp to 0.3, got {nonzero[-1][0]}"
    )
