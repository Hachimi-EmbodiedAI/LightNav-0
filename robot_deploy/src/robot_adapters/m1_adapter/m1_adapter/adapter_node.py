"""Bridge the generic LightNav robot interface to the Genisom M1 / robot_control.

WHAT THIS NODE IS FOR
    ``vln_mpc`` publishes ``TwistStamped`` on ``mpc/cmd_vel``.  The M1's
    ``robot_control`` node subscribes to plain ``geometry_msgs/Twist`` on
    ``/cmd_vel``.  So *something* has to sit between them.  This adapter is that
    something, and it deliberately does more than rename a type, because the M1
    has three constraints the generic stack knows nothing about:

      1. POSTURE.  The Genisom SDK only accepts Move() in walk/stair/crawl_walk/
         climb/gait/slim/sk_walk/sand.  In lie_down / crawl / locked a velocity
         command is ignored or silently changes posture.  We refuse up front.
      2. SOFT E-STOP.  The robot has its own software emergency stop, exposed as
         ``/robot_control/soft_emergency_stop`` (std_srvs/SetBool).  While it is
         engaged every velocity command must be zero, regardless of source.
      3. CONTROL OWNERSHIP.  The SDK grants control to one client at a time and
         an attached phone APP always wins -- ``TakeControl()`` will fail while
         the APP is connected.  We call it on startup and report loudly if it
         does not succeed, because otherwise every command silently does nothing.

WHAT IT DELIBERATELY DOES NOT DO
    * It does not publish odometry.  ``robot_control`` already publishes
      ``/robot_control/odometry`` (nav_msgs/Odometry, odom frame); remap it
      straight into ``vln_mpc``'s ``odom`` topic.  Re-publishing would add
      latency and a second source of truth.
    * It does not publish TF.  Same reason: robot_control owns odom->base_link.
    * It does not touch posture service semantics.  ``stand``/``walk`` both map
      to ``stand_up`` because that is what the SDK exposes; ``sit`` maps to
      ``lie_down``.  ``crawl``/``stair``/``locked`` are exposed as-is.

INTERFACE (mirrors tron_adapter so vln_bringup launches stay interchangeable)
    subscribes  web/cmd_vel        TwistStamped   manual (web UI sticks/WASD)
                mpc/cmd_vel        TwistStamped   automatic (MPC)
                robot/state        RobotStatus    posture + e-stop + control src
    publishes   /cmd_vel           Twist          -> robot_control
                cmd_vel_sent       TwistStamped   what we actually forwarded
                control/source     String         disabled | manual | auto
                robot/events       String         human-readable transitions
                diagnostics        DiagnosticArray
    services    control/set_manual SetBool
                control/set_auto   SetBool
                control/stop       Trigger        disable + zero velocity
                robot/stand        Trigger        -> stand_up
                robot/walk         Trigger        -> stand_up (same service)
                robot/sit          Trigger        -> lie_down
                robot/crawl        Trigger        -> crawl
                robot/stair        Trigger        -> stair
                robot/emergency    SetBool        -> soft_emergency_stop

TIMING
    ``robot_control`` has ``command_timeout_ms: 300``, so if this process dies
    the dog stops by itself within 300 ms.  Our own watchdog (default 350 ms on
    the INPUT side, i.e. how stale an MPC/web command may be) is therefore a
    second line of defence, not the only one.  The output publish rate is 20 Hz
    to match what robot_control expects to see.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import rclpy
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from robot_control.msg import RobotStatus
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger

from .safety import normalise_posture, posture_can_move, select_command, stop_reason

VALID_SOURCES = {"disabled", "manual", "auto"}

# Which robot_control service implements each requested posture.
#   stand / walk -> stand_up : StandUp() reports WALK when it completes, so
#                              "walk" is not a separate service on this robot.
#   sit          -> lie_down : the SDK has no "sit"; 卧倒 is the resting pose.
POSTURE_SERVICES = {
    "stand": "stand_up",
    "walk": "stand_up",
    "sit": "lie_down",
    "crawl": "crawl",
    "stair": "stair",
    "locked": "locked",
}


class M1AdapterNode(Node):
    def __init__(self) -> None:
        super().__init__("m1_adapter")

        # ---- parameters -------------------------------------------------
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("input_watchdog_s", 0.35)
        self.declare_parameter("state_timeout_s", 1.0)
        self.declare_parameter("zero_frames", 5)
        self.declare_parameter("auto_take_control", True)
        self.declare_parameter("take_control_retry_s", 5.0)
        self.declare_parameter("max_linear", 0.0)
        self.declare_parameter("max_angular", 0.0)
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("state_topic", "/robot_control/state")

        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.input_watchdog_s = float(self.get_parameter("input_watchdog_s").value)
        self.state_timeout_s = float(self.get_parameter("state_timeout_s").value)
        self.final_zero_frames = int(self.get_parameter("zero_frames").value)
        self.auto_take_control = bool(self.get_parameter("auto_take_control").value)
        self.take_control_retry_s = float(
            self.get_parameter("take_control_retry_s").value
        )
        self.max_linear = float(self.get_parameter("max_linear").value)
        self.max_angular = float(self.get_parameter("max_angular").value)
        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.state_topic = str(self.get_parameter("state_topic").value)

        if (
            self.control_rate_hz <= 0.0
            or self.input_watchdog_s <= 0.0
            or self.state_timeout_s <= 0.0
            or self.final_zero_frames < 0
            or self.max_linear < 0.0
            or self.max_angular < 0.0
        ):
            raise ValueError("invalid M1 adapter parameter")

        # ---- publishers / subscriptions ---------------------------------
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.sent_pub = self.create_publisher(TwistStamped, "cmd_vel_sent", 20)
        self.source_pub = self.create_publisher(String, "control/source", latched)
        self.event_pub = self.create_publisher(String, "robot/events", 20)
        self.diagnostics_pub = self.create_publisher(DiagnosticArray, "diagnostics", 10)

        self.create_subscription(
            TwistStamped, "web/cmd_vel", self._on_manual_command, 10
        )
        self.create_subscription(
            TwistStamped, "mpc/cmd_vel", self._on_auto_command, 10
        )
        # robot_control publishes /robot_control/state with RELIABLE; match it or
        # we silently get nothing (this exact mismatch wasted time in hachimi_ws).
        self.create_subscription(
            RobotStatus, self.state_topic, self._on_robot_status, 10
        )

        # ---- services ----------------------------------------------------
        # Posture and e-stop handlers block while they wait for robot_control's
        # reply.  That is only safe in a ReentrantCallbackGroup: in the default
        # MutuallyExclusive group the service client's future needs the very
        # executor thread we are sleeping on, and the call times out every time.
        # (With the real robot_control in another process it happens to work,
        # but a co-located stub -- our tests, or a future merged node --
        # deadlocks.  Reentrant + MultiThreadedExecutor is correct either way.)
        self._service_group = ReentrantCallbackGroup()
        self.create_service(
            SetBool,
            "control/set_manual",
            self._set_manual_enabled,
            callback_group=self._service_group,
        )
        self.create_service(
            SetBool,
            "control/set_auto",
            self._set_auto_enabled,
            callback_group=self._service_group,
        )
        self.create_service(
            Trigger,
            "control/stop",
            self._software_stop,
            callback_group=self._service_group,
        )
        for name in ("stand", "walk", "sit", "crawl", "stair"):
            self.create_service(
                Trigger,
                f"robot/{name}",
                self._make_posture_handler(name),
                callback_group=self._service_group,
            )
        self.create_service(
            SetBool,
            "robot/emergency",
            self._set_emergency,
            callback_group=self._service_group,
        )

        # ---- robot_control service clients -------------------------------
        self._posture_clients = {}
        for requested, service in POSTURE_SERVICES.items():
            name = f"/robot_control/{service}"
            if name in self._posture_clients:
                continue
            self._posture_clients[name] = self.create_client(Trigger, name)
        # soft_emergency_stop is std_srvs/SetBool, NOT Trigger.  Getting this
        # wrong makes the e-stop path silently dead code (hachimi_ws BUGFIX).
        self._estop_client = self.create_client(
            SetBool, "/robot_control/soft_emergency_stop"
        )
        self._take_control_client = self.create_client(
            Trigger, "/robot_control/take_control"
        )

        # ---- shared state -------------------------------------------------
        self._lock = threading.Lock()
        self._source = "disabled"
        self._manual_command = (0.0, 0.0)
        self._auto_command = (0.0, 0.0)
        self._manual_received_s = float("-inf")
        self._auto_received_s = float("-inf")
        self._zero_frames_remaining = 0
        self._posture = ""
        self._estop_engaged = False
        self._control_source_name = ""
        self._last_status_s = float("-inf")
        self._last_hold_reason = ""
        self._have_control = False
        self._next_take_control_s = 0.0

        # ---- timers --------------------------------------------------------
        self.create_timer(1.0 / self.control_rate_hz, self._command_tick)
        self.create_timer(1.0, self._diagnostics_tick)
        if self.auto_take_control:
            self.create_timer(1.0, self._take_control_tick)

        self._publish_source()
        self.get_logger().info(
            f"m1_adapter up: {self.control_rate_hz:.0f} Hz -> {self.cmd_vel_topic}, "
            f"state <- {self.state_topic}, "
            f"watchdog {self.input_watchdog_s:.2f}s, "
            f"clamp linear={self.max_linear or 'off'} angular={self.max_angular or 'off'}"
        )

    # ------------------------------------------------------------- inputs
    def _on_manual_command(self, message: TwistStamped) -> None:
        command = (float(message.twist.linear.x), float(message.twist.angular.z))
        with self._lock:
            self._manual_command = command
            self._manual_received_s = time.monotonic()

    def _on_auto_command(self, message: TwistStamped) -> None:
        command = (float(message.twist.linear.x), float(message.twist.angular.z))
        with self._lock:
            self._auto_command = command
            self._auto_received_s = time.monotonic()

    def _on_robot_status(self, message: RobotStatus) -> None:
        posture = normalise_posture(message.motion_status_name)
        # software_emergency_status is a uint8 where 0 means "not engaged".
        # Reading the string is friendlier in logs but the number is what we
        # compare; accept the string spelling too in case it changes.
        estop = bool(message.software_emergency_status) or (
            "engage" in str(message.software_emergency_status_name).lower()
            and "dis" not in str(message.software_emergency_status_name).lower()
        )
        with self._lock:
            previous_posture = self._posture
            was_estopped = self._estop_engaged
            self._posture = posture
            self._estop_engaged = estop
            self._control_source_name = str(message.control_source_name)
            self._last_status_s = time.monotonic()
        if posture != previous_posture:
            self._event(f"posture -> {posture or 'unknown'}")
        if estop != was_estopped:
            self._event(f"software e-stop {'engaged' if estop else 'released'}")

    # ------------------------------------------------------------ services
    def _set_manual_enabled(self, request, response):
        self._set_source("manual" if request.data else "disabled")
        response.success = True
        response.message = (
            "manual control enabled" if request.data else "control disabled"
        )
        return response

    def _set_auto_enabled(self, request, response):
        self._set_source("auto" if request.data else "disabled")
        response.success = True
        response.message = "auto control enabled" if request.data else "control disabled"
        return response

    def _set_source(self, source: str) -> None:
        if source not in VALID_SOURCES:
            raise ValueError(f"invalid control source {source!r}")
        now = time.monotonic()
        with self._lock:
            self._source = source
            self._manual_command = (0.0, 0.0)
            self._auto_command = (0.0, 0.0)
            self._manual_received_s = now
            self._auto_received_s = now
            # Always emit a few explicit zeros after a source change: if the
            # previous source was mid-command, one zero frame can be lost to a
            # QoS hiccup and the dog keeps walking.
            self._zero_frames_remaining = max(
                self._zero_frames_remaining, self.final_zero_frames
            )
        self._publish_source()
        self._event(f"control source -> {source}")

    def _software_stop(self, _request, response):
        self._set_source("disabled")
        self._publish_zero()
        response.success = True
        response.message = "control disabled and zero velocity published"
        return response

    def _make_posture_handler(self, requested: str):
        def handler(_request, response):
            return self._request_posture(requested, response)

        return handler

    def _request_posture(self, requested: str, response):
        service = POSTURE_SERVICES[requested]
        client = self._posture_clients.get(f"/robot_control/{service}")
        if client is None:
            response.success = False
            response.message = f"no client for {service}"
            return response
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=2.0):
            response.success = False
            response.message = (
                f"/robot_control/{service} not available -- is robot_control running?"
            )
            return response
        # Refuse to change posture while moving: robot_control would accept the
        # call and the SDK would drop the velocity stream mid-gait.
        with self._lock:
            self._source = "disabled"
            self._manual_command = (0.0, 0.0)
            self._auto_command = (0.0, 0.0)
        self._publish_zero()
        self._publish_source()

        future = client.call_async(Trigger.Request())
        # Short spin so the Trigger response is filled in; the caller (a service
        # callback on this node) cannot block the executor indefinitely.
        deadline = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done():
            response.success = False
            response.message = f"/robot_control/{service} did not answer within 3s"
            return response
        if future.result() is None:
            response.success = False
            response.message = f"/robot_control/{service} returned no response"
            return response
        result = future.result()
        response.success = bool(result.success)
        response.message = f"{service}: {result.message}"
        self._event(f"posture request {requested} -> {service}: {result.message}")
        return response

    def _set_emergency(self, request, response):
        engage = bool(request.data)
        if not self._estop_client.service_is_ready() and not self._estop_client.wait_for_service(
            timeout_sec=2.0
        ):
            response.success = False
            response.message = "/robot_control/soft_emergency_stop not available"
            return response
        # Zero our own output first: the service call may take a moment and we
        # must not keep streaming velocity into a robot being braked.
        self._set_source("disabled")
        self._publish_zero()

        message = SetBool.Request()
        message.data = engage
        future = self._estop_client.call_async(message)
        deadline = time.monotonic() + 3.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)
        if not future.done() or future.result() is None:
            response.success = False
            response.message = "soft_emergency_stop did not answer"
            return response
        result = future.result()
        response.success = bool(result.success)
        response.message = f"e-stop {'engage' if engage else 'release'}: {result.message}"
        self._event(response.message)
        return response

    # ------------------------------------------------------ control ownership
    def _take_control_tick(self) -> None:
        """Claim SDK control once, retry while it fails.

        An attached phone APP always wins and TakeControl() cannot preempt it
        (sdk_control_ownership_zh.md).  Retrying is the only option; shouting
        about it in the log is more useful than failing silently.
        """
        with self._lock:
            have_control = self._have_control
        if have_control:
            return
        now = time.monotonic()
        if now < self._next_take_control_s:
            return
        self._next_take_control_s = now + self.take_control_retry_s

        client = self._take_control_client
        if not client.service_is_ready() and not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                "/robot_control/take_control not available -- is robot_control running?"
            )
            return
        future = client.call_async(Trigger.Request())

        def _done(_future) -> None:
            try:
                result = _future.result()
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                self.get_logger().warn(f"take_control failed: {exc}")
                return
            if result is not None and result.success:
                with self._lock:
                    self._have_control = True
                self._event("control acquired from SDK")
                self.get_logger().info(f"take_control OK: {result.message}")
            else:
                message = result.message if result is not None else "no response"
                self.get_logger().warn(
                    "take_control refused (%s). An attached phone APP has priority; "
                    "close it, or control will not work.", message
                )

        future.add_done_callback(_done)

    # ------------------------------------------------------------ main loop
    def _command_tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            command = select_command(
                source=self._source,
                posture=self._posture,
                estop_engaged=self._estop_engaged,
                manual_command=self._manual_command,
                manual_age_s=now - self._manual_received_s,
                auto_command=self._auto_command,
                auto_age_s=now - self._auto_received_s,
                watchdog_s=self.input_watchdog_s,
                max_linear=self.max_linear,
                max_angular=self.max_angular,
            )
            status_age = now - self._last_status_s
            if command is not None:
                self._zero_frames_remaining = self.final_zero_frames
            elif self._zero_frames_remaining > 0:
                command = (0.0, 0.0)
                self._zero_frames_remaining -= 1

            reason = stop_reason(
                posture=self._posture,
                estop_engaged=self._estop_engaged,
                last_status_age_s=status_age,
                status_timeout_s=self.state_timeout_s,
            )
            if reason != self._last_hold_reason:
                self._last_hold_reason = reason
                if reason:
                    self.get_logger().warn(f"holding zero velocity: {reason}")

        if command is None:
            return
        self._publish_cmd(command[0], command[1])

    def _publish_cmd(self, linear: float, angular: float) -> None:
        if not rclpy.ok():
            return
        # Plain Twist: robot_control subscribes geometry_msgs/Twist (verified on
        # hardware with `ros2 topic info /cmd_vel -v`).
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self.cmd_pub.publish(twist)

        sent = TwistStamped()
        sent.header.stamp = self.get_clock().now().to_msg()
        sent.header.frame_id = "base_link"
        sent.twist.linear.x = float(linear)
        sent.twist.angular.z = float(angular)
        self.sent_pub.publish(sent)

    def _publish_zero(self) -> None:
        self._publish_cmd(0.0, 0.0)

    # ---------------------------------------------------------- diagnostics
    def _diagnostics_tick(self) -> None:
        now = time.monotonic()
        with self._lock:
            posture = self._posture
            estop = self._estop_engaged
            source = self._source
            status_age = now - self._last_status_s
            control_source_name = self._control_source_name
            have_control = self._have_control

        reason = stop_reason(
            posture=posture,
            estop_engaged=estop,
            last_status_age_s=status_age,
            status_timeout_s=self.state_timeout_s,
        )
        status = DiagnosticStatus()
        status.name = "m1_adapter"
        status.hardware_id = "genisom_m1"
        if estop:
            status.level = DiagnosticStatus.ERROR
            status.message = "software e-stop engaged"
        elif status_age > self.state_timeout_s:
            status.level = DiagnosticStatus.ERROR
            status.message = "no /robot_control/state"
        elif not posture_can_move(posture):
            status.level = DiagnosticStatus.WARN
            status.message = f"posture {posture or 'unknown'} cannot move"
        else:
            status.level = DiagnosticStatus.OK
            status.message = f"{posture}, {source}"

        for key, value in (
            ("posture", posture or "unknown"),
            ("can_move", str(posture_can_move(posture))),
            ("estop_engaged", str(estop)),
            ("control_source", source),
            ("robot_control_source", control_source_name),
            ("sdk_control_acquired", str(have_control)),
            ("state_age_s", f"{status_age:.2f}"),
            ("hold_reason", reason or ""),
            ("max_linear", str(self.max_linear)),
            ("max_angular", str(self.max_angular)),
        ):
            item = KeyValue()
            item.key = key
            item.value = value
            status.values.append(item)

        array = DiagnosticArray()
        array.header.stamp = self.get_clock().now().to_msg()
        array.status.append(status)
        self.diagnostics_pub.publish(array)

    def _publish_source(self) -> None:
        if not rclpy.ok():
            return
        with self._lock:
            source = self._source
        message = String()
        message.data = source
        self.source_pub.publish(message)

    def _event(self, text: str) -> None:
        if not rclpy.ok():
            return
        message = String()
        message.data = text
        self.event_pub.publish(message)
        self.get_logger().info(text)


def main(args: Optional[list[str]] = None) -> None:
    rclpy.init(args=args)
    node = M1AdapterNode()
    # MultiThreadedExecutor: the posture/e-stop service handlers block while
    # waiting for robot_control's reply, so they must not own the only executor
    # thread.  Paired with ReentrantCallbackGroup on those services.
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node._publish_zero()
        except Exception:  # noqa: BLE001 - shutting down
            pass
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
