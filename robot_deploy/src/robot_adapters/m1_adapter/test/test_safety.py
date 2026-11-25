"""Safety-rule and RTSP encoding tests for the M1 adapter (no ROS, no robot)."""

import math

import numpy as np

from m1_adapter.rtsp_camera_node import LatestFrameBuffer, encode_frame_jpeg
from m1_adapter.safety import (
    MOVABLE_POSTURES,
    is_estop_engaged,
    machine_status_can_move,
    normalise_machine_status,
    normalise_posture,
    posture_can_move,
    select_command,
    stop_reason,
)


def arguments(**overrides):
    values = {
        "source": "manual",
        "posture": "walk",
        "estop_engaged": False,
        "hardware_estop_engaged": False,
        "machine_status": "remote",
        "status_stale": False,
        "manual_command": (0.4, -0.3),
        "manual_age_s": 0.1,
        "auto_command": (0.2, 0.1),
        "auto_age_s": 0.1,
        "watchdog_s": 0.35,
    }
    values.update(overrides)
    return values


# ---------------------------------------------------------------- posture gate
def test_all_documented_movable_postures_are_accepted():
    """sdk_state_zh.md 移动能力 column -- every one of these can Move()."""
    for posture in (
        "walk",
        "stand_up",
        "stair",
        "crawl_walk",
        "climb",
        "gait",
        "slim",
        "sk_walk",
        "sand",
    ):
        assert posture_can_move(posture), posture


def test_non_movable_postures_are_rejected():
    """卧倒/匍匐/平衡站立/锁定 -- Move() is refused or silently ignored."""
    for posture in ("lie_down", "crawl", "balance_stand", "locked"):
        assert not posture_can_move(posture), posture


def test_unknown_posture_is_not_movable():
    """No state yet must never mean 'go ahead'."""
    for value in (None, "", "   "):
        assert not posture_can_move(value), value


def test_case_and_prefix_are_normalised():
    assert normalise_posture("WALK") == "walk"
    assert normalise_posture("  Walk  ") == "walk"
    assert normalise_posture("MotionStatus.WALK") == "walk"
    assert normalise_posture("CrawlWalk") == "crawlwalk"
    assert posture_can_move("WALK")
    assert posture_can_move("CrawlWalk")
    assert posture_can_move("MotionStatus.STAIR")
    assert not posture_can_move("MotionStatus.LIE_DOWN")


def test_every_movable_posture_round_trips_through_normalisation():
    for posture in MOVABLE_POSTURES:
        assert posture_can_move(posture.upper())
        assert posture_can_move(posture.replace("_", ""))


# ----------------------------------------------------- machine status & stale
def test_machine_status_requires_remote():
    assert normalise_machine_status("REMOTE") == "remote"
    assert normalise_machine_status("MachineStatus.REMOTE") == "remote"
    assert machine_status_can_move("remote")
    assert machine_status_can_move("REMOTE")
    for status in (None, "", "idle", "charging", "unknown"):
        assert not machine_status_can_move(status), status
        assert select_command(**arguments(machine_status=status)) is None


def test_stale_status_fails_closed():
    assert select_command(**arguments(status_stale=True)) is None


# ------------------------------------------------------------- e-stop gates
def test_estop_beats_everything():
    """Both soft and hardware e-stops outrank a fresh, in-posture command."""
    assert select_command(**arguments(estop_engaged=True)) is None
    assert select_command(**arguments(hardware_estop_engaged=True)) is None
    assert (
        select_command(
            **arguments(estop_engaged=True, source="auto", auto_command=(1.0, 0.5))
        )
        is None
    )
    assert (
        select_command(
            **arguments(
                hardware_estop_engaged=True, source="auto", auto_command=(1.0, 0.5)
            )
        )
        is None
    )


def test_estop_latches_even_with_movable_posture():
    for posture in MOVABLE_POSTURES:
        assert select_command(**arguments(estop_engaged=True, posture=posture)) is None
        assert (
            select_command(**arguments(hardware_estop_engaged=True, posture=posture))
            is None
        )


# ------------------------------------------------------- source + watchdog
def test_manual_command_requires_movable_posture_and_fresh_input():
    assert select_command(**arguments()) == (0.4, -0.3)
    assert select_command(**arguments(posture="lie_down")) is None
    assert select_command(**arguments(posture="locked")) is None
    assert select_command(**arguments(manual_age_s=0.36)) is None


def test_auto_command_is_forwarded_without_clamping():
    assert select_command(
        **arguments(source="auto", auto_command=(2.0, -3.0))
    ) == (2.0, -3.0)


def test_disabled_or_non_finite_command_is_rejected():
    assert select_command(**arguments(source="disabled")) is None
    assert select_command(**arguments(source="")) is None
    assert select_command(**arguments(manual_command=(float("nan"), 0.0))) is None
    assert select_command(**arguments(manual_command=(0.0, float("inf")))) is None
    assert (
        select_command(**arguments(source="auto", auto_command=(float("-inf"), 0.0)))
        is None
    )


def test_stale_command_from_inactive_source_does_not_leak_through():
    """A fresh manual command must not be used while auto is selected."""
    assert (
        select_command(**arguments(source="auto", auto_age_s=0.9, watchdog_s=0.35))
        is None
    )


# ----------------------------------------------------------------- clamping
def test_clamping_preserves_sign_and_caps_magnitude():
    assert select_command(
        **arguments(manual_command=(5.0, -9.0), max_linear=0.3, max_angular=1.0)
    ) == (0.3, -1.0)
    assert select_command(
        **arguments(manual_command=(-5.0, 9.0), max_linear=0.3, max_angular=1.0)
    ) == (-0.3, 1.0)


def test_clamping_leaves_small_commands_untouched():
    assert select_command(
        **arguments(manual_command=(0.1, 0.2), max_linear=0.5, max_angular=1.0)
    ) == (0.1, 0.2)


def test_zero_limit_disables_clamping():
    assert select_command(
        **arguments(manual_command=(5.0, -9.0), max_linear=0.0, max_angular=0.0)
    ) == (5.0, -9.0)


# ---------------------------------------------------------------- stop_reason
def test_stop_reason_explains_the_active_gate():
    assert (
        stop_reason(
            posture="walk",
            estop_engaged=False,
            hardware_estop_engaged=True,
            machine_status="remote",
            last_status_age_s=0.0,
            status_timeout_s=1.0,
        )
        == "hardware e-stop engaged"
    )
    assert (
        stop_reason(
            posture="walk",
            estop_engaged=True,
            last_status_age_s=0.0,
            status_timeout_s=1.0,
        )
        == "software e-stop engaged"
    )
    assert (
        stop_reason(
            posture="walk",
            estop_engaged=False,
            last_status_age_s=2.0,
            status_timeout_s=1.0,
        )
        == "robot_control/state is stale"
    )
    assert "idle" in stop_reason(
        posture="walk",
        estop_engaged=False,
        machine_status="idle",
        last_status_age_s=0.0,
        status_timeout_s=1.0,
    )
    assert "unknown" in stop_reason(
        posture=None, estop_engaged=False, last_status_age_s=0.0, status_timeout_s=1.0
    )
    assert "lie_down" in stop_reason(
        posture="lie_down",
        estop_engaged=False,
        last_status_age_s=0.0,
        status_timeout_s=1.0,
    )
    assert (
        stop_reason(
            posture="walk",
            estop_engaged=False,
            machine_status="remote",
            last_status_age_s=0.1,
            status_timeout_s=1.0,
        )
        is None
    )


# ------------------------------------------------------------- integration
def test_watchdog_boundary_is_inclusive_at_limit():
    assert select_command(**arguments(manual_age_s=0.35)) == (0.4, -0.3)
    assert select_command(**arguments(manual_age_s=0.3500001)) is None


def test_finite_twist_rejects_inf_and_nan_only():
    from m1_adapter.safety import finite_twist

    assert finite_twist(0.0, 0.0)
    assert finite_twist(-1.5, 2.5)
    assert not finite_twist(math.inf, 0.0)
    assert not finite_twist(0.0, math.nan)


# ----------------------------------------------------------- RTSP JPEG helper
def test_encode_frame_jpeg_resizes_and_encodes():
    frame = np.full((1080, 1920, 3), 120, dtype=np.uint8)
    data = encode_frame_jpeg(frame, output_width=640, jpeg_quality=80)
    assert data is not None
    assert data[:2] == b"\xff\xd8"  # JPEG SOI marker


def test_encode_frame_jpeg_rejects_empty():
    assert encode_frame_jpeg(np.empty((0, 0, 3), dtype=np.uint8)) is None


def test_latest_frame_buffer_overwrites_stale_and_deduplicates():
    buf = LatestFrameBuffer()
    seq, frame, stamp = buf.take_if_newer(0)
    assert seq == 0 and frame is None and stamp is None

    f1 = np.zeros((4, 4, 3), dtype=np.uint8)
    f2 = np.ones((4, 4, 3), dtype=np.uint8)
    buf.put(f1, "t1")
    buf.put(f2, "t2")

    seq, frame, stamp = buf.take_if_newer(0)
    assert seq == 2 and stamp == "t2" and frame is f2

    # Same seq returns nothing until a newer frame arrives.
    seq2, frame2, stamp2 = buf.take_if_newer(seq)
    assert seq2 == 2 and frame2 is None and stamp2 is None


# --------------------------------------------------- SDK 0.2.3 emergency enum
def test_sdk_emergency_status_interpretation():
    # SDK 0.2.3: 1 = RECOVER (safe), 2 = STOP (engaged), 0 = UNKNOWN (fail-closed)
    assert not is_estop_engaged(1, "recover")
    assert not is_estop_engaged(1, "EMERGENCY_STATUS_RECOVER")
    assert not is_estop_engaged(1, "")
    assert not is_estop_engaged(0, "disengaged")
    assert is_estop_engaged(2, "stop")
    assert is_estop_engaged(2, "EMERGENCY_STATUS_STOP")
    assert is_estop_engaged(2, "")
    assert is_estop_engaged(0, "")
    assert is_estop_engaged(0, "unknown")
    assert is_estop_engaged(1, "engaged")
