"""Safety-rule tests for the M1 adapter. Run without ROS, without a robot."""

import math

import pytest

from m1_adapter.safety import (
    MOVABLE_POSTURES,
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


# ------------------------------------------------------------- e-stop gate
def test_estop_beats_everything():
    """The operator's brake outranks a fresh, in-posture command."""
    assert select_command(**arguments(estop_engaged=True)) is None
    assert select_command(
        **arguments(estop_engaged=True, source="auto", auto_command=(1.0, 0.5))
    ) is None


def test_estop_latches_even_with_movable_posture():
    for posture in MOVABLE_POSTURES:
        assert select_command(**arguments(estop_engaged=True, posture=posture)) is None


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
    assert select_command(
        **arguments(source="auto", auto_command=(float("-inf"), 0.0))
    ) is None


def test_stale_command_from_inactive_source_does_not_leak_through():
    """A fresh manual command must not be used while auto is selected."""
    assert select_command(
        **arguments(source="auto", auto_age_s=0.9, watchdog_s=0.35)
    ) is None


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
    assert stop_reason(
        posture="walk", estop_engaged=True, last_status_age_s=0.0, status_timeout_s=1.0
    ) == "software e-stop engaged"
    assert (
        stop_reason(
            posture="walk",
            estop_engaged=False,
            last_status_age_s=2.0,
            status_timeout_s=1.0,
        )
        == "robot_control/state is stale"
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
