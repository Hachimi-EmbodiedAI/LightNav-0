"""Pure command-selection helpers for the Genisom M1 adapter.

No ROS imports here on purpose: every rule in this file is a safety rule, and
safety rules must be testable without a robot, a DDS graph, or a venv that can
import rclpy.  ``test/test_safety.py`` exercises all of it.

WHY THIS DIFFERS FROM ``tron_adapter/safety.py``
    The TRON adapter gates on one string: its vendor status must be ``"WALK"``.
    The M1 needs more, because the Genisom SDK refuses or silently drops
    velocity commands in several postures, and because the robot has its own
    software emergency stop that outranks anything this node wants to do.

    Concretely, three independent gates, all of which must pass:

      1. SOURCE + WATCHDOG   -- whose command is it, and is it fresh?
      2. POSTURE             -- can this posture move at all?
      3. SOFT E-STOP         -- has the operator braked the robot?

    Any one of them failing yields zero velocity.  They are kept separate
    because they fail for different reasons and the diagnostics should say
    which one.
"""

from __future__ import annotations

import math
from typing import Optional

Twist2D = tuple[float, float]

# ---------------------------------------------------------------------------
# posture gate
# ---------------------------------------------------------------------------
# From genisom_robot_sdk/docs/zh/sdk_state_zh.md, the 移动能力 column: these are
# the postures in which Move()/MoveVelocity() is accepted.  Everything else
# (lie_down 卧倒, crawl 匍匐, locked 锁定, balance_stand 平衡站立) either ignores
# a velocity command or, worse, silently transitions the robot.
#
# Names are LOWERCASE snake_case because that is what robot_control publishes in
# RobotStatus.motion_status_name.  The Genisom SDK's own MotionStatus enum is
# UPPERCASE (WALK, STAIR, ...); normalise_posture() accepts either spelling so
# this keeps working if we ever read the SDK directly.
#
# Verified on hardware by hachimi_ws (M1, SN 04A016610100737, system 0.3.3):
# robot_control emits lowercase, and a dog resting in crawl_walk reported
# "当前姿态未知" until crawl_walk was included here.
MOVABLE_POSTURES = frozenset(
    {
        "walk",         # also what StandUp reports once it completes
        "stand_up",     # the transitional report; harmless to accept
        "stair",
        "crawl_walk",
        "climb",
        "gait",
        "slim",
        "sk_walk",
        "sand",
    }
)

# Postures that are legitimate resting states and should NOT be reported as a
# fault.  Used only for diagnostics wording.
RESTING_POSTURES = frozenset({"lie_down", "crawl", "locked", "balance_stand"})

# Empty/None means "we have not received state yet" -- never allow motion then.
# hachimi_ws learned this the hard way: it is not the same as "not movable".
UNKNOWN_POSTURE = ""


def normalise_posture(name: Optional[str]) -> str:
    """Lowercase, strip, and drop a vendor prefix if one sneaks in.

    ``"WALK"`` -> ``"walk"``, ``"CrawlWalk"`` -> ``"crawlwalk"``,
    ``"MotionStatus.WALK"`` -> ``"walk"``.
    """
    if not name:
        return UNKNOWN_POSTURE
    value = str(name).strip()
    if "." in value:
        value = value.rsplit(".", 1)[-1]
    return value.lower()


def posture_can_move(name: Optional[str]) -> bool:
    """Does the current posture accept velocity commands?

    Deliberately accepts BOTH ``crawl_walk`` and ``crawlwalk``: the SDK docs use
    snake_case, the enum uses CamelCase, and normalise_posture() lowercases
    without inserting underscores.  Missing this cost real debugging time in
    hachimi_ws.
    """
    value = normalise_posture(name)
    if not value:
        return False
    if value in MOVABLE_POSTURES:
        return True
    # camel-case enum spelling that lost its underscore
    return value in {p.replace("_", "") for p in MOVABLE_POSTURES}


# ---------------------------------------------------------------------------
# command selection
# ---------------------------------------------------------------------------
def finite_twist(linear: float, angular: float) -> bool:
    return math.isfinite(linear) and math.isfinite(angular)


def select_command(
    *,
    source: str,
    posture: Optional[str],
    estop_engaged: bool,
    manual_command: Twist2D,
    manual_age_s: float,
    auto_command: Twist2D,
    auto_age_s: float,
    watchdog_s: float,
    max_linear: float = 0.0,
    max_angular: float = 0.0,
) -> Optional[Twist2D]:
    """Return the safe active command, or None when output must stop.

    Gate order matters for diagnostics and for safety:

      * soft e-stop is checked FIRST.  It is the operator's brake and must not
        be defeatable by a posture that happens to look movable, or by a fresh
        command arriving at the same instant.
      * posture next.  A fresh command in a non-movable posture is not a
        reason to move, it is a reason to stop.
      * source/watchdog last, because that is the normal running path.

    ``max_linear``/``max_angular`` of 0 disable clamping (kept 0 by default so
    the pure tests stay about gating, not about limits).  When set, they clamp
    per-axis and preserve sign -- clamp, do not reject, so the operator gets a
    predictable command rather than a silent no-op.
    """
    if estop_engaged:
        return None
    if not posture_can_move(posture):
        return None

    if source == "manual" and manual_age_s <= watchdog_s:
        command: Optional[Twist2D] = manual_command
    elif source == "auto" and auto_age_s <= watchdog_s:
        command = auto_command
    else:
        return None

    if command is None or not finite_twist(*command):
        return None

    linear, angular = command
    if max_linear > 0.0:
        linear = max(-max_linear, min(max_linear, linear))
    if max_angular > 0.0:
        angular = max(-max_angular, min(max_angular, angular))
    return (linear, angular)


def stop_reason(
    *,
    posture: Optional[str],
    estop_engaged: bool,
    last_status_age_s: float,
    status_timeout_s: float,
) -> Optional[str]:
    """Human-readable reason the robot is being held at zero, or None.

    Only used for diagnostics/logging; it never decides anything.
    """
    if estop_engaged:
        return "software e-stop engaged"
    if last_status_age_s > status_timeout_s:
        return "robot_control/state is stale"
    value = normalise_posture(posture)
    if not value:
        return "posture unknown (no state received yet)"
    if not posture_can_move(value):
        return f"posture {value!r} does not accept velocity commands"
    return None
