"""Pure command-selection helpers for the Genisom M1 adapter.

No ROS imports here on purpose: every rule in this file is a safety rule, and
safety rules must be testable without a robot, a DDS graph, or a venv that can
import rclpy.  ``test/test_safety.py`` exercises all of it.

WHY THIS DIFFERS FROM ``tron_adapter/safety.py``
    The TRON adapter gates on one string: its vendor status must be ``"WALK"``.
    The M1 (SDK 0.2.3 / ``m1_robot_driver``) enforces several independent gates,
    all of which must pass:

      1. HARDWARE / SOFT E-STOP -- operator brakes outrank everything.
      2. STATUS FRESHNESS       -- stale or missing status never permits motion.
      3. MACHINE STATUS         -- SDK 0.2.3 requires ``remote`` mode.
      4. POSTURE                -- only documented movable postures accept Move.
      5. SOURCE + WATCHDOG      -- whose command is it, and is it fresh?

    Any one of them failing yields zero velocity.  They are kept separate
    because they fail for different reasons and the diagnostics should say
    which one.
"""

from __future__ import annotations

import math
from typing import Optional

Twist2D = tuple[float, float]

# ---------------------------------------------------------------------------
# posture & machine-status gates
# ---------------------------------------------------------------------------
# From genisom_robot_sdk/docs/zh/sdk_state_zh.md, the 移动能力 column: these are
# the postures in which Move()/MoveVelocity() is accepted.  Everything else
# (lie_down 卧倒, crawl 匍匐, locked 锁定, balance_stand 平衡站立) either ignores
# a velocity command or, worse, silently transitions the robot.
#
# Names are LOWERCASE snake_case because that is what m1_robot_driver publishes
# in RobotStatus.motion_status_name.  The Genisom SDK's own MotionStatus enum is
# UPPERCASE (WALK, STAIR, ...); normalise_posture() accepts either spelling so
# this keeps working if we ever read the SDK directly.
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
UNKNOWN_POSTURE = ""

# SDK 0.2.3 m1_robot_driver only accepts motion commands when machine_status is
# `remote`.  Everything else (`idle`, `charging`, `unknown`, empty) holds zero.
MOVABLE_MACHINE_STATUSES = frozenset({"remote"})

# SDK 0.2.3 EmergencyStatus enum: 0=UNKNOWN, 1=RECOVER (safe), 2=STOP (engaged).
ESTOP_RECOVERED_NAMES = frozenset(
    {
        "recover",
        "emergency_status_recover",
        "disengaged",
        "released",
        "normal",
        "ok",
        "off",
    }
)
ESTOP_ENGAGED_NAMES = frozenset(
    {
        "stop",
        "emergency_status_stop",
        "engaged",
        "triggered",
        "active",
        "on",
    }
)


def is_estop_engaged(status_code: int, status_name: Optional[str]) -> bool:
    """Interpret RobotStatus software/hardware emergency fields (fail-closed).

    In Genisom SDK 0.2.3, ``1`` is ``EMERGENCY_STATUS_RECOVER`` (``"recover"``,
    safe/released), ``2`` is ``EMERGENCY_STATUS_STOP`` (``"stop"``, engaged),
    and ``0`` is unknown.  Any unknown or ambiguous state fails closed (True).
    """
    name = normalise_posture(status_name)
    if name in ESTOP_RECOVERED_NAMES:
        return False
    if name in ESTOP_ENGAGED_NAMES:
        return True
    if name:
        return True
    return int(status_code) != 1


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


def normalise_machine_status(name: Optional[str]) -> str:
    """Normalise ``RobotStatus.machine_status_name`` to lowercase token."""
    return normalise_posture(name)


def posture_can_move(name: Optional[str]) -> bool:
    """Does the current posture accept velocity commands?

    Deliberately accepts BOTH ``crawl_walk`` and ``crawlwalk``: the SDK docs use
    snake_case, the enum uses CamelCase, and normalise_posture() lowercases
    without inserting underscores.
    """
    value = normalise_posture(name)
    if not value:
        return False
    if value in MOVABLE_POSTURES:
        return True
    # camel-case enum spelling that lost its underscore
    return value in {p.replace("_", "") for p in MOVABLE_POSTURES}


def machine_status_can_move(name: Optional[str]) -> bool:
    """Does the current machine state accept velocity commands (SDK 0.2.3)?"""
    value = normalise_machine_status(name)
    return bool(value) and value in MOVABLE_MACHINE_STATUSES


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
    hardware_estop_engaged: bool = False,
    machine_status: Optional[str] = "remote",
    status_stale: bool = False,
) -> Optional[Twist2D]:
    """Return the safe active command, or None when output must stop.

    Gate order matters for diagnostics and for safety:

      * hardware & software e-stops FIRST.
      * status freshness next (stale telemetry must fail closed).
      * machine_status == 'remote' next (SDK 0.2.3 contract).
      * posture next.
      * source/watchdog last, because that is the normal running path.

    ``max_linear``/``max_angular`` of 0 disable clamping.  When set, they clamp
    per-axis and preserve sign.
    """
    if hardware_estop_engaged or estop_engaged:
        return None
    if status_stale:
        return None
    if not machine_status_can_move(machine_status):
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
    hardware_estop_engaged: bool = False,
    machine_status: Optional[str] = "remote",
) -> Optional[str]:
    """Human-readable reason the robot is being held at zero, or None.

    Only used for diagnostics/logging; it never decides anything.
    """
    if hardware_estop_engaged:
        return "hardware e-stop engaged"
    if estop_engaged:
        return "software e-stop engaged"
    if last_status_age_s > status_timeout_s:
        return "robot_control/state is stale"
    mval = normalise_machine_status(machine_status)
    if not mval:
        return "machine status unknown (no state received yet)"
    if not machine_status_can_move(mval):
        return f"machine status {mval!r} is not 'remote'"
    value = normalise_posture(posture)
    if not value:
        return "posture unknown (no state received yet)"
    if not posture_can_move(value):
        return f"posture {value!r} does not accept velocity commands"
    return None
