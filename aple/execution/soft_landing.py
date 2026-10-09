"""Deterministic Phase-B descent, release, and retreat targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from aple.execution.goal_builder import MotionTargets, PlaceTargets, Pose3D


class GripperAction(str, Enum):
    HOLD = "hold"
    RELEASE = "release"


@dataclass(frozen=True)
class SoftLandingStep:
    pose: Pose3D
    speed_scale: float
    gripper_action: GripperAction


@dataclass(frozen=True)
class SoftLandingPlan:
    steps: tuple[SoftLandingStep, ...]
    slow_descent_start_z_m: float


def build_soft_landing_plan(
    targets: MotionTargets | PlaceTargets,
    *,
    final_descent_m: float,
    slow_speed_scale: float,
    approach_speed_scale: float,
) -> SoftLandingPlan:
    """Create a hover-to-release-to-hover sequence with a bounded slow segment."""
    values = (final_descent_m, slow_speed_scale, approach_speed_scale)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("soft-landing distances and speed scales must be finite")
    if final_descent_m <= 0:
        raise ValueError("final_descent_m must be positive")
    if not 0 < slow_speed_scale <= 1 or not 0 < approach_speed_scale <= 1:
        raise ValueError("speed scales must be in (0, 1]")
    if targets.hover_pose.position_m[2] < targets.place_pose.position_m[2]:
        raise ValueError("hover pose must not be below the release pose")

    place_z = targets.place_pose.position_m[2]
    hover_z = targets.hover_pose.position_m[2]
    slow_start_z = min(hover_z, place_z + final_descent_m)
    slow_start_pose = Pose3D(
        (
            targets.place_pose.position_m[0],
            targets.place_pose.position_m[1],
            slow_start_z,
        ),
        targets.place_pose.orientation_xyzw,
    )
    steps = (
        SoftLandingStep(slow_start_pose, approach_speed_scale, GripperAction.HOLD),
        SoftLandingStep(targets.place_pose, slow_speed_scale, GripperAction.RELEASE),
        SoftLandingStep(targets.hover_pose, approach_speed_scale, GripperAction.HOLD),
    )
    return SoftLandingPlan(steps, slow_start_z)
