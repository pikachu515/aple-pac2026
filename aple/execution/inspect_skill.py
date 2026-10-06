"""Build deterministic in-hand inspection poses from measured apple-in-TCP data."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from aple.execution.goal_builder import Pose3D, compose, inverse


@dataclass(frozen=True)
class InspectionPlan:
    target_tool_poses: tuple[Pose3D, ...]
    settle_time_s: float


def build_inspection_plan(
    apple_in_tcp: Pose3D,
    apple_center_in_base: tuple[float, float, float],
    *,
    tilt_rad: float,
    rotation_angles_rad: Sequence[float],
    settle_time_s: float,
) -> InspectionPlan:
    """Rotate the held apple around its center and return one tool pose per view."""
    if len(apple_center_in_base) != 3 or not all(
        math.isfinite(value) for value in apple_center_in_base
    ):
        raise ValueError("apple_center_in_base must contain three finite coordinates")
    if not math.isfinite(tilt_rad):
        raise ValueError("tilt_rad must be finite")
    if not math.isfinite(settle_time_s) or settle_time_s < 0:
        raise ValueError("settle_time_s must be finite and non-negative")
    if not rotation_angles_rad:
        raise ValueError("rotation_angles_rad must contain at least one view angle")
    if not all(math.isfinite(angle) for angle in rotation_angles_rad):
        raise ValueError("rotation_angles_rad must contain finite angles")

    apple_center_pose = Pose3D(
        apple_center_in_base,
        (0.0, 0.0, 0.0, 1.0),
    )
    tcp_from_apple_inverse = inverse(apple_in_tcp)
    target_poses = tuple(
        compose(
            Pose3D(
                apple_center_pose.position_m,
                _multiply(
                    _axis_angle((0.0, 0.0, 1.0), angle),
                    _axis_angle((0.0, 1.0, 0.0), tilt_rad),
                ),
            ),
            tcp_from_apple_inverse,
        )
        for angle in rotation_angles_rad
    )
    return InspectionPlan(target_poses, settle_time_s)


def _axis_angle(
    axis: tuple[float, float, float], angle_rad: float
) -> tuple[float, float, float, float]:
    half_angle = angle_rad / 2.0
    sine = math.sin(half_angle)
    return (
        axis[0] * sine,
        axis[1] * sine,
        axis[2] * sine,
        math.cos(half_angle),
    )


def _multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )
