"""Fail-closed joint trajectory planning for an injected IK/safety backend.

This module plans trajectories only. It does not connect to Piper Web or send
robot commands; a robot backend must provide IK and segment-safety validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence

from aple.schema import FailureCode

JointVector = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class ToolPose:
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


@dataclass(frozen=True)
class JointTrajectory:
    points_rad: tuple[JointVector, ...]
    waypoint_end_indices: tuple[int, ...]


class IKSolver(Protocol):
    def solve(self, pose: ToolPose, seed_rad: JointVector) -> Sequence[float] | None: ...


class SegmentValidator(Protocol):
    def is_segment_safe(self, start_rad: JointVector, end_rad: JointVector) -> bool: ...


class TrajectoryPlanningError(RuntimeError):
    def __init__(self, failure_code: FailureCode, message: str) -> None:
        super().__init__(message)
        self.failure_code = failure_code


class JointTrajectoryPlanner:
    """Convert Cartesian waypoints into checked, bounded joint-space steps."""

    def __init__(
        self,
        ik_solver: IKSolver,
        segment_validator: SegmentValidator,
        joint_limits_rad: Sequence[Sequence[float]],
        *,
        max_joint_step_rad: float,
    ) -> None:
        self._ik_solver = ik_solver
        self._segment_validator = segment_validator
        self._joint_limits = self._validate_limits(joint_limits_rad)
        self._max_joint_step_rad = self._positive_finite(
            max_joint_step_rad, "max_joint_step_rad"
        )

    def plan(
        self,
        waypoints: Sequence[ToolPose],
        start_joints_rad: Sequence[float],
    ) -> JointTrajectory:
        if not waypoints:
            raise ValueError("waypoints must contain at least one pose")

        current = self._validate_joints(start_joints_rad, "start_joints_rad")
        self._validate_within_limits(current, "start_joints_rad")
        points: list[JointVector] = []
        waypoint_end_indices: list[int] = []

        for index, pose in enumerate(waypoints):
            self._validate_pose(pose, index)
            solution = self._ik_solver.solve(pose, current)
            if solution is None:
                raise TrajectoryPlanningError(
                    FailureCode.IK_FAIL, f"IK failed for waypoint {index}"
                )
            goal = self._validate_joints(solution, f"IK solution for waypoint {index}")
            self._validate_within_limits(goal, f"IK solution for waypoint {index}")

            segment_count = max(
                1,
                math.ceil(max(abs(end - start) for start, end in zip(current, goal))
                          / self._max_joint_step_rad),
            )
            segment_start = current
            for step in range(1, segment_count + 1):
                fraction = step / segment_count
                point = tuple(
                    start + (end - start) * fraction
                    for start, end in zip(segment_start, goal)
                )
                next_point: JointVector = (
                    point[0], point[1], point[2], point[3], point[4], point[5]
                )
                if not self._segment_validator.is_segment_safe(current, next_point):
                    raise TrajectoryPlanningError(
                        FailureCode.COLLISION,
                        f"unsafe joint segment while approaching waypoint {index}",
                    )
                points.append(next_point)
                current = next_point
            waypoint_end_indices.append(len(points) - 1)

        return JointTrajectory(tuple(points), tuple(waypoint_end_indices))

    @classmethod
    def _validate_joints(cls, values: Sequence[float], name: str) -> JointVector:
        if len(values) != 6:
            raise ValueError(f"{name} must contain six joint angles")
        result = tuple(float(value) for value in values)
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{name} must contain only finite joint angles")
        return (result[0], result[1], result[2], result[3], result[4], result[5])

    @classmethod
    def _validate_limits(
        cls, limits: Sequence[Sequence[float]]
    ) -> tuple[tuple[float, float], ...]:
        if len(limits) != 6:
            raise ValueError("joint_limits_rad must contain six [lower, upper] pairs")
        validated: list[tuple[float, float]] = []
        for index, pair in enumerate(limits):
            if len(pair) != 2:
                raise ValueError(f"joint_limits_rad[{index}] must be a [lower, upper] pair")
            lower, upper = float(pair[0]), float(pair[1])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(f"joint_limits_rad[{index}] must be finite and increasing")
            validated.append((lower, upper))
        return tuple(validated)

    def _validate_within_limits(self, joints: JointVector, name: str) -> None:
        for index, (value, (lower, upper)) in enumerate(zip(joints, self._joint_limits)):
            if not lower <= value <= upper:
                raise TrajectoryPlanningError(
                    FailureCode.IK_FAIL,
                    f"{name} joint{index + 1} is outside configured limits",
                )

    @staticmethod
    def _validate_pose(pose: ToolPose, index: int) -> None:
        if len(pose.position_m) != 3 or not all(
            math.isfinite(float(value)) for value in pose.position_m
        ):
            raise ValueError(f"waypoints[{index}].position_m must contain three finite values")
        if len(pose.orientation_xyzw) != 4 or not all(
            math.isfinite(float(value)) for value in pose.orientation_xyzw
        ):
            raise ValueError(
                f"waypoints[{index}].orientation_xyzw must contain four finite values"
            )
        norm = math.sqrt(sum(float(value) ** 2 for value in pose.orientation_xyzw))
        if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"waypoints[{index}].orientation_xyzw must be a unit quaternion")

    @staticmethod
    def _positive_finite(value: float, name: str) -> float:
        result = float(value)
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be finite and positive")
        return result
