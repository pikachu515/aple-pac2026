"""Fail-closed joint trajectory planning for an injected IK/safety backend.

This module plans trajectories only. It does not connect to Piper Web or send
robot commands; a robot backend must provide IK and segment-safety validation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Protocol, Sequence

import numpy as np

from aple.schema import FailureCode
from aple.execution.goal_builder import Pose3D

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


class PiperIKFunction(Protocol):
    def __call__(
        self, target: np.ndarray, seed: np.ndarray
    ) -> Mapping[str, object]: ...


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

    def plan_joint_goal(self, goal_rad, start_joints_rad) -> JointTrajectory:
        current = self._validate_joints(start_joints_rad, "start_joints_rad")
        goal = self._validate_joints(goal_rad, "goal_rad")
        self._validate_within_limits(current, "start_joints_rad")
        self._validate_within_limits(goal, "goal_rad")
        count = max(1, math.ceil(max(abs(b - a) for a, b in zip(current, goal))
                                 / self._max_joint_step_rad))
        points = []
        for step in range(1, count + 1):
            point = tuple(a + (b - a) * step / count for a, b in zip(start_joints_rad, goal))
            if not self._segment_validator.is_segment_safe(current, point):
                raise TrajectoryPlanningError(FailureCode.COLLISION, "unsafe named-pose segment")
            points.append(point)
            current = point
        return JointTrajectory(tuple(points), (len(points) - 1,))

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


class PiperKinematicsIKSolver:
    """Adapt Piper Studio's shared numerical IK function to ``IKSolver``."""

    def __init__(self, ik_function: PiperIKFunction | None = None, *,
                 tcp_transform: PiperTCPTransform | None = None) -> None:
        if not isinstance(tcp_transform, PiperTCPTransform):
            raise ValueError("a measured workcell.tcp_offset_m transform is required")
        self.tcp_transform = tcp_transform
        self._ik_function = (
            ik_function if ik_function is not None else self._load_ik_function()
        )

    def solve(self, pose: ToolPose, seed_rad: JointVector) -> JointVector | None:
        JointTrajectoryPlanner._validate_pose(pose, 0)
        seed = JointTrajectoryPlanner._validate_joints(seed_rad, "seed_rad")
        target = np.eye(4, dtype=float)
        x, y, z, w = pose.orientation_xyzw
        target[:3, :3] = (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
        target[:3, 3] = pose.position_m
        target = target @ self.tcp_transform.inverse_matrix()

        result = self._ik_function(target, np.asarray(seed, dtype=float))
        if not isinstance(result, Mapping):
            raise RuntimeError("Piper IK returned a non-mapping result")
        if result.get("ok") is not True:
            return None
        if "q" not in result:
            raise RuntimeError("Piper IK reported success without a joint solution")

        joints = np.asarray(result["q"], dtype=float)
        if joints.shape != (6,) or not np.isfinite(joints).all():
            raise RuntimeError("Piper IK returned an invalid six-joint solution")
        return (
            float(joints[0]),
            float(joints[1]),
            float(joints[2]),
            float(joints[3]),
            float(joints[4]),
            float(joints[5]),
        )

    @staticmethod
    def _load_ik_function() -> PiperIKFunction:
        try:
            kinematics = import_module("piper_robot.kinematics")
        except ImportError as exc:
            raise RuntimeError(
                "Piper's shared piper_robot.kinematics module is unavailable"
            ) from exc
        ik_function = getattr(kinematics, "ik", None)
        if not callable(ik_function):
            raise RuntimeError("Piper's kinematics module does not expose ik()")
        return ik_function


@dataclass(frozen=True)
class PiperTCPTransform:
    """Measured flange-frame translation in meters; TCP axes match flange axes."""

    offset_m: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.offset_m is None:
            raise ValueError("workcell.tcp_offset_m must be measured before using Piper IK/FK")
        offset = np.asarray(self.offset_m, dtype=float)
        if offset.shape != (3,) or not np.isfinite(offset).all():
            raise ValueError("workcell.tcp_offset_m must contain three finite meters")
        object.__setattr__(self, "offset_m", tuple(float(v) for v in offset))

    @classmethod
    def from_config(cls, cfg: dict) -> PiperTCPTransform:
        from aple.execution.gripper_geometry import apply_attachment_geometry
        cfg = apply_attachment_geometry(cfg)
        return cls(cfg.get("workcell", {}).get("tcp_offset_m"))

    def matrix(self) -> np.ndarray:
        result = np.eye(4)
        result[:3, 3] = self.offset_m
        return result

    def inverse_matrix(self) -> np.ndarray:
        result = np.eye(4)
        result[:3, 3] = -np.asarray(self.offset_m)
        return result


class PiperKinematicsFKProvider:
    """Callable tool_pose_provider: six measured radians → base_link TCP Pose3D.

    Piper's link_transforms returns a batch (1, links, 4, 4). Never apply
    norm_to_rad here: PiperStudio.joint_state() has already converted raw mdeg.
    """

    def __init__(self, tcp_transform: PiperTCPTransform, *,
                 link_transforms, tip_index: int) -> None:
        if not isinstance(tcp_transform, PiperTCPTransform):
            raise ValueError("a measured workcell.tcp_offset_m transform is required")
        if not callable(link_transforms):
            raise TypeError("link_transforms must be callable")
        if isinstance(tip_index, bool) or not isinstance(tip_index, int) or tip_index < 0:
            raise ValueError("tip_index must be a non-negative integer")
        self.tcp_transform = tcp_transform
        self._link_transforms = link_transforms
        self._tip_index = tip_index

    def __call__(self, joints_rad: Sequence[float]) -> Pose3D:
        joints = JointTrajectoryPlanner._validate_joints(joints_rad, "joints_rad")
        links = np.asarray(self._link_transforms(np.asarray(joints)[None, :]), dtype=float)
        if (links.ndim != 4 or links.shape[0] != 1 or links.shape[2:] != (4, 4)
                or self._tip_index >= links.shape[1]):
            raise RuntimeError("Piper FK must return (1, links, 4, 4) including the flange tip")
        flange = links[0, self._tip_index]
        R = flange[:3, :3]
        if (not np.isfinite(flange).all()
                or not np.allclose(flange[3], (0, 0, 0, 1), atol=1e-6, rtol=0)
                or not np.allclose(R.T @ R, np.eye(3), atol=1e-6, rtol=0)
                or not math.isclose(float(np.linalg.det(R)), 1.0, abs_tol=1e-6)):
            raise RuntimeError("Piper FK returned an invalid rigid flange transform")
        tcp = flange @ self.tcp_transform.matrix()
        return Pose3D(tuple(float(v) for v in tcp[:3, 3]), _rotation_quaternion(R))


def _rotation_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a validated rotation, including 180-degree rotations, to xyzw."""
    trace = float(np.trace(R))
    if trace > 0:
        s = 2 * math.sqrt(1 + trace)
        q = np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                      (R[1, 0] - R[0, 1]) / s, s / 4])
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = np.empty(4)
        q[i], q[j], q[k], q[3] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s, (R[k, j] - R[j, k]) / s
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def build_piper_kinematics(cfg: dict, *, kinematics_module=None):
    """Create IK and a shared FK callable with one measured offset and geometry.

    Pass the returned FK object unchanged to both tool_pose_provider and the
    in-hand provider's tool_pose_fn. This loads math only, never robot devices.
    """
    transform = PiperTCPTransform.from_config(cfg)  # fail before loading runtime on null
    module = kinematics_module or import_module("piper_robot.kinematics")
    geometry = module.geometry()
    if geometry.tip != "link6":
        raise ValueError("Piper TCP offset requires link6 flange geometry")
    ik = PiperKinematicsIKSolver(
        lambda target, seed: module.ik(target, seed, geom=geometry), tcp_transform=transform,
    )
    fk = PiperKinematicsFKProvider(
        transform, link_transforms=lambda q: module.link_transforms(q, geometry),
        tip_index=geometry.tip_index,
    )
    return ik, fk
