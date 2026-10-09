"""Fail-closed pick/place orchestration behind a hardware-independent motion adapter.

The adapter is deliberately injected: the downloaded project does not contain a
verified Piper Studio command/IK API. No method in this module opens hardware.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol

from aple.execution.gripper import effort_for
from aple.execution.goal_builder import (
    MotionTargets,
    PickTargets,
    PlaceTargets,
    Pose3D,
    build_pick_targets,
    build_place_targets,
    build_targets,
    compose,
    relative_pose,
)
from aple.execution.inspect_skill import InspectionPlan, build_inspection_plan
from aple.execution.soft_landing import SoftLandingPlan, build_soft_landing_plan
from aple.inspection.in_hand_capture import InHandCapture
from aple.schema import (AppleState, ExecResult, FailureCode, GraspProfile, Phase,
                         PlacementPlan)


class GraspCandidate(Protocol):
    """Structural view of the M1 grasp-candidate contract consumed by role B."""

    scene_id: int
    apple_id: int
    grasp_pose: Pose3D
    grasp_yaw_rad: float
    approach_vector: tuple[float, float, float]
    approach_height_m: float
    min_clearance_m: float
    collision_free: bool
    score: float


@dataclass(frozen=True)
class PhaseAResult:
    """Evidence returned by a backend after its approach/grasp/transfer phase."""

    success: bool
    failure_code: FailureCode
    phase_reached: Phase
    holding: bool
    hover_reached: bool


@dataclass(frozen=True)
class PhaseBResult:
    """Evidence returned after slow descent, release, and retreat."""

    success: bool
    phase_reached: Phase


@dataclass(frozen=True)
class PickPhaseResult:
    success: bool
    failure_code: FailureCode
    phase_reached: Phase
    holding: bool
    hover_reached: bool
    tool_pose_at_grasp_base: Pose3D | None
    tool_pose_at_hover_base: Pose3D | None


class ViewVerdict(Enum):
    """Per-view in-hand check result, decided right after each capture."""

    OK = "ok"
    UNVERIFIED = "unverified"  # e.g. too few depth points: retake once, then treat as slip
    SLIP = "slip"


# (view_index, capture) -> verdict. The HAL calls it after every capture while the apple is held.
ViewCheck = Callable[[int, InHandCapture], ViewVerdict]


@dataclass(frozen=True)
class InspectionPhaseResult:
    success: bool
    captured_views: int
    captures: tuple[InHandCapture, ...] = ()
    slip_view_index: int | None = None


@dataclass(frozen=True)
class PickExecutionResult:
    success: bool
    failure_code: FailureCode
    phase_reached: Phase
    apple_in_tcp: Pose3D | None = None
    apple_center_in_base: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class InspectionExecutionResult:
    success: bool
    failure_code: FailureCode
    phase_reached: Phase
    captured_views: int
    captures: tuple[InHandCapture, ...] = ()
    slip_view_index: int | None = None


class MotionAdapter(Protocol):
    """Hardware boundary. Real adapters must implement and validate these calls."""

    def move_named(
        self, name: str, joint_target_rad: tuple[float, ...], *, speed_scale: float, timeout_s: float
    ) -> bool: ...

    def execute_phase_a(
        self,
        apple: AppleState,
        profile: GraspProfile,
        candidate: GraspCandidate,
        plan: PlacementPlan,
        targets: MotionTargets,
        *,
        backend: str,
        effort: int,
        task_text: str,
        speed_scale: float,
        timeout_s: float,
    ) -> PhaseAResult: ...

    def execute_phase_b(
        self,
        apple: AppleState,
        profile: GraspProfile,
        plan: PlacementPlan,
        targets: MotionTargets | PlaceTargets,
        landing_plan: SoftLandingPlan,
        *,
        effort: int,
        final_descent_m: float,
        slow_speed_scale: float,
        timeout_s: float,
    ) -> PhaseBResult: ...

    def execute_pick(
        self,
        apple: AppleState,
        profile: GraspProfile,
        candidate: GraspCandidate,
        targets: PickTargets,
        *,
        backend: str,
        effort: int,
        task_text: str,
        speed_scale: float,
        timeout_s: float,
    ) -> PickPhaseResult: ...

    def execute_place_phase_a(
        self,
        apple: AppleState,
        profile: GraspProfile,
        placement: PlacementPlan,
        targets: PlaceTargets,
        *,
        backend: str,
        effort: int,
        task_text: str,
        speed_scale: float,
        timeout_s: float,
    ) -> PhaseAResult: ...

    def inspect_views(
        self,
        profile: GraspProfile,
        inspection_plan: InspectionPlan,
        *,
        effort: int,
        timeout_s: float,
        view_check: ViewCheck | None = None,
    ) -> InspectionPhaseResult: ...

    def inspect(
        self,
        profile: GraspProfile,
        inspection_plan: InspectionPlan,
        *,
        effort: int,
        timeout_s: float,
    ) -> bool: ...


class MotionExecutor:
    def __init__(self, hal: MotionAdapter, cfg: dict) -> None:
        self.hal = hal
        self.cfg = cfg
        self.stage_timings = {}

    def _timed_call(self, stage, call, *args, **kwargs):
        started = time.monotonic()
        try:
            return call(*args, **kwargs)
        finally:
            self.stage_timings[stage] = self.stage_timings.get(stage, 0.0) + time.monotonic()-started

    def move_named(self, name: str) -> ExecResult:
        if name not in ("home", "scan"):
            raise ValueError("name must be 'home' or 'scan'")
        joint_poses = self.cfg["workcell"]["poses"]
        key = f"{name}_joint_rad"
        joints = joint_poses.get(key)
        if joints is None:
            return ExecResult(False, FailureCode.ABORTED, Phase.MOVE_NAMED)
        if len(joints) != 6 or not all(math.isfinite(float(value)) for value in joints):
            raise ValueError(f"workcell.poses.{key} must contain six finite joint angles")

        timeout_s = self._positive_timeout("named_pose_s")
        speed = self._speed_scale("speed_scale_max")
        success = self.hal.move_named(name, tuple(float(value) for value in joints),
                                      speed_scale=speed, timeout_s=timeout_s)
        return ExecResult(success, FailureCode.NONE if success else FailureCode.TIMEOUT,
                          Phase.MOVE_NAMED)

    def pick_place(self, apple: AppleState, profile: GraspProfile,
                   candidate: GraspCandidate, plan: PlacementPlan,
                   backend: str | None = None) -> ExecResult:
        self.stage_timings = {}
        execution_cfg = self.cfg["execution"]
        if backend is None:
            backend = execution_cfg.get("default_backend", "scripted")
        if backend not in ("scripted", "act"):
            raise ValueError("backend must be 'scripted' or 'act'")
        self._validate_match(apple, profile, candidate, plan)
        if not profile.feasible or not plan.feasible:
            return ExecResult(False, FailureCode.ABORTED, Phase.IDLE)
        if plan.planned_drop_m < 0:
            raise ValueError("plan.planned_drop_m must be non-negative")
        max_drop_m = float(self.cfg["workcell"]["placement"]["max_drop_m"])
        if not math.isfinite(max_drop_m) or max_drop_m < 0:
            raise ValueError("workcell.placement.max_drop_m must be finite and non-negative")
        if plan.planned_drop_m > max_drop_m:
            raise ValueError("plan.planned_drop_m exceeds workcell.placement.max_drop_m")

        effort = self._checked_effort(profile)
        turn, tilt = self._place_rotation(apple, plan)
        targets = build_targets(apple, profile, candidate, plan, place_turn_rad=turn, place_tilt_rad=tilt)
        if backend == "act":
            task_text = execution_cfg["act"]["task_text"]
            if not isinstance(task_text, str) or not task_text.strip():
                raise ValueError("execution.act.task_text must be a non-empty string")
            act_timeout = float(execution_cfg["act"]["timeout_s"])
            if not math.isfinite(act_timeout) or act_timeout <= 0:
                raise ValueError("execution.act.timeout_s must be finite and positive")
            phase_a_timeout = min(self._positive_timeout("phase_a_s"), act_timeout)
        else:
            task_text = ""
            phase_a_timeout = self._positive_timeout("phase_a_s")

        phase_a = self._timed_call('t_pick_s', self.hal.execute_phase_a,
            apple,
            profile,
            candidate,
            plan,
            targets,
            backend=backend,
            effort=effort,
            task_text=task_text,
            speed_scale=self._speed_scale("speed_scale_max"),
            timeout_s=phase_a_timeout,
        )
        if not phase_a.success:
            code = phase_a.failure_code
            if code == FailureCode.NONE:
                code = FailureCode.ABORTED
            return ExecResult(False, code, phase_a.phase_reached)
        if not phase_a.holding:
            return ExecResult(False, FailureCode.NOT_HOLDING, phase_a.phase_reached)
        if not phase_a.hover_reached or phase_a.phase_reached != Phase.HOVER:
            return ExecResult(False, FailureCode.HOVER_NOT_REACHED, phase_a.phase_reached)

        soft = execution_cfg["soft_landing"]
        descent = float(soft["final_descent_m"])
        if not math.isfinite(descent) or descent <= 0:
            raise ValueError("execution.soft_landing.final_descent_m must be finite and positive")
        phase_b = self._timed_call('t_place_s', self.hal.execute_phase_b,
            apple,
            profile,
            plan,
            targets,
            build_soft_landing_plan(
                targets,
                final_descent_m=descent,
                slow_speed_scale=self._bounded_speed(
                    "slow_speed_scale", soft["slow_speed_scale"]
                ),
                approach_speed_scale=self._speed_scale("speed_scale_max"),
            ),
            effort=effort,
            final_descent_m=descent,
            slow_speed_scale=self._bounded_speed("slow_speed_scale", soft["slow_speed_scale"]),
            timeout_s=self._positive_timeout("phase_b_s"),
        )
        if not phase_b.success or phase_b.phase_reached != Phase.RETREAT:
            return ExecResult(False, FailureCode.ABORTED, phase_b.phase_reached)
        return ExecResult(True, FailureCode.NONE, phase_b.phase_reached)

    def pick_for_inspection(
        self,
        apple: AppleState,
        profile: GraspProfile,
        candidate: GraspCandidate,
        backend: str | None = None,
    ) -> PickExecutionResult:
        """Pick and lift an apple, returning its measured transform in the TCP frame."""
        self.stage_timings = {}
        self._validate_pick_match(apple, profile, candidate)
        if not profile.feasible:
            return PickExecutionResult(False, FailureCode.ABORTED, Phase.IDLE)
        targets = build_pick_targets(apple, profile, candidate)
        backend, task_text, timeout_s = self._backend_options(backend)
        result = self._timed_call('t_pick_s', self.hal.execute_pick,
            apple,
            profile,
            candidate,
            targets,
            backend=backend,
            effort=self._checked_effort(profile),
            task_text=task_text,
            speed_scale=self._speed_scale("speed_scale_max"),
            timeout_s=timeout_s,
        )
        if not result.success:
            code = result.failure_code
            if code == FailureCode.NONE:
                code = FailureCode.ABORTED
            return PickExecutionResult(False, code, result.phase_reached)
        if not result.holding:
            return PickExecutionResult(
                False, FailureCode.NOT_HOLDING, result.phase_reached
            )
        if not result.hover_reached or result.phase_reached != Phase.HOVER:
            return PickExecutionResult(
                False, FailureCode.HOVER_NOT_REACHED, result.phase_reached
            )
        if result.tool_pose_at_grasp_base is None or result.tool_pose_at_hover_base is None:
            return PickExecutionResult(
                False, FailureCode.ABORTED, result.phase_reached
            )
        apple_in_tcp = relative_pose(
            result.tool_pose_at_grasp_base,
            Pose3D(apple.center_m, (0.0, 0.0, 0.0, 1.0)),
        )
        apple_center_in_base = compose(
            result.tool_pose_at_hover_base,
            apple_in_tcp,
        ).position_m
        return PickExecutionResult(
            True,
            FailureCode.NONE,
            result.phase_reached,
            apple_in_tcp,
            apple_center_in_base,
        )

    def inspect_for_regrade(
        self,
        profile: GraspProfile,
        apple_in_tcp: Pose3D,
        apple_center_in_base: tuple[float, float, float],
        *,
        view_check: ViewCheck | None = None,
    ) -> InspectionExecutionResult:
        """Execute all configured in-hand views and require every capture.

        ``view_check`` is called by the adapter right after each capture; a SLIP verdict
        stops the remaining rotations (turning a shifted apple can drop it).
        """
        self.stage_timings = {}
        if not profile.feasible:
            return InspectionExecutionResult(False, FailureCode.ABORTED, Phase.IDLE, 0)
        views_cfg = self.cfg["inspection"]["views"]
        inspection_plan = build_inspection_plan(
            apple_in_tcp,
            tuple(views_cfg['center_m']) if views_cfg.get('center_m') is not None else apple_center_in_base,
            tilt_rad=float(views_cfg["tilt_rad"]),
            rotation_angles_rad=views_cfg["rotation_angles_rad"],
            settle_time_s=float(views_cfg["settle_time_s"]),
        )
        extra = {} if view_check is None else {"view_check": view_check}
        result = self._timed_call('t_inspect_s', self.hal.inspect_views,
            profile,
            inspection_plan,
            effort=self._checked_effort(profile),
            timeout_s=self._positive_timeout("inspect_s"),
            **extra,
        )
        expected_views = len(inspection_plan.target_tool_poses)
        if result.captured_views < 0 or result.captured_views > expected_views:
            raise ValueError("inspection backend returned an invalid captured_views count")
        if result.captures and len(result.captures) != result.captured_views:
            raise ValueError("inspection backend capture payload count is inconsistent")
        slip_view = getattr(result, "slip_view_index", None)
        if slip_view is not None and not 0 <= slip_view < expected_views:
            raise ValueError("inspection backend returned an invalid slip_view_index")
        success = (
            result.success and slip_view is None and result.captured_views == expected_views
        )
        if success:
            failure = FailureCode.NONE
        elif slip_view is not None:
            failure = FailureCode.SLIP
        else:
            failure = FailureCode.ABORTED
        return InspectionExecutionResult(
            success,
            failure,
            Phase.INSPECT,
            result.captured_views,
            result.captures,
            slip_view,
        )

    def place_held_apple(
        self,
        apple: AppleState,
        profile: GraspProfile,
        apple_in_tcp: Pose3D,
        plan: PlacementPlan,
        backend: str | None = None,
        *,
        speed_scale: float | None = None,
    ) -> ExecResult:
        """Transfer an inspected apple to its new grade's placement and release it.

        ``speed_scale`` overrides the transit speed (slip recovery runs slowly); it may
        not exceed ``safety.speed_scale_max``. The gripper opens only at the target.
        """
        self.stage_timings = {}
        self._validate_held_match(apple, profile, plan)
        if not profile.feasible or not plan.feasible:
            return ExecResult(False, FailureCode.ABORTED, Phase.IDLE)
        if plan.planned_drop_m < 0:
            raise ValueError("plan.planned_drop_m must be non-negative")
        max_drop_m = float(self.cfg["workcell"]["placement"]["max_drop_m"])
        if not math.isfinite(max_drop_m) or max_drop_m < 0:
            raise ValueError("workcell.placement.max_drop_m must be finite and non-negative")
        if plan.planned_drop_m > max_drop_m:
            raise ValueError("plan.planned_drop_m exceeds workcell.placement.max_drop_m")
        turn, tilt = self._place_rotation(apple, plan)
        targets = build_place_targets(apple_in_tcp, plan, turn_rad=turn, outward_tilt_rad=tilt)
        backend, task_text, timeout_s = self._backend_options(backend)
        effort = self._checked_effort(profile)
        transit_speed = self._speed_scale("speed_scale_max")
        if speed_scale is not None:
            transit_speed = self._bounded_speed("speed_scale", speed_scale)
            if transit_speed > self._speed_scale("speed_scale_max"):
                raise ValueError("speed_scale must not exceed safety.speed_scale_max")
        phase_a = self._timed_call('t_place_s', self.hal.execute_place_phase_a,
            apple,
            profile,
            plan,
            targets,
            backend=backend,
            effort=effort,
            task_text=task_text,
            speed_scale=transit_speed,
            timeout_s=timeout_s,
        )
        if not phase_a.success:
            code = phase_a.failure_code
            if code == FailureCode.NONE:
                code = FailureCode.ABORTED
            return ExecResult(False, code, phase_a.phase_reached)
        if not phase_a.holding:
            return ExecResult(False, FailureCode.NOT_HOLDING, phase_a.phase_reached)
        if not phase_a.hover_reached or phase_a.phase_reached != Phase.HOVER:
            return ExecResult(False, FailureCode.HOVER_NOT_REACHED, phase_a.phase_reached)

        soft = self.cfg["execution"]["soft_landing"]
        descent = float(soft["final_descent_m"])
        if not math.isfinite(descent) or descent <= 0:
            raise ValueError("execution.soft_landing.final_descent_m must be finite and positive")
        landing_speed = self._bounded_speed("slow_speed_scale", soft["slow_speed_scale"])
        if speed_scale is not None:
            landing_speed = min(landing_speed, transit_speed)
        phase_b = self._timed_call('t_place_s', self.hal.execute_phase_b,
            apple,
            profile,
            plan,
            targets,
            build_soft_landing_plan(
                targets,
                final_descent_m=descent,
                slow_speed_scale=landing_speed,
                approach_speed_scale=transit_speed,
            ),
            effort=effort,
            final_descent_m=descent,
            slow_speed_scale=landing_speed,
            timeout_s=self._positive_timeout("phase_b_s"),
        )
        if not phase_b.success or phase_b.phase_reached != Phase.RETREAT:
            return ExecResult(False, FailureCode.ABORTED, phase_b.phase_reached)
        return ExecResult(True, FailureCode.NONE, phase_b.phase_reached)

    def recovery_speed_scale(self) -> float:
        """Low transit speed for every slip-recovery motion (execution.recovery.speed_scale)."""
        speed = self._bounded_speed(
            "execution.recovery.speed_scale",
            self.cfg["execution"]["recovery"]["speed_scale"],
        )
        if speed > self._speed_scale("speed_scale_max"):
            raise ValueError("execution.recovery.speed_scale must not exceed safety.speed_scale_max")
        return speed

    def gripper_width_m(self) -> float | None:
        """Measured finger width, or None when the adapter cannot read it."""
        reader = getattr(self.hal, "gripper_width_m", None)
        if reader is None:
            return None
        width = reader()
        if width is None:
            return None
        width = float(width)
        if not math.isfinite(width) or width < 0:
            raise ValueError("gripper width must be finite and non-negative")
        return width

    def grip_lost(self) -> bool:
        """True when the fingers are (almost) fully closed, i.e. the apple is already gone.

        Unknown width (adapter cannot read it) is reported as not lost; the caller then
        proceeds with the slow recovery placement.
        """
        width = self.gripper_width_m()
        if width is None:
            return False
        limit = float(self.cfg["execution"]["recovery"]["lost_grip_width_m"])
        if not math.isfinite(limit) or limit < 0:
            raise ValueError("execution.recovery.lost_grip_width_m must be finite and non-negative")
        return width <= limit

    def inspect(
        self,
        profile: GraspProfile,
        apple_in_tcp: Pose3D,
        apple_center_in_base: tuple[float, float, float],
        views_cfg: dict,
    ) -> ExecResult:
        if not profile.feasible:
            return ExecResult(False, FailureCode.ABORTED, Phase.IDLE)
        inspection_plan = build_inspection_plan(
            apple_in_tcp,
            apple_center_in_base,
            tilt_rad=float(views_cfg["tilt_rad"]),
            rotation_angles_rad=views_cfg["rotation_angles_rad"],
            settle_time_s=float(views_cfg["settle_time_s"]),
        )
        success = self.hal.inspect(
            profile,
            inspection_plan,
            effort=self._checked_effort(profile),
            timeout_s=self._positive_timeout("inspect_s"),
        )
        return ExecResult(success, FailureCode.NONE if success else FailureCode.ABORTED,
                          Phase.INSPECT)

    def _place_rotation(self, apple: AppleState, plan: PlacementPlan) -> tuple[float, float]:
        """Release posture from ``execution.transfer`` (sphere: any orientation is valid).

        ``face_place_bearing`` turns the tool from the apple's bearing to the
        cell's bearing; cells beyond ``far_place_radius_m`` tilt the tool tip
        outward by ``far_place_tilt_rad`` to stay within the wrist range.
        """
        transfer = self.cfg["execution"].get("transfer") or {}
        if transfer.get("height_m") is None:
            return 0.0, 0.0
        turn = 0.0
        if transfer.get("face_place_bearing", False):
            turn = (math.atan2(plan.place_center_m[1], plan.place_center_m[0])
                    - math.atan2(apple.center_m[1], apple.center_m[0]))
        tilt = 0.0
        radius = transfer.get("far_place_radius_m")
        if radius is not None and math.hypot(*plan.place_center_m[:2]) > float(radius):
            tilt = float(transfer.get("far_place_tilt_rad", 0.0))
        return turn, tilt

    def _checked_effort(self, profile: GraspProfile) -> int:
        effort = effort_for(profile)
        gripper_cfg = self.cfg["execution"]["gripper"]
        if gripper_cfg.get("control_mode", "calibrated_force") == "fixed_effort":
            fixed_effort = gripper_cfg["fixed_effort"]
            if isinstance(fixed_effort, bool) or not isinstance(fixed_effort, int):
                raise ValueError("execution.gripper.fixed_effort must be an integer")
            if effort != fixed_effort:
                raise ValueError("profile effort does not match the configured fixed effort")
            maximum = self.cfg.get("safety", {}).get("gripper_effort_max")
            if isinstance(maximum, bool) or not isinstance(maximum, int) or not 0 <= maximum <= 5000:
                raise ValueError("safety.gripper_effort_max must be an integer from 0 to 5000")
            if effort > maximum:
                raise ValueError("fixed effort exceeds safety.gripper_effort_max")
            return effort

        force_limit = self.cfg["size_bins"]["force"].get("f_bruise_n")
        if force_limit is None:
            raise ValueError("size_bins.force.f_bruise_n must be measured before grasp execution")
        force_limit = float(force_limit)
        if not math.isfinite(force_limit) or force_limit <= 0:
            raise ValueError("size_bins.force.f_bruise_n must be finite and positive")
        if (
            not math.isfinite(profile.grasp_force_n)
            or profile.grasp_force_n <= 0
            or profile.grasp_force_n >= force_limit
        ):
            raise ValueError("profile grasp force must be positive and below the bruise limit")

        maximum = self.cfg.get("safety", {}).get("gripper_effort_max")
        if maximum is None:
            raise ValueError("safety.gripper_effort_max must be set before grasp execution")
        if isinstance(maximum, bool) or not isinstance(maximum, int) or not 0 <= maximum <= 5000:
            raise ValueError("safety.gripper_effort_max must be an integer from 0 to 5000")
        if effort > maximum:
            raise ValueError("profile effort exceeds safety.gripper_effort_max")
        return effort

    def _positive_timeout(self, key: str) -> float:
        value = float(self.cfg["execution"]["timeouts"][key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"execution.{key} must be finite and positive")
        return value

    def _speed_scale(self, key: str) -> float:
        value = float(self.cfg["safety"][key])
        return self._bounded_speed(key, value)

    @staticmethod
    def _bounded_speed(key: str, value: float) -> float:
        value = float(value)
        if not math.isfinite(value) or not 0 < value <= 1:
            raise ValueError(f"{key} must be finite and in (0, 1]")
        return value

    @staticmethod
    def _validate_match(
        apple: AppleState,
        profile: GraspProfile,
        candidate: GraspCandidate,
        plan: PlacementPlan,
    ) -> None:
        MotionExecutor._validate_pick_match(apple, profile, candidate)
        if len({apple.scene_id, plan.scene_id}) != 1:
            raise ValueError("apple, profile, candidate, and placement must share a scene_id")
        if len({apple.apple_id, plan.apple_id}) != 1:
            raise ValueError("apple, profile, candidate, and placement must share an apple_id")
        if not math.isfinite(plan.planned_drop_m):
            raise ValueError("plan.planned_drop_m must be finite")

    @staticmethod
    def _validate_pick_match(
        apple: AppleState,
        profile: GraspProfile,
        candidate: GraspCandidate,
    ) -> None:
        if len({apple.scene_id, profile.scene_id, candidate.scene_id}) != 1:
            raise ValueError("apple, profile, and candidate must share a scene_id")
        if len({apple.apple_id, profile.apple_id, candidate.apple_id}) != 1:
            raise ValueError("apple, profile, and candidate must share an apple_id")
        if candidate.collision_free is not True:
            raise ValueError("cannot execute a grasp candidate that is not collision-free")
        if not math.isfinite(candidate.grasp_yaw_rad):
            raise ValueError("candidate.grasp_yaw_rad must be finite")
        if not math.isfinite(candidate.approach_height_m) or candidate.approach_height_m <= 0:
            raise ValueError("candidate.approach_height_m must be finite and positive")
        if not math.isfinite(candidate.min_clearance_m) or candidate.min_clearance_m < 0:
            raise ValueError("candidate.min_clearance_m must be finite and non-negative")
        if not math.isfinite(candidate.score):
            raise ValueError("candidate.score must be finite")
        if not math.isfinite(apple.diameter_m) or apple.diameter_m <= 0:
            raise ValueError("apple.diameter_m must be finite and positive")

    @staticmethod
    def _validate_held_match(
        apple: AppleState,
        profile: GraspProfile,
        plan: PlacementPlan,
    ) -> None:
        if len({apple.scene_id, profile.scene_id, plan.scene_id}) != 1:
            raise ValueError("apple, profile, and placement must share a scene_id")
        if len({apple.apple_id, profile.apple_id, plan.apple_id}) != 1:
            raise ValueError("apple, profile, and placement must share an apple_id")
        if not math.isfinite(apple.diameter_m) or apple.diameter_m <= 0:
            raise ValueError("apple.diameter_m must be finite and positive")

    def _backend_options(self, backend: str | None) -> tuple[str, str, float]:
        execution_cfg = self.cfg["execution"]
        selected_backend = backend or execution_cfg.get("default_backend", "scripted")
        if selected_backend not in ("scripted", "act"):
            raise ValueError("backend must be 'scripted' or 'act'")
        if selected_backend == "act":
            task_text = execution_cfg["act"]["task_text"]
            if not isinstance(task_text, str) or not task_text.strip():
                raise ValueError("execution.act.task_text must be a non-empty string")
            act_timeout = float(execution_cfg["act"]["timeout_s"])
            if not math.isfinite(act_timeout) or act_timeout <= 0:
                raise ValueError("execution.act.timeout_s must be finite and positive")
            timeout_s = min(self._positive_timeout("phase_a_s"), act_timeout)
        else:
            task_text = ""
            timeout_s = self._positive_timeout("phase_a_s")
        return selected_backend, task_text, timeout_s
