"""Deterministic motion adapter for unit tests; it never connects to hardware."""

from __future__ import annotations

from dataclasses import dataclass, field

from aple.execution.goal_builder import MotionTargets, PickTargets, PlaceTargets
from aple.execution.inspect_skill import InspectionPlan
from aple.execution.motion_executor import (
    GraspCandidate,
    InspectionPhaseResult,
    PhaseAResult,
    PhaseBResult,
    PickPhaseResult,
)
from aple.execution.soft_landing import SoftLandingPlan
from aple.schema import (AppleState, FailureCode, GraspProfile, Phase,
                         PlacementPlan)


@dataclass
class RecordingMotionAdapter:
    """Record requested operations and return explicitly configured outcomes."""

    phase_a_result: PhaseAResult = field(default_factory=lambda: PhaseAResult(
        True, FailureCode.NONE, Phase.HOVER, True, True
    ))
    phase_b_result: PhaseBResult = field(default_factory=lambda: PhaseBResult(
        True, Phase.RETREAT
    ))
    named_pose_success: bool = True
    inspect_success: bool = True
    captured_views: int | None = None
    calls: list[tuple[str, dict]] = field(default_factory=list)

    def move_named(self, name: str, joint_target_rad: tuple[float, ...], *,
                   speed_scale: float, timeout_s: float) -> bool:
        self.calls.append(("move_named", {
            "name": name,
            "joint_target_rad": joint_target_rad,
            "speed_scale": speed_scale,
            "timeout_s": timeout_s,
        }))
        return self.named_pose_success

    def execute_phase_a(self, apple: AppleState, profile: GraspProfile,
                        candidate: GraspCandidate, plan: PlacementPlan,
                        targets: MotionTargets, *,
                        backend: str, effort: int, task_text: str,
                        speed_scale: float, timeout_s: float) -> PhaseAResult:
        self.calls.append(("phase_a", {
            "apple_id": apple.apple_id,
            "profile_apple_id": profile.apple_id,
            "candidate_apple_id": candidate.apple_id,
            "grasp_pose": candidate.grasp_pose,
            "grasp_yaw_rad": candidate.grasp_yaw_rad,
            "approach_height_m": candidate.approach_height_m,
            "motion_targets": targets,
            "plan_apple_id": plan.apple_id,
            "backend": backend,
            "effort": effort,
            "task_text": task_text,
            "speed_scale": speed_scale,
            "timeout_s": timeout_s,
        }))
        return self.phase_a_result

    def execute_phase_b(self, apple: AppleState, profile: GraspProfile, plan: PlacementPlan,
                        targets: MotionTargets | PlaceTargets,
                        landing_plan: SoftLandingPlan, *,
                        effort: int, final_descent_m: float, slow_speed_scale: float,
                        timeout_s: float) -> PhaseBResult:
        self.calls.append(("phase_b", {
            "apple_id": apple.apple_id,
            "motion_targets": targets,
            "landing_plan": landing_plan,
            "effort": effort,
            "final_descent_m": final_descent_m,
            "slow_speed_scale": slow_speed_scale,
            "timeout_s": timeout_s,
        }))
        return self.phase_b_result

    def execute_pick(self, apple: AppleState, profile: GraspProfile,
                     candidate: GraspCandidate, targets: PickTargets, *,
                     backend: str, effort: int, task_text: str,
                     speed_scale: float, timeout_s: float) -> PickPhaseResult:
        self.calls.append(("pick", {
            "apple_id": apple.apple_id,
            "candidate_apple_id": candidate.apple_id,
            "targets": targets,
            "backend": backend,
            "effort": effort,
            "task_text": task_text,
            "speed_scale": speed_scale,
            "timeout_s": timeout_s,
        }))
        return PickPhaseResult(
            self.phase_a_result.success,
            self.phase_a_result.failure_code,
            self.phase_a_result.phase_reached,
            self.phase_a_result.holding,
            self.phase_a_result.hover_reached,
            targets.pick_pose if self.phase_a_result.success else None,
            targets.pregrasp_pose if self.phase_a_result.success else None,
        )

    def execute_place_phase_a(self, apple: AppleState, profile: GraspProfile,
                              placement: PlacementPlan, targets: PlaceTargets, *,
                              backend: str, effort: int, task_text: str,
                              speed_scale: float, timeout_s: float) -> PhaseAResult:
        self.calls.append(("place_phase_a", {
            "apple_id": apple.apple_id,
            "targets": targets,
            "placement_apple_id": placement.apple_id,
            "backend": backend,
            "effort": effort,
            "task_text": task_text,
            "speed_scale": speed_scale,
            "timeout_s": timeout_s,
        }))
        return self.phase_a_result

    def inspect_views(self, profile: GraspProfile, inspection_plan: InspectionPlan, *,
                      effort: int, timeout_s: float) -> InspectionPhaseResult:
        self.calls.append(("inspect_views", {
            "apple_id": profile.apple_id,
            "inspection_plan": inspection_plan,
            "effort": effort,
            "timeout_s": timeout_s,
        }))
        count = (
            len(inspection_plan.target_tool_poses)
            if self.captured_views is None and self.inspect_success
            else self.captured_views or 0
        )
        return InspectionPhaseResult(self.inspect_success, count)

    def inspect(self, profile: GraspProfile, inspection_plan: InspectionPlan, *,
                effort: int, timeout_s: float) -> bool:
        self.calls.append(("inspect", {
            "apple_id": profile.apple_id,
            "inspection_plan": inspection_plan,
            "effort": effort,
            "timeout_s": timeout_s,
        }))
        return self.inspect_success
