"""Scripted Cartesian motion over an injected, speed-aware Piper transport.

The transport boundary is intentional: Piper Web v0.5.9's public joint jog
does not accept per-goal speed, so that API cannot safely implement Phase B.
"""

from __future__ import annotations

import math
import time
from typing import Callable, Protocol

from aple.execution.goal_builder import (
    MotionTargets,
    PickTargets,
    PlaceTargets,
    Pose3D,
)
from aple.execution.inspect_skill import InspectionPlan
from aple.execution.motion_executor import (
    GraspCandidate,
    InspectionPhaseResult,
    PhaseAResult,
    PhaseBResult,
    PickPhaseResult,
    ViewCheck,
    ViewVerdict,
)
from aple.execution.soft_landing import GripperAction, SoftLandingPlan
from aple.execution.trajectory import (
    IKSolver,
    JointTrajectoryPlanner,
    SegmentValidator,
    ToolPose,
    TrajectoryPlanningError,
)
from aple.inspection.in_hand_capture import InHandCapture
from aple.schema import (
    AppleState,
    FailureCode,
    GraspProfile,
    Phase,
    PlacementPlan,
)


class PiperMotionTransport(Protocol):
    """Low-level adapter that must enforce speed and wait for each goal."""

    def joint_state(self) -> tuple[float, ...]: ...

    def move_joint_goal(
        self,
        position_rad: tuple[float, ...],
        *,
        speed_scale: float,
        timeout_s: float,
    ) -> bool: ...

    def set_gripper_width(
        self, width_m: float, effort: int, *, timeout_s: float
    ) -> bool: ...

    def tool_pose(self) -> Pose3D: ...

    def capture_in_hand(
        self, apple_id: int, view_index: int, *, timeout_s: float
    ) -> InHandCapture | None: ...


class PiperTrajectoryMotionAdapter:
    """Implement the motion-executor contract using IK and checked joint steps.

    ``segment_validator`` and the transport are mandatory; no collision checker
    or speed capability is fabricated when Piper's runtime does not provide one.
    """

    def __init__(
        self,
        transport: PiperMotionTransport,
        cfg: dict,
        ik_solver: IKSolver,
        segment_validator: SegmentValidator,
    ) -> None:
        from aple.execution.gripper_geometry import apply_attachment_geometry
        cfg = apply_attachment_geometry(cfg)
        self._transport = transport
        self._cfg = cfg
        safety = cfg["safety"]
        limits = safety.get("joint_limits_rad")
        if limits is None:
            raise ValueError("safety.joint_limits_rad must be measured before scripted motion")
        step = cfg["execution"]["trajectory"]["max_joint_step_rad"]
        self._planner = JointTrajectoryPlanner(
            ik_solver,
            segment_validator,
            limits,
            max_joint_step_rad=step,
        )

    def gripper_width_m(self) -> float | None:
        """Finger width if the transport can read it; None means unknown, not lost."""
        reader = getattr(self._transport, "gripper_width_m", None)
        return None if reader is None else reader()

    def move_named(
        self,
        name: str,
        joint_target_rad: tuple[float, ...],
        *,
        speed_scale: float,
        timeout_s: float,
    ) -> bool:
        del name
        self._validate_speed_scale(speed_scale)
        deadline = self._deadline(timeout_s)
        current = self._transport.joint_state()
        # Named moves need the same bounds and collision checks as Cartesian moves.
        try:
            trajectory = self._planner.plan_joint_goal(joint_target_rad, current)
            for point in trajectory.points_rad:
                if not self._transport.move_joint_goal(
                    point, speed_scale=speed_scale, timeout_s=self._remaining(deadline)
                ):
                    return False
        except (TrajectoryPlanningError, TimeoutError):
            return False
        return True

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
    ) -> PhaseAResult:
        del candidate, plan
        if backend != "scripted":
            raise ValueError("PiperTrajectoryMotionAdapter supports only the scripted backend")
        if task_text:
            raise ValueError("task_text is not used by the scripted backend")
        deadline = self._deadline(timeout_s)
        phase = Phase.APPROACH
        holding = False
        try:
            if not self._transport.set_gripper_width(
                profile.open_width_m, effort, timeout_s=self._remaining(deadline)
            ):
                return PhaseAResult(False, FailureCode.TIMEOUT, Phase.APPROACH, False, False)
            self._move_poses(
                (targets.pregrasp_pose, targets.pick_pose),
                speed_scale=speed_scale,
                deadline=deadline,
            )
            if not self._transport.set_gripper_width(
                apple.diameter_m, effort, timeout_s=self._remaining(deadline)
            ):
                return PhaseAResult(False, FailureCode.TIMEOUT, Phase.GRASP, False, False)
            holding = True
            phase = Phase.TRANSIT
            self._move_poses(
                (targets.pregrasp_pose, targets.hover_pose),
                speed_scale=speed_scale,
                deadline=deadline,
            )
        except TrajectoryPlanningError as exc:
            return PhaseAResult(False, exc.failure_code, phase, holding, False)
        except TimeoutError:
            return PhaseAResult(False, FailureCode.TIMEOUT, phase, holding, False)
        return PhaseAResult(True, FailureCode.NONE, Phase.HOVER, True, True)

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
    ) -> PickPhaseResult:
        del candidate
        if backend != "scripted":
            raise ValueError("PiperTrajectoryMotionAdapter supports only the scripted backend")
        if task_text:
            raise ValueError("task_text is not used by the scripted backend")
        deadline = self._deadline(timeout_s)
        phase = Phase.APPROACH
        holding = False
        try:
            if not self._transport.set_gripper_width(
                profile.open_width_m, effort, timeout_s=self._remaining(deadline)
            ):
                return PickPhaseResult(
                    False, FailureCode.TIMEOUT, Phase.APPROACH, False, False, None, None
                )
            self._move_poses(
                (targets.pregrasp_pose, targets.pick_pose),
                speed_scale=speed_scale,
                deadline=deadline,
            )
            if not self._transport.set_gripper_width(
                apple.diameter_m, effort, timeout_s=self._remaining(deadline)
            ):
                return PickPhaseResult(
                    False, FailureCode.TIMEOUT, Phase.GRASP, False, False, None, None
                )
            holding = True
            phase = Phase.LIFT
            grasp_pose = self._transport.tool_pose()
            self._move_poses(
                (targets.pregrasp_pose,),
                speed_scale=speed_scale,
                deadline=deadline,
            )
            hover_pose = self._transport.tool_pose()
        except TrajectoryPlanningError as exc:
            return PickPhaseResult(
                False, exc.failure_code, phase, holding, False, None, None
            )
        except TimeoutError:
            return PickPhaseResult(
                False, FailureCode.TIMEOUT, phase, holding, False, None, None
            )
        return PickPhaseResult(
            True, FailureCode.NONE, Phase.HOVER, True, True, grasp_pose, hover_pose
        )

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
    ) -> PhaseAResult:
        del apple, profile, placement
        if backend != "scripted":
            raise ValueError("PiperTrajectoryMotionAdapter supports only the scripted backend")
        if task_text:
            raise ValueError("task_text is not used by the scripted backend")
        try:
            self._move_poses(
                (targets.hover_pose,),
                speed_scale=speed_scale,
                deadline=self._deadline(timeout_s),
            )
        except TrajectoryPlanningError as exc:
            return PhaseAResult(False, exc.failure_code, Phase.TRANSIT, True, False)
        except TimeoutError:
            return PhaseAResult(False, FailureCode.TIMEOUT, Phase.TRANSIT, True, False)
        if effort < 0:
            raise ValueError("effort must be non-negative")
        return PhaseAResult(True, FailureCode.NONE, Phase.HOVER, True, True)

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
    ) -> PhaseBResult:
        del apple, plan, targets, final_descent_m, slow_speed_scale
        deadline = self._deadline(timeout_s)
        phase = Phase.SOFT_LAND
        for index, step in enumerate(landing_plan.steps):
            try:
                self._move_poses(
                    (step.pose,),
                    speed_scale=step.speed_scale,
                    deadline=deadline,
                )
            except (TrajectoryPlanningError, TimeoutError):
                return PhaseBResult(False, phase)
            if step.gripper_action == GripperAction.RELEASE:
                phase = Phase.RELEASE
                try:
                    released = self._transport.set_gripper_width(
                        profile.open_width_m,
                        effort,
                        timeout_s=self._remaining(deadline),
                    )
                except TimeoutError:
                    return PhaseBResult(False, Phase.RELEASE)
                if not released:
                    return PhaseBResult(False, Phase.RELEASE)
            if index == len(landing_plan.steps) - 1:
                phase = Phase.RETREAT
        return PhaseBResult(True, Phase.RETREAT)

    def inspect_views(
        self,
        profile: GraspProfile,
        inspection_plan: InspectionPlan,
        *,
        effort: int,
        timeout_s: float,
        view_check: ViewCheck | None = None,
    ) -> InspectionPhaseResult:
        del effort
        deadline = self._deadline(timeout_s)
        speed_scale = float(self._cfg["safety"]["speed_scale_max"])
        captures: list[InHandCapture] = []
        width_watch = self._grip_width_watch()
        for view_index, pose in enumerate(inspection_plan.target_tool_poses):
            try:
                self._move_poses((pose,), speed_scale=speed_scale, deadline=deadline)
            except (TrajectoryPlanningError, TimeoutError):
                return InspectionPhaseResult(False, len(captures), tuple(captures))
            remaining = self._remaining(deadline)
            if inspection_plan.settle_time_s >= remaining:
                return InspectionPhaseResult(False, len(captures), tuple(captures))
            time.sleep(inspection_plan.settle_time_s)
            capture = self._transport.capture_in_hand(
                profile.apple_id,
                view_index,
                timeout_s=self._remaining(deadline),
            )
            if capture is None:
                return InspectionPhaseResult(False, len(captures), tuple(captures))
            if width_watch is not None and width_watch():
                # 그리퍼 폭 변화가 1차 미끄러짐 신호 (S8-6): 영상 검사 전에 바로 멈춘다.
                captures.append(capture)
                return InspectionPhaseResult(
                    False, len(captures), tuple(captures), slip_view_index=view_index
                )
            if view_check is not None:
                verdict = view_check(view_index, capture)
                if verdict == ViewVerdict.UNVERIFIED:
                    # Depth may simply be empty: retake once at the same pose before calling it slip.
                    retake = self._transport.capture_in_hand(
                        profile.apple_id,
                        view_index,
                        timeout_s=self._remaining(deadline),
                    )
                    if retake is None:
                        return InspectionPhaseResult(False, len(captures), tuple(captures))
                    capture = retake
                    if width_watch is not None and width_watch():
                        captures.append(capture)
                        return InspectionPhaseResult(
                            False, len(captures), tuple(captures), slip_view_index=view_index
                        )
                    verdict = view_check(view_index, capture)
                    if verdict == ViewVerdict.UNVERIFIED:
                        verdict = ViewVerdict.SLIP
                if verdict == ViewVerdict.SLIP:
                    captures.append(capture)
                    return InspectionPhaseResult(
                        False, len(captures), tuple(captures), slip_view_index=view_index
                    )
            captures.append(capture)
        return InspectionPhaseResult(True, len(captures), tuple(captures))

    def _grip_width_watch(self) -> Callable[[], bool] | None:
        """Callable returning True once the finger width shows the apple slipping or lost.

        Needs a transport that can read the width and ``execution.recovery.width_watch``;
        the reference is the width at the start of the inspection (apple just grasped).
        """
        execution = self._cfg.get("execution", {})
        if not execution.get("recovery", {}).get("width_watch", False):
            return None
        reference = self.gripper_width_m()
        if reference is None:
            return None
        slip_width_m = float(execution["gripper"]["slip_width_m"])
        lost_width_m = float(execution["recovery"]["lost_grip_width_m"])

        def slipped() -> bool:
            width = self.gripper_width_m()
            return width is not None and (
                width - reference > slip_width_m or width <= lost_width_m
            )

        return slipped

    def inspect(
        self,
        profile: GraspProfile,
        inspection_plan: InspectionPlan,
        *,
        effort: int,
        timeout_s: float,
    ) -> bool:
        return self.inspect_views(
            profile, inspection_plan, effort=effort, timeout_s=timeout_s
        ).success

    def _move_poses(
        self,
        poses: tuple[Pose3D, ...],
        *,
        speed_scale: float,
        deadline: float,
    ) -> None:
        self._validate_speed_scale(speed_scale)
        current = self._transport.joint_state()
        for pose in poses:
            trajectory = self._planner.plan(
                (ToolPose(pose.position_m, pose.orientation_xyzw),), current
            )
            for joints in trajectory.points_rad:
                if not self._transport.move_joint_goal(
                    joints,
                    speed_scale=speed_scale,
                    timeout_s=self._remaining(deadline),
                ):
                    raise TimeoutError("Piper joint goal did not reach its target")
                current = joints

    @staticmethod
    def _deadline(timeout_s: float) -> float:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and positive")
        return time.monotonic() + timeout_s

    @staticmethod
    def _validate_speed_scale(speed_scale: float) -> None:
        if not math.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError("speed_scale must be finite and in (0, 1]")

    @staticmethod
    def _remaining(deadline: float) -> float:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("motion operation exceeded its deadline")
        return remaining


def build_piper_motion_adapter(
    studio,
    cfg: dict,
    ik_solver: IKSolver,
    segment_validator: SegmentValidator,
    *,
    tool_pose_provider=None,
    in_hand_capture_provider=None,
) -> PiperTrajectoryMotionAdapter:
    """Assemble the scripted adapter on Piper Web with waypoint-subdivision slowdown."""
    from aple.hal.piper_studio import PiperStudioMotionTransport
    from aple.execution.trajectory import (
        PiperKinematicsIKSolver, PiperKinematicsFKProvider, PiperTCPTransform,
    )

    if isinstance(ik_solver, PiperKinematicsIKSolver):
        measured = PiperTCPTransform.from_config(cfg)
        if measured != ik_solver.tcp_transform:
            raise ValueError("Piper IK transform must match workcell.tcp_offset_m")
        if not isinstance(tool_pose_provider, PiperKinematicsFKProvider):
            raise ValueError("Piper IK requires the FK provider from build_piper_kinematics")
        if tool_pose_provider.tcp_transform != ik_solver.tcp_transform:
            raise ValueError("Piper IK and FK must share the same measured TCP offset")

    transport = PiperStudioMotionTransport(
        studio,
        max_joint_step_rad=cfg["execution"]["trajectory"]["max_joint_step_rad"],
        tool_pose_provider=tool_pose_provider,
        in_hand_capture_provider=in_hand_capture_provider,
    )
    return PiperTrajectoryMotionAdapter(transport, cfg, ik_solver, segment_validator)
