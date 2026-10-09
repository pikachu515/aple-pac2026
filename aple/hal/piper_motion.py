"""Scripted Cartesian motion over an injected, speed-aware Piper transport.

The Studio transport schedules timed joint setpoints under measured limits and
monitors feedback; the underlying jog API itself has no per-goal speed field.
"""

from __future__ import annotations

import logging
import math
import time
from functools import wraps
from typing import Callable, Protocol

import numpy as np

from aple.execution.goal_builder import (
    MotionTargets,
    PickTargets,
    PlaceTargets,
    Pose3D,
    relative_pose,
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


def _invalidate_load_on_failure(method):
    """A failed operation must never leave a stale empty-gripper speed permit."""
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            result = method(self, *args, **kwargs)
        except Exception:
            self._load_state = 'unknown'
            raise
        if result is False or getattr(result, 'success', True) is False:
            self._load_state = 'unknown'
        return result
    return guarded


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
        self._segment_validator = segment_validator
        self._cfg = cfg
        safety = cfg["safety"]
        # Opening/release acknowledgment grants empty state; startup and failures
        # remain unknown. No width-only inference from a failed grasp is allowed.
        self._load_state = 'unknown'
        self._empty_speed_scale = cfg['execution'].get('motion', {}).get('empty_speed_scale')
        if self._empty_speed_scale is not None:
            if (isinstance(self._empty_speed_scale, bool)
                    or not isinstance(self._empty_speed_scale, (int, float))
                    or not math.isfinite(self._empty_speed_scale)
                    or not 0 < self._empty_speed_scale <= 1):
                # 10/10 현장: 빈손 이동을 일반 속도(speed_scale_max)보다 느리게 두는 것도 허용한다 — 느린 쪽은 더 안전하다
                raise ValueError('empty_speed_scale must be in (0, 1]')
        limits = safety.get("joint_limits_rad")
        if limits is None:
            raise ValueError("safety.joint_limits_rad must be measured before scripted motion")
        step = cfg["execution"]["trajectory"]["max_joint_step_rad"]
        cartesian_step = cfg["execution"]["trajectory"].get("max_cartesian_step_m")
        if isinstance(cartesian_step, bool):
            raise ValueError("max_cartesian_step_m must be a positive finite number or null")
        self._cartesian_step_m = (
            None if cartesian_step is None else JointTrajectoryPlanner._positive_finite(
                cartesian_step, "max_cartesian_step_m"
            )
        )
        self._planner = JointTrajectoryPlanner(
            ik_solver,
            segment_validator,
            limits,
            max_joint_step_rad=step,
        )
        self._transfer = _transfer_settings(cfg['execution'].get('transfer'))

    def _empty_transit_speed(self, requested):
        # Explicitly slow recovery/contact moves retain their requested speed.
        limit = float(self._cfg['safety']['speed_scale_max'])
        if (self._load_state == 'empty' and self._empty_speed_scale is not None
                and requested >= limit):
            return float(self._empty_speed_scale)
        return min(requested, limit)

    def gripper_width_m(self) -> float | None:
        """Finger width if the transport can read it; None means unknown, not lost."""
        reader = getattr(self._transport, "gripper_width_m", None)
        return None if reader is None else reader()

    @_invalidate_load_on_failure
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
            self._send_trajectory(trajectory.points_rad, speed_scale=self._empty_transit_speed(speed_scale), deadline=deadline)
        except (TrajectoryPlanningError, TimeoutError):
            return False
        return True

    @_invalidate_load_on_failure
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
            self._load_state = 'empty'
            self._transfer_over((targets.pregrasp_pose,),
                speed_scale=self._empty_transit_speed(speed_scale), deadline=deadline)
            self._move_poses((targets.pregrasp_pose,),
                speed_scale=self._empty_transit_speed(speed_scale), deadline=deadline)
            # Descent close to the fruit retains the normal approach speed.
            self._move_poses((targets.pick_pose,), speed_scale=speed_scale, deadline=deadline)
            self._load_state = 'unknown'
            if not self._close_for_grasp(apple, profile, effort, deadline):
                return PhaseAResult(False, FailureCode.TIMEOUT, Phase.GRASP, False, False)
            self._set_payload(targets.apple_in_tcp, apple, allow_support_contact=True)
            phase = Phase.LIFT
            self._move_poses(
                (targets.pregrasp_pose,),
                speed_scale=speed_scale,
                deadline=deadline,
            )
            holding = self._confirm_holding(apple, targets.apple_in_tcp, deadline)
            if not holding:
                return PhaseAResult(False, FailureCode.NOT_HOLDING, Phase.LIFT, False, False)
            self._load_state = 'held'
            self._set_payload(targets.apple_in_tcp, apple)
            phase = Phase.TRANSIT
            self._transfer_over((targets.hover_pose,), speed_scale=speed_scale, deadline=deadline)
            self._move_poses((targets.hover_pose,), speed_scale=speed_scale, deadline=deadline)
        except TrajectoryPlanningError as exc:
            return PhaseAResult(False, exc.failure_code, phase, holding, False)
        except TimeoutError:
            return PhaseAResult(False, FailureCode.TIMEOUT, phase, holding, False)
        return PhaseAResult(True, FailureCode.NONE, Phase.HOVER, True, True)

    @_invalidate_load_on_failure
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
            self._load_state = 'empty'
            self._transfer_over((targets.pregrasp_pose,),
                speed_scale=self._empty_transit_speed(speed_scale), deadline=deadline)
            self._move_poses((targets.pregrasp_pose,),
                speed_scale=self._empty_transit_speed(speed_scale), deadline=deadline)
            # Descent close to the fruit retains the normal approach speed.
            self._move_poses((targets.pick_pose,), speed_scale=speed_scale, deadline=deadline)
            self._load_state = 'unknown'
            if not self._close_for_grasp(apple, profile, effort, deadline):
                return PickPhaseResult(
                    False, FailureCode.TIMEOUT, Phase.GRASP, False, False, None, None
                )
            phase = Phase.LIFT
            grasp_pose = self._transport.tool_pose()
            actual_offset = relative_pose(grasp_pose, Pose3D(apple.center_m, (0., 0., 0., 1.)))
            self._set_payload(actual_offset, apple, allow_support_contact=True)
            self._move_poses(
                (targets.pregrasp_pose,),
                speed_scale=speed_scale,
                deadline=deadline,
            )
            hover_pose = self._transport.tool_pose()
            holding = self._confirm_holding(apple, actual_offset, deadline)
            if not holding:
                return PickPhaseResult(False, FailureCode.NOT_HOLDING, Phase.LIFT, False, False, grasp_pose, hover_pose)
            self._load_state = 'held'
            self._set_payload(actual_offset, apple)
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

    @_invalidate_load_on_failure
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
            deadline = self._deadline(timeout_s)
            self._transfer_over((targets.hover_pose,), speed_scale=speed_scale, deadline=deadline)
            self._move_poses(
                (targets.hover_pose,),
                speed_scale=speed_scale,
                deadline=deadline,
            )
        except TrajectoryPlanningError as exc:
            return PhaseAResult(False, exc.failure_code, Phase.TRANSIT, True, False)
        except TimeoutError:
            return PhaseAResult(False, FailureCode.TIMEOUT, Phase.TRANSIT, True, False)
        if effort < 0:
            raise ValueError("effort must be non-negative")
        return PhaseAResult(True, FailureCode.NONE, Phase.HOVER, True, True)

    @_invalidate_load_on_failure
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
                    speed_scale=self._empty_transit_speed(step.speed_scale)
                        if phase == Phase.RELEASE else step.speed_scale,
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
                self._load_state = 'empty'
                clear_payload = getattr(self._segment_validator, "clear_payload", None)
                if clear_payload is not None:
                    clear_payload()
            if index == len(landing_plan.steps) - 1:
                phase = Phase.RETREAT
        return PhaseBResult(True, Phase.RETREAT)

    @_invalidate_load_on_failure
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

    def _transfer_over(self, targets: tuple[Pose3D, ...], *, speed_scale: float, deadline: float) -> None:
        """Rise to the transfer plane, cross it, and stop above the target.

        The caller then descends to the target itself. ``execution.transfer``
        null keeps the previous direct motion. All poses pass the same IK and
        segment checks as any other Cartesian move.
        """
        if self._transfer is None:
            return
        current = self._transport.tool_pose()
        poses = transfer_plane_poses(current, targets[0], **self._transfer)
        if current is None:
            poses = poses[-1:]  # unknown current pose: go straight to the point above the target
        if poses:
            self._move_poses(poses, speed_scale=speed_scale, deadline=deadline)

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
            step = self._cartesian_step_m
            waypoints = (pose,)
            if step is not None:
                start_pose = self._transport.tool_pose()
                distance = math.dist(start_pose.position_m, pose.position_m)
                count = max(1, math.ceil(distance / step))
                a = np.asarray(start_pose.orientation_xyzw, dtype=float)
                b = np.asarray(pose.orientation_xyzw, dtype=float)
                if np.dot(a, b) < 0:
                    b = -b
                values = []
                for i in range(1, count + 1):
                    t = i / count
                    orientation = (1-t) * a + t * b
                    orientation /= np.linalg.norm(orientation)
                    position = tuple((1-t)*x + t*y for x, y in zip(start_pose.position_m, pose.position_m))
                    values.append(Pose3D(position, tuple(orientation)))
                waypoints = tuple(values)
            for waypoint in waypoints:
                trajectory = self._planner.plan(
                    (ToolPose(waypoint.position_m, waypoint.orientation_xyzw),), current
                )
                self._send_trajectory(trajectory.points_rad, speed_scale=speed_scale, deadline=deadline)
                current = self._transport.joint_state()

    def _send_trajectory(self, points, *, speed_scale, deadline):
        batch = getattr(self._transport, 'move_joint_trajectory', None)
        if callable(batch):
            if not batch(points, speed_scale=speed_scale, timeout_s=self._remaining(deadline)):
                raise TimeoutError('Piper joint trajectory did not arrive')
            return
        for joints in points:
            if not self._transport.move_joint_goal(joints, speed_scale=speed_scale,
                                                 timeout_s=self._remaining(deadline)):
                raise TimeoutError('Piper joint goal did not arrive')

    def _close_for_grasp(self, apple, profile, effort, deadline):
        close = getattr(self._transport, 'close_for_grasp', None)
        if callable(close):
            return close(apple, profile, effort, timeout_s=self._remaining(deadline))
        return self._transport.set_gripper_width(apple.diameter_m, effort,
                                                timeout_s=self._remaining(deadline))

    def _confirm_holding(self, apple, apple_in_tcp, deadline):
        confirm = getattr(self._transport, 'confirm_holding', None)
        held = callable(confirm) and confirm(apple, apple_in_tcp, timeout_s=self._remaining(deadline)) is True
        if not held and callable(confirm) and self._cfg['execution'].get('holding_check', 'stop') == 'warn':
            # 10/9·10/10 실기: 실제로 쥐고 있는데 카메라 확인이 두 번 '안 잡음' (반대 경우는 없음).
            # 놓쳤다면 빈 그리퍼로 상자에 가서 여는 것뿐이라 경고만 하고 계속한다
            logging.getLogger(__name__).warning("apple %s: 들기 확인이 '안 잡음' — holding_check=warn 이라 계속",
                                                apple.apple_id)
            return True
        return held

    def _set_payload(self, apple_in_tcp, apple, **kwargs):
        setter = getattr(self._segment_validator, 'set_payload', None)
        if callable(setter):
            setter(apple_in_tcp, apple.diameter_m / 2, **kwargs)

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


def _transfer_settings(value):
    if value is None or value.get('height_m') is None:
        return None
    height = value['height_m']
    tilt = value.get('outward_tilt_rad', 0.0)
    for name, number in (('height_m', height), ('outward_tilt_rad', tilt)):
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
            raise ValueError(f'execution.transfer.{name} must be a finite number')
    if height <= 0:
        raise ValueError('execution.transfer.height_m must be positive')
    if not -math.pi/2 < tilt < math.pi/2:
        raise ValueError('execution.transfer.outward_tilt_rad must be within ±90°')
    return dict(height_m=float(height), outward_tilt_rad=float(tilt))


def transfer_plane_poses(current: Pose3D | None, target: Pose3D, *, height_m: float,
                         outward_tilt_rad: float) -> tuple[Pose3D, ...]:
    """Waypoints: above the current XY on the plane, then above the target XY.

    The plane orientation is the target orientation tilted by ``outward_tilt_rad``
    about the horizontal axis normal to the target's radial direction (tool tip
    outward from the base), then yawed with each waypoint's bearing. Base-frame
    only, so it does not depend on the tool RPY convention.
    """
    if (current is not None and target.position_m[2] >= height_m
            and math.dist(current.position_m[:2], target.position_m[:2]) < 1e-3):
        return ()
    bearing = math.atan2(target.position_m[1], target.position_m[0])
    axis = (-math.sin(bearing), math.cos(bearing), 0.0)
    tilted = _quat_multiply(_axis_quat(axis, -outward_tilt_rad), target.orientation_xyzw)

    def at(xy):
        turn = math.atan2(xy[1], xy[0]) - bearing
        return Pose3D((float(xy[0]), float(xy[1]), height_m),
                      _quat_multiply(_axis_quat((0.0, 0.0, 1.0), turn), tilted))
    above = at(target.position_m[:2])
    return (above,) if current is None else (at(current.position_m[:2]), above)


def _axis_quat(axis, angle):
    half = angle / 2
    return (axis[0]*math.sin(half), axis[1]*math.sin(half), axis[2]*math.sin(half), math.cos(half))


def _quat_multiply(left, right):
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (lw*rx + lx*rw + ly*rz - lz*ry, lw*ry - lx*rz + ly*rw + lz*rx,
            lw*rz + lx*ry - ly*rx + lz*rw, lw*rw - lx*rx - ly*ry - lz*rz)


def build_piper_motion_adapter(
    studio,
    cfg: dict,
    ik_solver: IKSolver,
    segment_validator: SegmentValidator,
    *,
    tool_pose_provider=None,
    in_hand_capture_provider=None,
    holding_verifier=None,
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

    from aple.execution.timed_motion import MotionLimits
    limits_cfg = cfg['execution'].get('motion')
    # Timed motion needs the whole measured set; live preflight refuses a partial one.
    keys = ('velocity_limits_rad_s', 'acceleration_limits_rad_s2', 'command_period_s',
            'tracking_tolerance_rad', 'velocity_tolerance_rad_s')
    limits = (MotionLimits.from_config(limits_cfg)
              if limits_cfg and all(limits_cfg.get(k) is not None for k in keys) else None)
    transport = PiperStudioMotionTransport(
        studio,
        max_joint_step_rad=cfg["execution"]["trajectory"]["max_joint_step_rad"],
        tool_pose_provider=tool_pose_provider,
        in_hand_capture_provider=in_hand_capture_provider,
        motion_limits=limits,
        holding_verifier=holding_verifier,
    )
    return PiperTrajectoryMotionAdapter(transport, cfg, ik_solver, segment_validator)
