"""Connect perception outputs to role-B grasp, placement, and execution planning.

This module consumes the existing schema objects. Robot motion is optional and
is available only through an explicitly injected motion adapter.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, replace, field
from copy import deepcopy
import time
from typing import Callable, Sequence

import numpy as np

from aple.inspection.coloration import InHandView, InspectParams, accumulate_views
from aple.inspection.geometry import Sphere
from aple.inspection.grade_classifier import classify
from aple.inspection.inhand import InHandSlip, SlipCheck, apple_center_cam, check_slip
from aple.execution.goal_builder import Pose3D
from aple.execution.motion_executor import (
    GraspCandidate,
    InspectionExecutionResult,
    MotionAdapter,
    MotionExecutor,
    PickExecutionResult,
    ViewCheck,
    ViewVerdict,
)
from aple.inspection.in_hand_capture import InHandCapture
from aple.orchestration.recovery import RecoveryAction, RecoveryDecision, RecoveryPolicy
from aple.planning.grasp_profile import select as select_grasp_profile
from aple.planning.placement_planner import PlacementPlanner, PlacementVerification
from aple.planning.task_planner import plan_queue
from aple.schema import (
    AppleState,
    BoxOccupancy,
    CaptureMode,
    ExecResult,
    FailureCode,
    Grade,
    GradeResult,
    GraspProfile,
    InspectMode,
    InspectionPolicy,
    PlacementPlan,
    RejectReason,
    TaskAction,
    TaskItem,
    Source,
    Phase,
)

_LOG = logging.getLogger(__name__)

@dataclass(frozen=True)
class AppleWorkPlan:
    """Joined inputs and derived role-B work for one apple."""

    apple: AppleState
    task: TaskItem
    profile: GraspProfile
    grasp_candidate: GraspCandidate | None
    placement: PlacementPlan | None
    execution: ExecResult | None = None
    final_grade: GradeResult | None = None
    timings: dict[str, float] = field(default_factory=dict)
    placement_verification: PlacementVerification | None = None


@dataclass(frozen=True)
class SceneWorkPlan:
    """Ordered work items derived from one perception scene."""

    scene_id: int
    items: tuple[AppleWorkPlan, ...]


@dataclass(frozen=True)
class HeldAppleWork:
    """State carried between role-B pick/inspect/place stages."""

    item: AppleWorkPlan
    pick: PickExecutionResult
    inspection: InspectionExecutionResult | None = None
    regraded_grade: GradeResult | None = None
    slip: SlipCheck | None = None  # set when a per-view check stopped the inspection


@dataclass(frozen=True)
class RegradeOutcome:
    """Result of a regrade attempt that reports in-hand slip instead of raising.

    On success ``held`` carries ``regraded_grade``. On slip ``failure_code`` is
    ``FailureCode.SLIP`` and ``decision`` is the bounded recovery intent; the
    caller owns any actual re-grasp or safe-stop motion.
    """

    held: HeldAppleWork
    failure_code: FailureCode = FailureCode.NONE
    view_index: int | None = None
    slip: SlipCheck | None = None
    decision: RecoveryDecision | None = None


@dataclass(frozen=True)
class SlipRecoveryResult:
    """What the slip recovery did and what the caller must do next.

    ``stop`` means the cycle must halt (the gripper was NOT opened in the air).
    ``rescan`` means the apple was set back on the tray: recapture the top view and
    plan again, reusing the top-view grade. ``held`` has the mode-B results discarded.
    """

    action: RecoveryAction
    success: bool
    failure_code: FailureCode
    alert: bool
    stop: bool
    rescan: bool
    held: HeldAppleWork
    placement: PlacementPlan | None = None


@dataclass(frozen=True)
class ModeBResult:
    """One mode-B apple through pick → inspect → regrade → place, including slip handling."""

    held: HeldAppleWork
    item: AppleWorkPlan | None = None              # placed item (success path)
    failure_code: FailureCode = FailureCode.NONE
    slip: RegradeOutcome | None = None
    recovery: SlipRecoveryResult | None = None


class PickPlacePipeline:
    """Join scene signals by the shared ``(scene_id, apple_id)`` contract."""

    def __init__(
        self,
        cfg: dict,
        motion_adapter: MotionAdapter | None = None,
        *,
        inspection_begin: Callable[[AppleState, Pose3D], None] | None = None,
    ) -> None:
        """``inspection_begin`` 은 모드 B 검사 직전에 (apple, apple_in_tcp) 로 불리는 훅이다.

        InHandCaptureProvider.begin 을 넘기면 뷰 촬영 전에 사과 크기·파지 위치를 알려 준다.
        """
        from aple.execution.gripper_geometry import apply_attachment_geometry
        cfg = apply_attachment_geometry(cfg)
        if inspection_begin is not None and not callable(inspection_begin):
            raise TypeError("inspection_begin must be callable")
        self._inspection_begin = inspection_begin
        self._cfg = cfg
        self._executor = (
            MotionExecutor(motion_adapter, cfg) if motion_adapter is not None else None
        )
        self._placement_planner = PlacementPlanner(cfg)

    def plan_scene(
        self,
        apples: Sequence[AppleState],
        grades: Sequence[GradeResult],
        boxes: Sequence[BoxOccupancy],
        grasp_candidates: Sequence[GraspCandidate],
    ) -> SceneWorkPlan:
        """Build grasp and placement plans without sending robot commands."""
        if not apples:
            raise ValueError("apples must contain at least one AppleState")

        scene_ids = {apple.scene_id for apple in apples}
        scene_ids.update(box.scene_id for box in boxes)
        scene_ids.update(candidate.scene_id for candidate in grasp_candidates)
        if len(scene_ids) != 1:
            raise ValueError("apples, boxes, and grasp candidates must belong to the same scene")
        scene_id = next(iter(scene_ids))
        if any(grade.scene_id > scene_id for grade in grades):
            raise ValueError("grade scene_id cannot be newer than the apple scene")
        reuse_limit = self._cfg["inspection"]["reuse"]["max_displacement_m"]
        if (
            not math.isfinite(float(reuse_limit))
            or float(reuse_limit) < 0
        ):
            raise ValueError("inspection.reuse.max_displacement_m must be finite and non-negative")

        candidate_options: dict[tuple[int, int], list[GraspCandidate]] = {}
        for candidate in grasp_candidates:
            if not isinstance(candidate.collision_free, bool):
                raise ValueError("candidate.collision_free must be bool")
            if (
                not math.isfinite(candidate.min_clearance_m)
                or candidate.min_clearance_m < 0
                or not math.isfinite(candidate.score)
            ):
                raise ValueError("candidate clearance and score must be finite and valid")
            key = (candidate.scene_id, candidate.apple_id)
            candidate_options.setdefault(key, []).append(candidate)
        candidates_by_key = {
            key: max(
                options,
                key=lambda candidate: (
                    candidate.collision_free,
                    candidate.min_clearance_m,
                    candidate.score,
                ),
            )
            for key, options in candidate_options.items()
        }
        unknown_candidate_keys = candidates_by_key.keys() - {
            (apple.scene_id, apple.apple_id) for apple in apples
        }
        if unknown_candidate_keys:
            raise ValueError("grasp_candidates must match an AppleState in the scene")

        profiles = {
            (apple.scene_id, apple.apple_id): select_grasp_profile(apple, self._cfg)
            for apple in apples
        }
        candidate_clearance = {
            key: candidate.collision_free for key, candidate in candidates_by_key.items()
        }
        tasks = plan_queue(
            list(apples),
            list(grades),
            list(profiles.values()),
            candidate_clearance=candidate_clearance,
            max_grade_reuse_displacement_m=float(reuse_limit),
        )
        grades_by_key = {(grade.scene_id, grade.apple_id): grade for grade in grades}
        apples_by_key = {
            (apple.scene_id, apple.apple_id): apple for apple in apples
        }
        work_items: list[AppleWorkPlan] = []
        # Preview reservations are local to this plan. Repeated dry runs must
        # not consume real cells. Executed cells are claimed immediately before motion.
        preview_planner = deepcopy(self._placement_planner)

        for task in tasks:
            key = (task.scene_id, task.apple_id)
            profile = profiles[key]
            grade = grades_by_key.get((task.grade_scene_id, task.apple_id))
            placement = None
            if task.action == TaskAction.PICK_PLACE:
                if grade is None:
                    raise ValueError(
                        f"task for apple {task.apple_id} has no matching GradeResult"
                    )
                placement = preview_planner.plan(
                    apples_by_key[key], grade, list(boxes)
                )
            work_items.append(
                AppleWorkPlan(
                    apples_by_key[key],
                    task,
                    profile,
                    candidates_by_key.get(key),
                    placement,
                )
            )

        return SceneWorkPlan(scene_id, tuple(work_items))

    def execute_scene(
        self, work_plan: SceneWorkPlan, *, backend: str | None = None,
        inspect_params: InspectParams | None = None,
        boxes: Sequence[BoxOccupancy] = (),
        rescan_for_retry: Callable[
            [AppleWorkPlan], tuple[AppleWorkPlan, Sequence[BoxOccupancy]]
        ] | None = None,
    ) -> SceneWorkPlan:
        """Execute one-apple PICK_PLACE actions in order, stopping on first failure.

        Mode B is enabled by inspect_params. Its first slip may recapture and
        replan through rescan_for_retry; the second slip rejects the apple.
        After a rescan, remaining plans from the old scene are deferred until
        the next cycle. Without inspect_params, mode-B items remain untouched.
        """
        if self._executor is None:
            raise ValueError("a motion_adapter is required to execute a scene")

        completed: list[AppleWorkPlan] = []
        stopped = False
        scene_invalidated = False
        for item in work_plan.items:
            if stopped:
                completed.append(item)
                continue
            if item.task.action == TaskAction.PICK_INSPECT_PLACE and inspect_params is not None:
                started = time.monotonic()
                executed, rescanned = self._execute_mode_b(
                    item, inspect_params, boxes, rescan_for_retry, backend=backend
                )
                executed = replace(executed, timings={**executed.timings, "t_total_s": time.monotonic()-started})
                completed.append(executed)
                scene_invalidated = rescanned
                stopped = rescanned or not executed.execution.success
                continue
            if item.task.action != TaskAction.PICK_PLACE:
                completed.append(item)
                continue
            executed = self.execute_item(item, backend=backend)
            completed.append(executed)
            stopped = (
                executed.execution is not None
                and not executed.execution.success
            )

        return replace(work_plan, items=tuple(completed))

    def _execute_mode_b(
        self, item: AppleWorkPlan, params: InspectParams,
        boxes: Sequence[BoxOccupancy],
        rescan_for_retry: Callable[
            [AppleWorkPlan], tuple[AppleWorkPlan, Sequence[BoxOccupancy]]
        ] | None,
        *, backend: str | None,
    ) -> tuple[AppleWorkPlan, bool]:
        recovery = RecoveryPolicy(self._cfg)
        rescanned = False
        original = item
        for slip_count in range(2):
            outcome = self.run_mode_b_item(
                item, params, recovery, boxes,
                slip_retry_count=slip_count, backend=backend,
            )
            if outcome.item is not None:
                return outcome.item, rescanned
            result = outcome.recovery
            if result is not None and result.success and not result.rescan:
                grade = self._reject_grade(outcome.held)
                _LOG.warning("apple %d rejected after repeated in-hand slip", item.apple.apple_id)
                return replace(
                    item, placement=result.placement, final_grade=grade,
                    task=replace(item.task, grade=Grade.REJECT),
                    execution=ExecResult(True, FailureCode.NONE, Phase.RETREAT),
                ), rescanned
            if result is not None and result.success and result.rescan and slip_count == 0:
                rescanned = True
                if rescan_for_retry is not None:
                    item, boxes = rescan_for_retry(original)
                    if (item.apple.apple_id != original.apple.apple_id
                            or item.apple.scene_id <= original.apple.scene_id
                            or item.task.action != TaskAction.PICK_INSPECT_PLACE
                            or item.task.grade != original.task.grade):
                        raise ValueError("slip retry requires a fresh plan for the same apple and top-view grade")
                    continue
                _LOG.warning("apple returned to tray; rescan provider is required before retry")
            failure = result.failure_code if result is not None and not result.success else outcome.failure_code
            if result is not None:
                phase = Phase.RETREAT if result.success else Phase.INSPECT
            elif outcome.held.inspection is not None:
                phase = outcome.held.inspection.phase_reached
            else:
                phase = outcome.held.pick.phase_reached
            return replace(item, execution=ExecResult(False, failure, phase)), rescanned
        raise RuntimeError("mode-B retry exceeded its single-regrasp limit")

    def execute_item(
        self, item: AppleWorkPlan, *, backend: str | None = None
    ) -> AppleWorkPlan:
        """Execute at most one apple's complete pick-and-place action.

        This is the policy boundary described in the idea summary: planning
        selects one apple and one destination, then the chosen backend performs
        that single action. Non-PICK_PLACE tasks are returned untouched for the
        orchestration layer to handle.
        """
        if item.task.action != TaskAction.PICK_PLACE:
            return item
        if self._executor is None:
            raise ValueError("a motion_adapter is required to execute an item")
        if item.placement is None or not item.placement.feasible:
            return item
        if item.grasp_candidate is None:
            raise ValueError(
                f"PICK_PLACE task for apple {item.apple.apple_id} has no grasp candidate"
            )

        self._placement_planner.reserve(item.placement)
        started = time.monotonic()
        result = self._executor.pick_place(
            item.apple,
            item.profile,
            item.grasp_candidate,
            item.placement,
            backend=backend,
        )
        return replace(item, execution=result,
                       timings={**self._executor.stage_timings, "t_total_s": time.monotonic() - started})

    def verify_item(self, item: AppleWorkPlan, before: Sequence[BoxOccupancy],
                    after: Sequence[BoxOccupancy]) -> AppleWorkPlan:
        """Confirm measured placement before allowing the next apple to move."""
        if item.placement is None or item.execution is None or not item.execution.success:
            return item
        old = next((b for b in before if b.box_id == item.placement.box_id), None)
        new = next((b for b in after if b.box_id == item.placement.box_id), None)
        verification = None
        if old is not None and new is not None:
            verification = self._placement_planner.verify(item.placement, item.apple, old, new)
        ok = verification is not None and verification.verified and verification.success
        return replace(item, placement_verification=verification,
                       execution=item.execution if ok else ExecResult(False, FailureCode.ABORTED, Phase.RETREAT))

    def pick_for_inspection(
        self, item: AppleWorkPlan, *, backend: str | None = None
    ) -> HeldAppleWork:
        """Run the mode-B pick stage and retain the measured apple-in-TCP transform."""
        if item.task.action != TaskAction.PICK_INSPECT_PLACE:
            raise ValueError("item must have action PICK_INSPECT_PLACE")
        if self._executor is None:
            raise ValueError("a motion_adapter is required to execute a scene")
        if item.grasp_candidate is None:
            raise ValueError("mode-B task has no collision-free grasp candidate")
        result = self._executor.pick_for_inspection(
            item.apple,
            item.profile,
            item.grasp_candidate,
            backend=backend,
        )
        return HeldAppleWork(replace(item, timings={**item.timings, **self._executor.stage_timings}), result)

    def inspect_held_apple(self, held: HeldAppleWork) -> HeldAppleWork:
        """Run configured in-hand views; the adapter emits capture data for inspection."""
        if self._executor is None:
            raise ValueError("a motion_adapter is required to execute a scene")
        if (
            not held.pick.success
            or held.pick.apple_in_tcp is None
            or held.pick.apple_center_in_base is None
        ):
            raise ValueError("pick stage must succeed and measure the held apple transform")
        if self._inspection_begin is not None:
            self._inspection_begin(held.item.apple, held.pick.apple_in_tcp)
        view_check, slip_by_view = self._build_view_check(held)
        result = self._executor.inspect_for_regrade(
            held.item.profile,
            held.pick.apple_in_tcp,
            held.pick.apple_center_in_base,
            view_check=view_check,
        )
        slip = None
        if result.slip_view_index is not None:
            slip = slip_by_view.get(result.slip_view_index)
            _LOG.warning(
                "in-hand slip at view %d: inspection stopped, apple is still held",
                result.slip_view_index,
            )
        return replace(held, item=replace(held.item, timings={**held.item.timings, **self._executor.stage_timings}),
                       inspection=result, slip=slip)

    def _build_view_check(
        self, held: HeldAppleWork
    ) -> tuple[ViewCheck | None, dict[int, SlipCheck]]:
        """Per-view slip check run by the adapter right after each capture (~2 ms/view).

        None when ``workcell.T_base_camera`` is unset (no pose sphere to compare against).
        A retaken view overwrites its entry, so the dict ends with the decisive check.
        """
        sphere_source = self._pose_sphere_source(held, warn=False)
        slip_by_view: dict[int, SlipCheck] = {}
        if sphere_source is None:
            return None, slip_by_view

        def view_check(index: int, capture: InHandCapture) -> ViewVerdict:
            slip = self._check_view_slip(sphere_source, capture)
            slip_by_view[index] = slip
            if slip.ok:
                return ViewVerdict.OK
            return ViewVerdict.UNVERIFIED if slip.observed_center is None else ViewVerdict.SLIP

        return view_check, slip_by_view

    def _check_view_slip(
        self,
        sphere_source: tuple[np.ndarray, np.ndarray, float],
        capture: InHandCapture,
    ) -> SlipCheck:
        snapshot = capture.snapshot
        if snapshot.tool_pose is None:
            raise ValueError("in-hand snapshot must include the measured TCP tool_pose")
        T_base_camera, apple_in_tcp_m, radius_m = sphere_source
        center = apple_center_cam(
            T_base_camera, np.asarray(snapshot.tool_pose, float), apple_in_tcp_m
        )
        return check_slip(
            snapshot.depth_m, capture.mask, snapshot.intrinsics, center, radius_m,
            **self._cfg["inspection"]["slip_check"],
        )

    def regrade_held_apple(
        self, held: HeldAppleWork, params: InspectParams
    ) -> HeldAppleWork:
        """Accumulate captured views and classify using the existing mode-B pipeline."""
        started = time.monotonic()
        inspection = held.inspection
        if inspection is None or not inspection.success:
            raise ValueError("a successful in-hand inspection is required before regrading")
        if not inspection.captures:
            raise ValueError("inspection backend returned no image/depth capture payloads")
        if len(inspection.captures) != inspection.captured_views:
            raise ValueError("inspection capture payload count does not match captured_views")

        captures = inspection.captures
        sphere_source = self._pose_sphere_source(held)
        previous_scene_id = held.item.apple.scene_id
        views: list[InHandView] = []
        for index, capture in enumerate(captures):
            snapshot = capture.snapshot
            if snapshot.source != Source.IN_HAND or snapshot.capture_mode != CaptureMode.FULL:
                raise ValueError("mode-B capture must be a FULL IN_HAND SceneSnapshot")
            if snapshot.scene_id <= previous_scene_id:
                raise ValueError("in-hand scene_ids must increase beyond the source scene")
            if snapshot.apple_id != held.item.apple.apple_id or snapshot.view_index != index:
                raise ValueError("in-hand snapshot apple_id/view_index does not match the capture")
            if snapshot.tool_pose is None:
                raise ValueError("in-hand snapshot must include the measured TCP tool_pose")
            previous_scene_id = snapshot.scene_id
            sphere = None
            if sphere_source is not None:
                T_base_camera, apple_in_tcp_m, radius_m = sphere_source
                center = apple_center_cam(
                    T_base_camera, np.asarray(snapshot.tool_pose, float), apple_in_tcp_m
                )
                slip = self._check_view_slip(sphere_source, capture)
                if not slip.ok:
                    raise InHandSlip(index, slip)
                sphere = Sphere(center, radius_m)
            views.append(
                InHandView(
                    snapshot.color,
                    snapshot.depth_m,
                    snapshot.intrinsics,
                    capture.mask,
                    capture.rotation_camera_apple,
                    sphere,
                )
            )

        ratio = accumulate_views(
            views,
            params,
            scene_id=captures[-1].snapshot.scene_id,
            apple_id=held.item.apple.apple_id,
        )
        if ratio is None:
            raise ValueError("no usable in-hand views were available for regrading")
        mm_range = self._cfg["size_bins"]["gripper_range_mm"]
        grade = classify(
            ratio,
            params,
            apple=held.item.apple,
            policy=InspectionPolicy[
                self._cfg["inspection"].get("policy", "NEVER")
            ],
            gripper_range_mm=(float(mm_range[0]), float(mm_range[1])),
        )
        return replace(held, regraded_grade=grade, item=replace(held.item, timings={
            **held.item.timings, 't_inspect_s': held.item.timings.get('t_inspect_s', 0.) + time.monotonic()-started}))

    def regrade_with_slip_decision(
        self,
        held: HeldAppleWork,
        params: InspectParams,
        recovery: RecoveryPolicy,
        *,
        slip_retry_count: int = 0,
    ) -> RegradeOutcome:
        """Regrade, converting only ``InHandSlip`` into a SLIP recovery intent.

        Other ``ValueError`` s (bad captures, bad calibration) still propagate.
        A per-view check inside ``inspect_held_apple`` stops the inspection at the
        first slipped view; that is reported here without regrading. Either way the
        apple is still held when this returns a slip outcome.
        """
        inspection = held.inspection
        if inspection is not None and inspection.slip_view_index is not None:
            index = inspection.slip_view_index
            return RegradeOutcome(
                held,
                FailureCode.SLIP,
                index,
                held.slip,
                self._slip_decision(recovery, slip_retry_count),
            )
        try:
            return RegradeOutcome(self.regrade_held_apple(held, params))
        except InHandSlip as exc:
            _LOG.warning("in-hand slip at view %d: %s", exc.view_index, exc)
            return RegradeOutcome(
                held,
                FailureCode.SLIP,
                exc.view_index,
                exc.check,
                self._slip_decision(recovery, slip_retry_count),
            )

    def _slip_decision(
        self, recovery: RecoveryPolicy, slip_retry_count: int
    ) -> RecoveryDecision:
        lost = self._executor is not None and self._executor.grip_lost()
        return recovery.decide(
            failure_code=FailureCode.SLIP,
            slip_retry_count=slip_retry_count,
            gripper_lost=lost,
        )

    def recover_from_slip(
        self,
        held: HeldAppleWork,
        decision: RecoveryDecision,
        boxes: Sequence[BoxOccupancy],
        *,
        backend: str | None = None,
    ) -> SlipRecoveryResult:
        """Carry out a slip recovery decision with the apple still in the gripper.

        Common rules: every move is slow (``execution.recovery.speed_scale``), and the
        gripper is never opened in the air. Setting the apple down reuses
        ``place_held_apple`` whose first move returns the tool to the (vertical) grasp
        posture above the target before descending.

        * ``RETRY_GRASP_ONCE``: set it back on its original tray spot (empty), drop ≤
          ``max_drop_m``; the caller then recaptures the top view and re-plans/re-grasps
          with the top-view grade kept (mode-B results are discarded).
        * ``PLACE_REJECT_AND_ALERT``: set it in the reject box ("uncertain → reject") and go on.
        * ``SAFE_STOP_AND_ALERT`` or fingers already closed (apple lost): no motion, stop.
        """
        if self._executor is None:
            raise ValueError("a motion_adapter is required to execute a scene")
        if decision.action not in (
            RecoveryAction.RETRY_GRASP_ONCE,
            RecoveryAction.PLACE_REJECT_AND_ALERT,
            RecoveryAction.SAFE_STOP_AND_ALERT,
        ):
            raise ValueError(f"{decision.action.value} is not a slip recovery action")
        if held.pick.apple_in_tcp is None:
            raise ValueError("pick stage did not produce apple_in_tcp")
        discarded = replace(held, inspection=None, regraded_grade=None, slip=None)

        def stop(code: FailureCode, why: str) -> SlipRecoveryResult:
            _LOG.error("slip recovery stops the cycle: %s", why)
            return SlipRecoveryResult(
                RecoveryAction.SAFE_STOP_AND_ALERT, False, code, True, True, False, discarded
            )

        if decision.action == RecoveryAction.SAFE_STOP_AND_ALERT:
            return stop(FailureCode.DROP if self._executor.grip_lost() else FailureCode.SLIP,
                        "safe stop requested")
        if self._executor.grip_lost():
            return stop(FailureCode.DROP, "gripper is fully closed; the apple is already lost")

        retry = decision.action == RecoveryAction.RETRY_GRASP_ONCE
        placement = (
            self._original_spot_plan(held.item.apple, held.item.task.grade)
            if retry
            else self._reject_plan(held, boxes)
        )
        if not placement.feasible:
            return stop(FailureCode.SLIP, "no feasible place to set the apple down")
        result = self._executor.place_held_apple(
            held.item.apple,
            held.item.profile,
            held.pick.apple_in_tcp,
            placement,
            backend=backend,
            speed_scale=self._executor.recovery_speed_scale(),
        )
        if not result.success:
            return stop(result.failure_code, "slow set-down failed with the apple still held")
        return SlipRecoveryResult(
            decision.action, True, FailureCode.NONE, decision.alert, False, retry,
            discarded, placement,
        )

    def _original_spot_plan(self, apple: AppleState, grade: Grade | None) -> PlacementPlan:
        """Set-down plan on the apple's original (now empty) tray position."""
        cfg = self._cfg["workcell"]
        placement_cfg = cfg["placement"]
        max_drop_m = float(placement_cfg["max_drop_m"])
        margin_m = float(placement_cfg["margin_m"])
        release_gap_m = float(placement_cfg["release_gap_m"])
        wall_m = float(cfg["tray"]["wall_height_m"])
        if not all(math.isfinite(v) for v in (max_drop_m, margin_m, release_gap_m, wall_m)):
            raise ValueError("placement and tray distances must be finite")
        original_grade = grade if grade is not None else Grade.REJECT
        x, y, z = apple.center_m
        radius_m = apple.diameter_m / 2.0
        if release_gap_m > max_drop_m:
            return PlacementPlan(
                apple.scene_id, apple.apple_id, original_grade, -1,
                apple.center_m, z, z, 0.0, False,
            )
        release_z = z + release_gap_m
        hover_z = max(wall_m + radius_m + margin_m, release_z + margin_m)
        return PlacementPlan(
            scene_id=apple.scene_id,
            apple_id=apple.apple_id,
            box_id=original_grade,
            cell_index=-1,                       # -1 = 트레이 원래 자리 (상자 칸 아님)
            place_center_m=(x, y, release_z),
            hover_z_m=hover_z,
            release_z_m=release_z,
            planned_drop_m=release_gap_m,
            feasible=True,
        )

    def _reject_plan(
        self, held: HeldAppleWork, boxes: Sequence[BoxOccupancy]
    ) -> PlacementPlan:
        return self._placement_planner.plan(held.item.apple, self._reject_grade(held), list(boxes))

    def _reject_grade(self, held: HeldAppleWork) -> GradeResult:
        apple = held.item.apple
        return GradeResult(
            apple.scene_id, apple.apple_id, Grade.REJECT, 0.0, 0.0, 0.0, 0.0, 0.0,
            RejectReason.VIEW_DISAGREE, True, InspectMode.IN_HAND_MULTI,
            self._cfg["inspection"]["default_variety"],
        )

    def run_mode_b_item(
        self,
        item: AppleWorkPlan,
        params: InspectParams,
        recovery: RecoveryPolicy,
        boxes: Sequence[BoxOccupancy],
        *,
        slip_retry_count: int = 0,
        backend: str | None = None,
    ) -> ModeBResult:
        """Pick → in-hand inspection (per-view slip check) → regrade → place, one apple.

        On slip, the recovery decision is carried out here (see ``recover_from_slip``).
        ``slip_retry_count`` is the caller's count of earlier slips for this apple.
        The caller must honour ``recovery.stop`` / ``recovery.rescan`` in the result.
        """
        held = self.pick_for_inspection(item, backend=backend)
        if not held.pick.success:
            return ModeBResult(held, failure_code=held.pick.failure_code)
        held = self.inspect_held_apple(held)
        inspection = held.inspection
        if not inspection.success and inspection.slip_view_index is None:
            return ModeBResult(held, failure_code=inspection.failure_code)
        outcome = self.regrade_with_slip_decision(
            held, params, recovery, slip_retry_count=slip_retry_count
        )
        if outcome.failure_code == FailureCode.SLIP:
            result = self.recover_from_slip(outcome.held, outcome.decision, boxes, backend=backend)
            return ModeBResult(result.held, failure_code=FailureCode.SLIP, slip=outcome, recovery=result)
        held = outcome.held
        placed = self.place_regraded_apple(held, held.regraded_grade, boxes, backend=backend)
        failure = placed.execution.failure_code if placed.execution is not None else FailureCode.NONE
        return ModeBResult(held, item=placed, failure_code=failure)

    def _pose_sphere_source(
        self, held: HeldAppleWork, *, warn: bool = True
    ) -> tuple[np.ndarray, np.ndarray, float] | None:
        """Sphere inputs from robot pose; None keeps the silhouette fit (no calibration yet).

        Fingers hide the apple's sides, so a silhouette fit underestimates the radius
        and inflates coverage. With T_base_camera the center is
        inv(T_base_camera) @ tool_pose @ apple_in_tcp and the radius is diameter_m / 2.
        """
        T_base_camera = self._cfg.get("workcell", {}).get("T_base_camera")
        if T_base_camera is None:
            if warn:
                _LOG.warning(
                    "workcell.T_base_camera is not set; mode-B regrade uses the silhouette "
                    "sphere, which overstates coverage when fingers hide the apple"
                )
            return None
        T = np.asarray(T_base_camera, float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ValueError("workcell.T_base_camera must be a finite 4x4 matrix")
        if held.pick.apple_in_tcp is None:
            raise ValueError("pick stage did not produce apple_in_tcp")
        apple_in_tcp_m = np.asarray(held.pick.apple_in_tcp.position_m, float)
        if apple_in_tcp_m.shape != (3,) or not np.all(np.isfinite(apple_in_tcp_m)):
            raise ValueError("pick.apple_in_tcp.position_m must contain three finite values")
        radius_m = float(held.item.apple.diameter_m) / 2.0
        if not math.isfinite(radius_m) or radius_m <= 0:
            raise ValueError("apple.diameter_m must be finite and positive for the pose sphere")
        return T, apple_in_tcp_m, radius_m

    def place_regraded_apple(
        self,
        held: HeldAppleWork,
        grade: GradeResult,
        boxes: Sequence[BoxOccupancy],
        *,
        backend: str | None = None,
    ) -> AppleWorkPlan:
        """Place a mode-B apple after the inspection module supplies a final grade."""
        if self._executor is None:
            raise ValueError("a motion_adapter is required to execute a scene")
        if held.inspection is None or not held.inspection.success:
            raise ValueError("a successful in-hand inspection is required before placement")
        if held.pick.apple_in_tcp is None:
            raise ValueError("pick stage did not produce apple_in_tcp")
        if not grade.final:
            raise ValueError("grade must be final before planning placement")
        if grade.apple_id != held.item.apple.apple_id:
            raise ValueError("grade apple_id does not match the held apple")
        if held.regraded_grade is not None and grade != held.regraded_grade:
            raise ValueError("grade does not match this held apple's in-hand regrade")
        if held.inspection is not None and held.inspection.captures:
            capture_scene_id = held.inspection.captures[-1].snapshot.scene_id
            if grade.scene_id != capture_scene_id:
                raise ValueError("grade scene_id must match the latest in-hand capture")
        elif grade.scene_id > held.item.apple.scene_id:
            raise ValueError("a newer grade scene requires in-hand capture provenance")

        placement = self._placement_planner.plan(
            held.item.apple,
            grade,
            list(boxes),
            allow_newer_grade=True,
        )
        result = self._executor.place_held_apple(
            held.item.apple,
            held.item.profile,
            held.pick.apple_in_tcp,
            placement,
            backend=backend,
        )
        task = replace(
            held.item.task,
            action=TaskAction.PICK_PLACE,
            grade=grade.grade,
            grade_scene_id=grade.scene_id,
        )
        return replace(
            held.item,
            task=task,
            placement=placement,
            execution=result,
            final_grade=grade,
            timings={**held.item.timings, **self._executor.stage_timings},
        )

    def release_placement_reservation(self, box_id: int, cell_index: int) -> None:
        """Release a reserved box cell only after external placement verification."""
        self._placement_planner.release(box_id, cell_index)
