"""Connect perception outputs to role-B grasp, placement, and execution planning.

This module consumes the existing schema objects. Robot motion is optional and
is available only through an explicitly injected motion adapter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Sequence

from aple.inspection.coloration import InHandView, InspectParams, accumulate_views
from aple.inspection.grade_classifier import classify
from aple.execution.motion_executor import (
    GraspCandidate,
    InspectionExecutionResult,
    MotionAdapter,
    MotionExecutor,
    PickExecutionResult,
)

from aple.planning.grasp_profile import select as select_grasp_profile
from aple.planning.placement_planner import PlacementPlanner
from aple.planning.task_planner import plan_queue
from aple.schema import (
    AppleState,
    BoxOccupancy,
    CaptureMode,
    ExecResult,
    GradeResult,
    GraspProfile,
    InspectionPolicy,
    PlacementPlan,
    TaskAction,
    TaskItem,
    Source,
)


@dataclass(frozen=True)
class AppleWorkPlan:
    """Joined inputs and derived role-B work for one apple."""

    apple: AppleState
    task: TaskItem
    profile: GraspProfile
    grasp_candidate: GraspCandidate | None
    placement: PlacementPlan | None
    execution: ExecResult | None = None


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


class PickPlacePipeline:
    """Join scene signals by the shared ``(scene_id, apple_id)`` contract."""

    def __init__(self, cfg: dict, motion_adapter: MotionAdapter | None = None) -> None:
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
                placement = self._placement_planner.plan(
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
        self, work_plan: SceneWorkPlan, *, backend: str | None = None
    ) -> SceneWorkPlan:
        """Execute one-apple PICK_PLACE actions in order, stopping on first failure.

        DEFER, SKIP, and PICK_INSPECT_PLACE items are retained for the caller;
        they are never sent to the motion adapter by this method.
        """
        if self._executor is None:
            raise ValueError("a motion_adapter is required to execute a scene")

        completed: list[AppleWorkPlan] = []
        stopped = False
        for item in work_plan.items:
            if stopped or item.task.action != TaskAction.PICK_PLACE:
                completed.append(item)
                continue
            executed = self.execute_item(item, backend=backend)
            completed.append(executed)
            stopped = (
                executed.execution is not None
                and not executed.execution.success
            )

        return replace(work_plan, items=tuple(completed))

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

        result = self._executor.pick_place(
            item.apple,
            item.profile,
            item.grasp_candidate,
            item.placement,
            backend=backend,
        )
        return replace(item, execution=result)

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
        return HeldAppleWork(item, result)

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
        result = self._executor.inspect_for_regrade(
            held.item.profile,
            held.pick.apple_in_tcp,
            held.pick.apple_center_in_base,
        )
        return replace(held, inspection=result)

    def regrade_held_apple(
        self, held: HeldAppleWork, params: InspectParams
    ) -> HeldAppleWork:
        """Accumulate captured views and classify using the existing mode-B pipeline."""
        inspection = held.inspection
        if inspection is None or not inspection.success:
            raise ValueError("a successful in-hand inspection is required before regrading")
        if not inspection.captures:
            raise ValueError("inspection backend returned no image/depth capture payloads")
        if len(inspection.captures) != inspection.captured_views:
            raise ValueError("inspection capture payload count does not match captured_views")

        captures = inspection.captures
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
            views.append(
                InHandView(
                    snapshot.color,
                    snapshot.depth_m,
                    snapshot.intrinsics,
                    capture.mask,
                    capture.rotation_camera_apple,
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
        return replace(held, regraded_grade=grade)

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
        )

    def release_placement_reservation(self, box_id: int, cell_index: int) -> None:
        """Release a reserved box cell only after external placement verification."""
        self._placement_planner.release(box_id, cell_index)
