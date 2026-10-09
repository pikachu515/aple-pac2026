"""장면 1회 실행기 — 카메라 → 인식 → 판정 → 계획 → 실행 → 기록을 CycleManager 단계로 잇는다. M4 통합.

CycleManager 는 capture / plan_scene / execute_scene / make_records 를 주입받는다. 이 모듈은 그
세 단계(plan, execute, records)를 각 역할의 모듈로 채운다.

  plan     조명 상태 → 검출(M1) → 3D 위치(M1) → 판정(M2) → 상자 점유(M1) → 파지 후보(M1) → 작업 계획(M3·M5)
  execute  작업 계획 실행(M3). 조명 FAULT 이면 실행하지 않는다 — 판정이 LOW_CONF 리젝트로 바뀌어
           멀쩡한 사과가 리젝트 상자로 가기 때문이다. 대신 복구 정책의 RECALIBRATE_LIGHT 를 남긴다
  records  사과마다 CycleRecord (발표 지표의 원천)

모든 단계 함수는 주입받는다. `from_config` 는 각 역할 모듈의 기본 구현으로 채운다
(M1 인식은 #13 머지 후 사용 가능). 테스트는 합성 장면용 가짜 단계로 같은 흐름을 검증한다.

  runner = SceneRunner.from_config(cfg, pipeline, light_for=card_monitor.status_for)
  CycleManager(cfg, capture=card_monitor.capture, plan_scene=runner.plan,
               execute_scene=runner.execute, make_records=runner.records).run(5)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable, Sequence

import numpy as np

from aple.execution.pick_place_pipeline import AppleWorkPlan, PickPlacePipeline, SceneWorkPlan
from aple.inspection.coloration import params_from_config
from aple.inspection.pipeline import inspect_scene
from aple.orchestration.recovery import RecoveryDecision, RecoveryPolicy, is_lighting_fault
from aple.schema import (AppleDetection2D, AppleState, BoxOccupancy, CycleRecord, FailStage, FailureCode,
                         Grade, GradeResult, InspectionPolicy, LightingStatus, RejectReason,
                         SceneSnapshot, TaskAction, Phase, Source, CaptureMode, ExecResult)

_LOG = logging.getLogger(__name__)

DetectFn = Callable[[SceneSnapshot], Sequence[AppleDetection2D]]
ApplesFn = Callable[[SceneSnapshot, Sequence[AppleDetection2D], Sequence[AppleState] | None], Sequence[AppleState]]
BoxesFn = Callable[[SceneSnapshot], Sequence[BoxOccupancy]]
GraspsFn = Callable[[Sequence[AppleState]], Sequence[object]]
LightFn = Callable[[SceneSnapshot], LightingStatus]


@dataclass(frozen=True)
class ScenePlan:
    """plan 단계 결과 — CycleManager 의 PlanT."""
    snapshot: SceneSnapshot
    light: LightingStatus | None
    detections: tuple[AppleDetection2D, ...]
    apples: tuple[AppleState, ...]
    grades: tuple[GradeResult, ...]
    boxes: tuple[BoxOccupancy, ...]
    work: SceneWorkPlan | None                 # None 이면 실행할 것이 없다 (아래 blocked)
    blocked: str | None = None                 # "lighting" | "no_apples"
    recovery: RecoveryDecision | None = None
    timings: dict[str, float] = field(default_factory=dict)


class SceneRunner:
    def __init__(self, cfg: dict, pipeline: PickPlacePipeline, *, detect: DetectFn, estimate_apples: ApplesFn,
                 estimate_boxes: BoxesFn, plan_grasps: GraspsFn, light_for: LightFn | None = None,
                 backend: str | None = None, clock: Callable[[], float] = time.monotonic,
                 capture_for_retry: Callable[[], SceneSnapshot] | None = None,
                 capture_for_verification: Callable[[], SceneSnapshot] | None = None,
                 before_execute: Callable | None = None) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self._detect, self._apples, self._boxes, self._grasps = detect, estimate_apples, estimate_boxes, plan_grasps
        self._light_for = light_for
        self._backend = cfg["execution"]["default_backend"] if backend is None else backend
        if self._backend not in ("scripted", "act"):
            raise ValueError("backend must be scripted or act")
        self._clock = clock
        self._recovery = RecoveryPolicy(cfg)
        self._params = params_from_config(cfg)
        self._policy = InspectionPolicy[cfg["inspection"].get("policy", "NEVER")]
        mm = cfg["size_bins"]["gripper_range_mm"]
        self._gripper_range_mm = (float(mm[0]), float(mm[1]))
        self._prev_apples: list[AppleState] | None = None
        if capture_for_retry is not None and not callable(capture_for_retry):
            raise TypeError("capture_for_retry must be callable")
        self._capture_for_retry = capture_for_retry
        self._capture_for_verification = capture_for_verification
        self._before_execute = before_execute
        self._execution_halted = False
        self.last_verified_snapshot = None
        self._verify_timeout = float(cfg["orchestration"].get("verify_timeout_s", 5.0))
        if not math.isfinite(self._verify_timeout) or self._verify_timeout <= 0:
            raise ValueError("verify_timeout_s must be finite and positive")

    @classmethod
    def from_config(cls, cfg: dict, pipeline: PickPlacePipeline, **kwargs) -> "SceneRunner":
        """각 역할 모듈의 기본 구현으로 채운다. M1(detector·pose_estimator·box_occupancy·grasp_planner)은 #13."""
        from aple.execution.gripper_geometry import apply_attachment_geometry
        cfg = apply_attachment_geometry(cfg)
        T = cfg["workcell"].get("T_base_camera")
        if T is None:
            raise ValueError("workcell.T_base_camera is not set — run camera–robot calibration first")
        T = np.asarray(T, float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ValueError("workcell.T_base_camera must be a finite 4x4 matrix")
        from aple.perception import box_occupancy, detector, grasp_planner, pose_estimator
        from aple.perception.workcell_detect import AutoWorkcell, auto_enabled
        if not auto_enabled(cfg):
            return cls(cfg, pipeline,
                       detect=lambda snap: detector.detect(snap, cfg),
                       estimate_apples=lambda snap, dets, prev: pose_estimator.estimate(snap, dets, T, prev, cfg),
                       estimate_boxes=kwargs.pop("estimate_boxes", lambda snap: box_occupancy.estimate(snap, T, cfg)),
                       plan_grasps=lambda apples: grasp_planner.plan_grasps(list(apples), cfg),
                       **kwargs)
        # workcell.auto_detect: 장면마다 상자·리젝트 구역·집는 영역을 카메라로 찾아 채운 설정 사본으로 검출·추정한다.
        # plan() 은 detect 를 먼저 부르므로 그때 갱신한 설정을 같은 장면의 위치·상자·파지 추정이 쓴다.
        auto = AutoWorkcell(cfg, T)

        def detect(snap):
            auto.update(snap)
            return detector.detect(snap, auto.cfg)
        runner = cls(cfg, pipeline,
                     detect=detect,
                     estimate_apples=lambda snap, dets, prev: pose_estimator.estimate(snap, dets, T, prev, auto.cfg),
                     estimate_boxes=kwargs.pop("estimate_boxes", lambda snap: box_occupancy.estimate(snap, T, auto.cfg)),
                     plan_grasps=lambda apples: grasp_planner.plan_grasps(list(apples), auto.cfg),
                     **kwargs)
        runner.auto_workcell = auto
        return runner

    # ------------------------------------------------------------ CycleManager 단계

    def plan(self, snap: SceneSnapshot) -> ScenePlan:
        t0 = self._clock()
        light = self._light_for(snap) if self._light_for is not None else None
        dets = tuple(self._detect(snap))
        apples = tuple(self._apples(snap, dets, self._prev_apples))
        t_scan = self._clock() - t0

        grades = tuple(inspect_scene(snap, list(dets), list(apples), self._params, light=light,
                                     policy=self._policy, gripper_range_mm=self._gripper_range_mm)) if dets else ()
        boxes = tuple(self._boxes(snap))
        if light is not None and is_lighting_fault(light):
            decision = self._recovery.decide(lighting_fault=True)
            _LOG.warning("lighting fault (ΔE %.1f) — not executing scene %d; %s",
                         light.delta_e, snap.scene_id, decision.action.value)
            return ScenePlan(snap, light, dets, apples, grades, boxes, None, "lighting", decision,
                             {"t_scan_s": t_scan, "t_plan_s": self._clock() - t0 - t_scan})
        if not apples:
            return ScenePlan(snap, light, dets, apples, grades, boxes, None, "no_apples", None,
                             {"t_scan_s": t_scan, "t_plan_s": self._clock() - t0 - t_scan})

        grasps = tuple(self._grasps(apples))
        work = self.pipeline.plan_scene(list(apples), list(grades), list(boxes), list(grasps))
        self._prev_apples = list(apples)
        return ScenePlan(snap, light, dets, apples, grades, boxes, work, None, None,
                         {"t_scan_s": t_scan, "t_plan_s": self._clock() - t0 - t_scan})

    def execute(self, plan: ScenePlan) -> SceneWorkPlan | None:
        self.last_verified_snapshot = None
        if plan.work is None:
            return None
        if plan.work.scene_id != plan.snapshot.scene_id:
            raise ValueError("work plan must match the captured scene_id")
        if self._execution_halted:
            return replace(plan.work, items=tuple(
                replace(item, execution=ExecResult(False, FailureCode.ABORTED, Phase.IDLE))
                if item.task.action in (TaskAction.PICK_PLACE, TaskAction.PICK_INSPECT_PLACE) else item
                for item in plan.work.items
            ))
        t0 = self._clock()
        try:
            result = self._execute_verified(plan) if self._capture_for_verification is not None else self.pipeline.execute_scene(
                plan.work, backend=self._backend, inspect_params=self._params, boxes=plan.boxes,
                rescan_for_retry=(lambda item: self._retry_item(plan, item))
                    if self._capture_for_retry is not None else None,
            )
        except Exception:
            self._execution_halted = True
            raise
        self._execution_halted = any(
            item.execution is not None and not item.execution.success for item in result.items
        )
        plan.timings["t_exec_s"] = self._clock() - t0
        return result

    def _execute_verified(self, initial: ScenePlan) -> SceneWorkPlan:
        """Execute one tracked apple, rescan/verify, then replan remaining IDs.

        A new scene may move or lose an apple. Never execute its old coordinates,
        nor replace a missing ID with an unrelated detection.
        """
        current = initial
        remaining = {i.apple.apple_id for i in initial.work.items}
        completed = {}
        while remaining and current.work is not None:
            item = next((i for i in current.work.items if i.apple.apple_id in remaining
                         and i.task.action in (TaskAction.PICK_PLACE, TaskAction.PICK_INSPECT_PLACE)
                         and (i.placement is None or i.placement.feasible)), None)
            if item is None:
                break
            started = self._clock()
            if self._before_execute is not None:
                self._before_execute(item, current.apples, current.boxes)
            latest = [current.snapshot, current.boxes]
            def retry(original):
                fresh, boxes = self._retry_item(current, original)
                latest[:] = [self._latest_retry_snapshot, boxes]
                return fresh, boxes
            one = self.pipeline.execute_scene(SceneWorkPlan(current.work.scene_id, (item,)),
                backend=self._backend, inspect_params=self._params, boxes=current.boxes,
                rescan_for_retry=retry
                    if self._capture_for_retry is not None else None).items[0]
            if one.execution is None or not one.execution.success:
                completed[item.apple.apple_id] = replace(one, timings={**one.timings, 't_total_s': self._clock()-started})
                break
            try:
                deadline = self._clock() + self._verify_timeout
                post = self._capture_for_verification()
                if (not isinstance(post, SceneSnapshot) or post.source != Source.TOP
                        or post.capture_mode != CaptureMode.FULL
                        or post.scene_id <= latest[0].scene_id
                        or not math.isfinite(post.stamp_s) or post.stamp_s <= latest[0].stamp_s):
                    raise ValueError("placement verification requires a fresh TOP exposure")
                light = self._light_for(post) if self._light_for is not None else None
                if light is not None and is_lighting_fault(light):
                    raise ValueError('lighting fault during placement verification')
                post_boxes = tuple(self._boxes(post))
                one = self.pipeline.verify_item(one, latest[1], post_boxes)
                if self._clock() > deadline:
                    raise TimeoutError("placement verification deadline exceeded")
            except TimeoutError:
                one = replace(one, execution=ExecResult(False, FailureCode.TIMEOUT, Phase.RETREAT))
                post = None
            except (ValueError, RuntimeError) as exc:
                _LOG.error("placement verification failed: %s", exc)
                one = replace(one, execution=ExecResult(False, FailureCode.ABORTED, Phase.RETREAT))
                post = None
            one = replace(one, timings={**one.timings, 't_total_s': self._clock()-started})
            completed[item.apple.apple_id] = one
            remaining.remove(item.apple.apple_id)
            if not one.execution.success:
                break
            self.last_verified_snapshot = post
            if remaining:
                # plan() estimates fresh positions and grades, preserving tracking.
                current = self.plan(post)
                completed[item.apple.apple_id] = replace(one, timings={
                    **one.timings, 't_total_s': self._clock()-started})
        return SceneWorkPlan(initial.work.scene_id, tuple(
            completed.get(i.apple.apple_id, i if i.task.action not in
                          (TaskAction.PICK_PLACE, TaskAction.PICK_INSPECT_PLACE) or
                          (i.placement is not None and not i.placement.feasible) else
                          replace(i, task=replace(i.task, action=TaskAction.DEFER), placement=None))
            for i in initial.work.items))

    def _retry_item(self, plan: ScenePlan, original: AppleWorkPlan):
        """Recapture position/occupancy, retaining only this apple's original top grade.

        The estimator must keep apple IDs across scenes. Never substitute another
        apple when tracking fails. In-hand results are discarded by the pipeline.
        """
        assert self._capture_for_retry is not None
        snap = self._capture_for_retry()
        if (not isinstance(snap, SceneSnapshot) or snap.source != Source.TOP
                or snap.capture_mode != CaptureMode.FULL
                or snap.scene_id <= plan.snapshot.scene_id
                or not math.isfinite(snap.stamp_s) or snap.stamp_s <= plan.snapshot.stamp_s):
            raise ValueError("slip recovery requires a fresh FULL TOP snapshot")
        light = self._light_for(snap) if self._light_for is not None else None
        if light is not None and is_lighting_fault(light):
            raise ValueError("lighting fault during slip rescan; recalibrate before retry")
        dets = tuple(self._detect(snap))
        apples = tuple(self._apples(snap, dets, self._prev_apples))
        matches = [a for a in apples if a.apple_id == original.apple.apple_id]
        if len(matches) != 1:
            raise ValueError("slip rescan must track exactly the original apple")
        apple = matches[0]
        top_grade = next(g for g in plan.grades if g.apple_id == apple.apple_id)
        grade = replace(top_grade, scene_id=snap.scene_id, final=False)
        boxes = tuple(self._boxes(snap))
        grasps = tuple(g for g in self._grasps(apples) if g.apple_id == apple.apple_id)
        work = self.pipeline.plan_scene((apple,), (grade,), boxes, grasps)
        self._prev_apples = list(apples)
        self._latest_retry_snapshot = snap
        if self._before_execute is not None:
            self._before_execute(work.items[0], apples, boxes)
        return work.items[0], boxes

    def records(self, snap: SceneSnapshot, plan: ScenePlan, execution: SceneWorkPlan | None) -> list[CycleRecord]:
        if snap.scene_id != plan.snapshot.scene_id:
            raise ValueError("record snapshot must match the planned scene_id")
        if execution is not None and execution.scene_id != snap.scene_id:
            raise ValueError("execution must match the recorded scene_id")
        if execution is not None:
            ids = [item.task.apple_id for item in execution.items]
            planned_ids = {item.task.apple_id for item in plan.work.items} if plan.work is not None else set()
            if len(set(ids)) != len(ids) or not set(ids) <= planned_ids:
                raise ValueError("execution items must uniquely match planned apple_ids")
        grade_by_id = {g.apple_id: g for g in plan.grades}
        t_scan, t_plan = plan.timings.get("t_scan_s", 0.0), plan.timings.get("t_plan_s", 0.0)
        # Scene capture/planning is shared overhead. Allocate once across rows,
        # rather than repeating the scene duration for each apple.
        if plan.work is not None:
            count = max(1, len(plan.work.items))
            t_scan, t_plan = t_scan / count, t_plan / count

        if plan.work is None:                          # 조명 FAULT 또는 사과 없음 — 장면 1행으로 남긴다
            stage = FailStage.D1 if plan.blocked == "no_apples" else FailStage.NONE
            rows = [self._row(snap, g.apple_id, None, g, None, None, False, FailureCode.ABORTED, stage,
                              t_scan, t_plan, 0.0) for g in plan.grades]
            return rows or [self._row(snap, 0, None, None, None, None, False, FailureCode.ABORTED, stage,
                                      t_scan, t_plan, 0.0)]

        executed = {it.task.apple_id: it for it in (execution.items if execution is not None else ())}
        rows = []
        for item in plan.work.items:
            done = executed.get(item.task.apple_id, item)
            ex = done.execution
            success = ex is not None and ex.success
            code = ex.failure_code if ex is not None else FailureCode.NONE
            infeasible = done.placement is not None and not done.placement.feasible
            stage = FailStage.P1 if infeasible and not success else _fail_stage(done, ex)
            if infeasible and execution is not None and ex is None:
                code = FailureCode.ABORTED
            rows.append(self._row(snap, item.task.apple_id, done.apple,
                                  done.final_grade or grade_by_id.get(item.task.apple_id),
                                  done, ex, success, code, stage, t_scan, t_plan, 0.0))
        return rows

    # ------------------------------------------------------------ 내부

    def _row(self, snap, apple_id, apple, grade, item, ex, success, code, stage, t_scan, t_plan, t_exec) -> CycleRecord:
        nan = math.nan
        placement = item.placement if item is not None else None
        return CycleRecord(
            cycle_index=0, scene_id=snap.scene_id, apple_id=apple_id,
            variety=grade.variety if grade is not None else self._params.variety,
            inspection_policy=self._policy,
            task_action=item.task.action if item is not None else TaskAction.DEFER,
            backend=self._backend,
            diameter_m=apple.diameter_m if apple is not None else nan,
            ratio=grade.ratio if grade is not None else nan,
            ratio_lo=grade.ratio_lo if grade is not None else nan,
            ratio_hi=grade.ratio_hi if grade is not None else nan,
            coverage=grade.coverage if grade is not None else nan,
            grade=grade.grade if grade is not None else Grade.REJECT,
            reject_reason=grade.reject_reason if grade is not None else RejectReason.OCCLUDED,
            target_box=placement.box_id if placement is not None else Grade.REJECT,
            target_cell=placement.cell_index if placement is not None else -1,
            grasp_force_set_n=(item.profile.grasp_force_n if item is not None
                               and self.cfg["execution"]["gripper"]["control_mode"] == "calibrated_force" else nan),
            planned_drop_m=placement.planned_drop_m if placement is not None else nan,
            settle_error_m=(item.placement_verification.settle_error_m
                            if item is not None and item.placement_verification is not None else nan),
            t_scan_s=t_scan, t_plan_s=t_plan,
            t_pick_s=item.timings.get('t_pick_s', nan) if item else nan,
            t_inspect_s=item.timings.get('t_inspect_s', nan) if item else nan,
            t_place_s=item.timings.get('t_place_s', nan) if item else nan,
            t_total_s=t_scan + t_plan + (item.timings.get('t_total_s', 0.0) if item else 0.0),
            success=bool(success), failure_code=code, fail_stage=stage,
        )


def _fail_stage(item, ex) -> FailStage:
    if ex is None or ex.success:
        return FailStage.NONE
    if item is not None and item.placement is not None and not item.placement.feasible:
        return FailStage.P1
    if ex.failure_code == FailureCode.DROP:
        return FailStage.F3
    if ex.failure_code in (FailureCode.NO_CONTACT, FailureCode.SLIP, FailureCode.NOT_HOLDING):
        return FailStage.F2
    if ex.phase_reached == Phase.IDLE and ex.failure_code in (FailureCode.ABORTED, FailureCode.GATE_VIOLATION):
        return FailStage.NONE  # execution was held before any physical stage
    if ex.phase_reached in (Phase.IDLE, Phase.APPROACH, Phase.DESCEND, Phase.MOVE_NAMED):
        return FailStage.F1
    if ex.phase_reached in (Phase.GRASP, Phase.LIFT, Phase.INSPECT):
        return FailStage.F2
    return FailStage.F4
