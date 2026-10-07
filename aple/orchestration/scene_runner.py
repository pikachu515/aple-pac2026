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
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from aple.execution.pick_place_pipeline import PickPlacePipeline, SceneWorkPlan
from aple.inspection.coloration import params_from_config
from aple.inspection.pipeline import inspect_scene
from aple.orchestration.recovery import RecoveryDecision, RecoveryPolicy, is_lighting_fault
from aple.schema import (AppleDetection2D, AppleState, BoxOccupancy, CycleRecord, FailStage, FailureCode,
                         Grade, GradeResult, InspectionPolicy, LightingStatus, RejectReason,
                         SceneSnapshot, TaskAction)

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
                 backend: str = "scripted", clock: Callable[[], float] = time.monotonic) -> None:
        self.cfg = cfg
        self.pipeline = pipeline
        self._detect, self._apples, self._boxes, self._grasps = detect, estimate_apples, estimate_boxes, plan_grasps
        self._light_for = light_for
        self._backend = backend
        self._clock = clock
        self._recovery = RecoveryPolicy(cfg)
        self._params = params_from_config(cfg)
        self._policy = InspectionPolicy[cfg["inspection"].get("policy", "NEVER")]
        mm = cfg["size_bins"]["gripper_range_mm"]
        self._gripper_range_mm = (float(mm[0]), float(mm[1]))
        self._prev_apples: list[AppleState] | None = None

    @classmethod
    def from_config(cls, cfg: dict, pipeline: PickPlacePipeline, **kwargs) -> "SceneRunner":
        """각 역할 모듈의 기본 구현으로 채운다. M1(detector·pose_estimator·box_occupancy·grasp_planner)은 #13."""
        T = cfg["workcell"].get("T_base_camera")
        if T is None:
            raise ValueError("workcell.T_base_camera is not set — run camera–robot calibration first")
        T = np.asarray(T, float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ValueError("workcell.T_base_camera must be a finite 4x4 matrix")
        from aple.perception import box_occupancy, detector, grasp_planner, pose_estimator
        return cls(cfg, pipeline,
                   detect=lambda snap: detector.detect(snap, cfg),
                   estimate_apples=lambda snap, dets, prev: pose_estimator.estimate(snap, dets, T, prev, cfg),
                   estimate_boxes=lambda snap: box_occupancy.estimate(snap, T, cfg),
                   plan_grasps=lambda apples: grasp_planner.plan_grasps(list(apples), cfg),
                   **kwargs)

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
        if plan.work is None:
            return None
        t0 = self._clock()
        result = self.pipeline.execute_scene(plan.work)
        plan.timings["t_exec_s"] = self._clock() - t0
        return result

    def records(self, snap: SceneSnapshot, plan: ScenePlan, execution: SceneWorkPlan | None) -> list[CycleRecord]:
        grade_by_id = {g.apple_id: g for g in plan.grades}
        t_scan, t_plan = plan.timings.get("t_scan_s", 0.0), plan.timings.get("t_plan_s", 0.0)
        t_exec = plan.timings.get("t_exec_s", 0.0)

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
            rows.append(self._row(snap, item.task.apple_id, item.apple, grade_by_id.get(item.task.apple_id),
                                  item, ex, success, code, _fail_stage(item, ex), t_scan, t_plan, t_exec))
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
            grasp_force_set_n=item.profile.grasp_force_n if item is not None else nan,
            planned_drop_m=placement.planned_drop_m if placement is not None else nan,
            settle_error_m=nan,
            t_scan_s=t_scan, t_plan_s=t_plan, t_pick_s=nan, t_inspect_s=nan, t_place_s=nan,
            t_total_s=t_scan + t_plan + t_exec,
            success=bool(success), failure_code=code, fail_stage=stage,
        )


def _fail_stage(item, ex) -> FailStage:
    if item.task.action == TaskAction.PICK_PLACE and (item.placement is None or not item.placement.feasible):
        return FailStage.P1                     # 놓을 칸이 없어 실행하지 않았다 (상자 위치 미설정 포함)
    if ex is None or ex.success:
        return FailStage.NONE
    return {FailureCode.IK_FAIL: FailStage.F1, FailureCode.HOVER_NOT_REACHED: FailStage.F1,
            FailureCode.NO_CONTACT: FailStage.F2, FailureCode.SLIP: FailStage.F2,
            FailureCode.NOT_HOLDING: FailStage.F2, FailureCode.DROP: FailStage.F3}.get(ex.failure_code, FailStage.F4)
