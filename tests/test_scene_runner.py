"""M4 통합 — 합성 장면 1장으로 카메라 → 검출 → 판정 → 계획 → (가짜 로봇) 실행 → 기록까지 CycleManager 로 돌린다.

M1 인식(#13)은 아직 main 에 없으므로 검출은 inspection.standalone(깊이 기반 분할), 3D 위치는
합성 정답으로 대신한다. 로봇은 역할 B 의 RecordingMotionAdapter.
"""

import math

import numpy as np
import pytest

from aple.execution.goal_builder import Pose3D
from aple.execution.pick_place_pipeline import PickPlacePipeline
from aple.execution.testing import RecordingMotionAdapter
from aple.inspection.standalone import segment_apples
from aple.orchestration.cycle_manager import CycleManager
from aple.orchestration.recovery import RecoveryAction
from aple.orchestration.scene_runner import SceneRunner
from aple.schema import (AppleState, BoxOccupancy, CaptureMode, FailStage, FailureCode, Grade, LightingStatus,
                         LightState, SceneSnapshot, Source, TaskAction)
from tests.synthetic import INTR, Apple, half_red, render, side_band
from tests.test_inhand import T_BASE_CAM
from tests.test_pick_place_pipeline import GraspCandidateFixture, cfg  # noqa: F401

RADIUS = 0.039
APPLES = [Apple((-0.12, 0.03, 0.60), RADIUS, side_band(0.6)),      # 착색 적음
          Apple((0.0, 0.03, 0.60), RADIUS, half_red),
          Apple((0.12, 0.03, 0.60), RADIUS, side_band(-0.7))]       # 착색 많음
OK = LightingStatus(LightState.OK, 0.3, True)
FAULT = LightingStatus(LightState.FAULT, 9.0, True)


def to_base(p_cam):
    return T_BASE_CAM[:3, :3] @ np.asarray(p_cam) + T_BASE_CAM[:3, 3]


class Scene:
    """부를 때마다 scene_id 가 1 늘어나는 합성 상부 카메라."""

    def __init__(self, apples=APPLES):
        self.apples, self.n = apples, 0

    def __call__(self):
        self.n += 1
        color, depth, _ = render(self.apples, seed=self.n)
        return SceneSnapshot(self.n, Source.TOP, CaptureMode.FULL, color, depth, INTR, float(self.n))


def detect(snap):
    return segment_apples(snap.color, snap.depth_m, snap.intrinsics, scene_id=snap.scene_id)


def estimate_apples(snap, dets, prev):
    """합성 정답으로 3D 위치를 채운다 (M1 대역). 검출 마스크 중심에 가장 가까운 사과를 고른다."""
    fx, fy, cx, cy = INTR
    out = []
    for i, d in enumerate(dets):
        vs, us = np.nonzero(d.mask)
        a = min(APPLES, key=lambda a: (fx * a.center[0] / a.center[2] + cx - us.mean()) ** 2
                + (fy * a.center[1] / a.center[2] + cy - vs.mean()) ** 2)
        c = to_base(a.center)
        out.append(AppleState(snap.scene_id, i + 1, d.det_id, tuple(c), 2 * a.radius, c[2] + a.radius,
                              0.03, 0.001, 0.9, 0.9))
    return out


def estimate_boxes(snap):
    def box(grade, x):
        return BoxOccupancy(snap.scene_id, grade, 2, 3, 0.09, 0.08, 0.08, (x, 0.25, 0.0),
                            [0.0] * 6, [False] * 6, True)
    return [box(Grade.UPPER, 0.0), box(Grade.MIDDLE, 0.3), box(Grade.REJECT, 0.6)]


def plan_grasps(apples):
    return [GraspCandidateFixture(a.scene_id, a.apple_id, Pose3D(a.center_m, (0.0, 0.0, 0.0, 1.0)), 0.0,
                                  (0.0, 0.0, 1.0), 0.08, 0.02, True, 0.5) for a in apples]


def runner_for(config, adapter, light=OK):
    pipeline = PickPlacePipeline(config, adapter)
    return SceneRunner(config, pipeline, detect=detect, estimate_apples=estimate_apples,
                       estimate_boxes=estimate_boxes, plan_grasps=plan_grasps,
                       light_for=(lambda snap: light) if light is not None else None)


@pytest.fixture
def config(cfg, tmp_path):  # noqa: F811
    cfg["orchestration"]["logging"]["csv_dir"] = str(tmp_path)
    return cfg


def manager(config, runner, scene, *, dry_run):
    return CycleManager(config, capture=scene, plan_scene=runner.plan, execute_scene=runner.execute,
                        make_records=runner.records, dry_run_capture=scene if dry_run else None, dry_run=dry_run)


def test_dry_run_plans_every_apple_without_moving(config, tmp_path):
    adapter = RecordingMotionAdapter()
    runner = runner_for(config, adapter)
    records = manager(config, runner, Scene(), dry_run=True).run(2)
    assert adapter.calls == []                                          # 드라이런은 로봇을 부르지 않는다
    assert [r.cycle_index for r in records] == [1, 1, 1, 2, 2, 2]
    first = sorted(records[:3], key=lambda r: r.apple_id)              # 검출 순서 = 왼쪽부터
    assert [r.apple_id for r in first] == [1, 2, 3]
    assert all(r.ratio_lo <= r.ratio <= r.ratio_hi for r in first)
    assert first[0].ratio < first[1].ratio < first[2].ratio              # 착색 적음 < 반 < 많음
    assert all(r.diameter_m == pytest.approx(2 * RADIUS) for r in first)
    assert (tmp_path / "cycles.csv").exists()                            # 발표 지표 CSV


def test_live_cycle_executes_and_records_targets(config):
    adapter = RecordingMotionAdapter()
    runner = runner_for(config, adapter)
    records = manager(config, runner, Scene(), dry_run=False).run(1)
    assert adapter.calls                                                # 가짜 로봇에 명령이 갔다
    placed = [r for r in records if r.task_action == TaskAction.PICK_PLACE]
    assert placed
    for r in placed:
        assert r.target_box == r.grade and r.target_cell >= 0
        assert r.success and r.failure_code == FailureCode.NONE and r.fail_stage == FailStage.NONE
        assert math.isfinite(r.planned_drop_m) and r.t_total_s >= 0


def test_lighting_fault_holds_execution_and_asks_recalibration(config):
    adapter = RecordingMotionAdapter()
    runner = runner_for(config, adapter, light=FAULT)
    scene = Scene()
    plan = runner.plan(scene())
    assert plan.blocked == "lighting" and plan.work is None
    assert plan.recovery.action == RecoveryAction.RECALIBRATE_LIGHT and plan.recovery.alert
    records = manager(config, runner, Scene(), dry_run=False).run(1)
    assert adapter.calls == []                                          # 리젝트 상자로 보내지 않는다
    assert len(records) == 3 and all(not r.success and r.failure_code == FailureCode.ABORTED for r in records)


def test_empty_tray_records_one_row(config):
    runner = runner_for(config, RecordingMotionAdapter())
    records = manager(config, runner, Scene(apples=[]), dry_run=False).run(1)
    assert len(records) == 1 and records[0].fail_stage == FailStage.D1 and not records[0].success


def test_from_config_requires_calibration(config):
    with pytest.raises(ValueError, match="T_base_camera"):
        SceneRunner.from_config(config, PickPlacePipeline(config))


def test_execute_passes_backend_and_mode_b_options(config, monkeypatch):
    runner = runner_for(config, RecordingMotionAdapter())
    runner._backend = "act"
    plan = runner.plan(Scene()())
    calls = []

    def execute(work, **kwargs):
        calls.append(kwargs)
        return work

    monkeypatch.setattr(runner.pipeline, "execute_scene", execute)
    runner.execute(plan)

    assert calls[0]["backend"] == "act"
    assert calls[0]["inspect_params"] is not None
    assert calls[0]["boxes"] == plan.boxes


def test_records_use_executed_placement_and_grade(config):
    from dataclasses import replace
    from aple.schema import ExecResult, Phase

    runner = runner_for(config, RecordingMotionAdapter())
    snap = Scene()()
    plan = runner.plan(snap)
    item = next(it for it in plan.work.items if it.placement is not None)
    top = next(g for g in plan.grades if g.apple_id == item.apple.apple_id)
    done = replace(
        item, placement=replace(item.placement, box_id=Grade.REJECT, cell_index=4),
        final_grade=replace(top, grade=Grade.REJECT, final=True),
        execution=ExecResult(True, FailureCode.NONE, Phase.RETREAT),
    )
    execution = replace(plan.work, items=(done,))

    row = next(r for r in runner.records(snap, plan, execution) if r.apple_id == item.apple.apple_id)

    assert row.success and row.grade == Grade.REJECT
    assert row.target_box == Grade.REJECT and row.target_cell == 4


def test_placement_ik_failure_is_not_logged_as_approach_failure():
    from aple.orchestration.scene_runner import _fail_stage
    from aple.schema import ExecResult, Phase

    assert _fail_stage(None, ExecResult(False, FailureCode.IK_FAIL, Phase.TRANSIT)) == FailStage.F4
    assert _fail_stage(None, ExecResult(False, FailureCode.IK_FAIL, Phase.APPROACH)) == FailStage.F1


def test_execution_failure_blocks_automatic_motion_in_later_cycles(config):
    from aple.execution.motion_executor import PhaseAResult
    from aple.schema import Phase

    adapter = RecordingMotionAdapter(
        phase_a_result=PhaseAResult(False, FailureCode.IK_FAIL, Phase.APPROACH, False, False)
    )
    runner = runner_for(config, adapter)
    scene = Scene()
    runner.execute(runner.plan(scene()))
    count = len(adapter.calls)

    runner.execute(runner.plan(scene()))

    assert len(adapter.calls) == count


def test_backend_defaults_to_execution_config(config):
    config["execution"]["default_backend"] = "act"
    runner = runner_for(config, RecordingMotionAdapter())
    snap = Scene()()
    rows = runner.records(snap, runner.plan(snap), None)

    assert all(row.backend == "act" for row in rows)


@pytest.mark.parametrize("code,phase,stage", [
    (FailureCode.IK_FAIL, "DESCEND", FailStage.F1),
    (FailureCode.TIMEOUT, "GRASP", FailStage.F2),
    (FailureCode.NO_CONTACT, "DESCEND", FailStage.F2),
    (FailureCode.DROP, "TRANSIT", FailStage.F3),
    (FailureCode.COLLISION, "TRANSIT", FailStage.F4),
    (FailureCode.HOVER_NOT_REACHED, "HOVER", FailStage.F4),
    (FailureCode.ABORTED, "RELEASE", FailStage.F4),
    (FailureCode.ABORTED, "IDLE", FailStage.NONE),
    (FailureCode.GATE_VIOLATION, "IDLE", FailStage.NONE),
])
def test_failure_stage_uses_failure_code_and_physical_phase(code, phase, stage):
    from aple.orchestration.scene_runner import _fail_stage
    from aple.schema import ExecResult, Phase

    assert _fail_stage(None, ExecResult(False, code, Phase[phase])) == stage


def test_records_reject_a_different_scene(config):
    from dataclasses import replace

    runner = runner_for(config, RecordingMotionAdapter())
    snap = Scene()()
    plan = runner.plan(snap)

    with pytest.raises(ValueError, match="scene_id"):
        runner.records(replace(snap, scene_id=99), plan, None)
    with pytest.raises(ValueError, match="scene_id"):
        runner.records(snap, plan, replace(plan.work, scene_id=99))


def test_records_reject_duplicate_apple_executions(config):
    from dataclasses import replace

    runner = runner_for(config, RecordingMotionAdapter())
    snap = Scene()()
    plan = runner.plan(snap)
    item = plan.work.items[0]

    with pytest.raises(ValueError, match="apple_ids"):
        runner.records(snap, plan, replace(plan.work, items=(item, item)))


def test_infeasible_final_placement_records_no_free_cell(config):
    from dataclasses import replace
    from aple.schema import ExecResult, Phase

    runner = runner_for(config, RecordingMotionAdapter())
    snap = Scene()()
    plan = runner.plan(snap)
    item = next(it for it in plan.work.items if it.placement is not None)
    done = replace(item, placement=replace(item.placement, feasible=False, cell_index=-1),
                   execution=ExecResult(False, FailureCode.ABORTED, Phase.IDLE))

    rows = runner.records(snap, plan, replace(plan.work, items=(done,)))
    row = next(r for r in rows if r.apple_id == item.apple.apple_id)

    assert row.fail_stage == FailStage.P1 and not row.success
