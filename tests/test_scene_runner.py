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


# ---------------------------------------------------------------- tools/run_cycle.py

def save_scene(folder, n):
    folder.mkdir(parents=True, exist_ok=True)
    scene = Scene()
    for i in range(n):
        s = scene()
        np.savez_compressed(folder / f"snap_{i:04d}.npz", color=s.color, depth_m=s.depth_m,
                            intrinsics=np.array(s.intrinsics), stamp_s=s.stamp_s)


def test_run_cycle_replay_dry_run(config, tmp_path, monkeypatch, capsys):
    from tools import run_cycle
    monkeypatch.setattr(run_cycle, "load_config", lambda: config)
    save_scene(tmp_path / "desk", 2)
    adapter = RecordingMotionAdapter()
    records = run_cycle.main(["--replay", str(tmp_path / "desk")],
                             make_runner=lambda c, p: runner_for(c, adapter))
    out = capsys.readouterr().out
    assert len(records) == 6 and adapter.calls == []
    assert "드라이런 - 2 사이클" in out and "PICK_PLACE" in out


def test_run_cycle_refuses_execution_until_parts_exist(config, monkeypatch, capsys):
    from tools import run_cycle
    monkeypatch.setattr(run_cycle, "load_config", lambda: config)
    with pytest.raises(SystemExit) as e:
        run_cycle.main(["--live", "--execute"], make_live_camera=lambda c: Scene())
    out = capsys.readouterr().out
    assert e.value.code == 2
    assert "T_base_camera" in out and "tcp_offset_m" in out and "SegmentValidator" in out


def test_run_cycle_live_plans_without_moving(config, monkeypatch, capsys):
    from tools import run_cycle
    monkeypatch.setattr(run_cycle, "load_config", lambda: config)
    adapter = RecordingMotionAdapter()
    records = run_cycle.main(["--live"], make_runner=lambda c, p: runner_for(c, adapter),
                             make_live_camera=lambda c: Scene())
    assert len(records) == 3 and adapter.calls == []


def test_run_cycle_saves_live_scenes_and_replays_with_override(config, tmp_path, monkeypatch):
    from tools import run_cycle
    monkeypatch.setattr(run_cycle, "load_config", lambda: config)
    adapter = RecordingMotionAdapter()
    live = run_cycle.main(["--live", "--cycles", "2", "--save", str(tmp_path / "field")],
                          make_runner=lambda c, p: runner_for(c, adapter), make_live_camera=lambda c: Scene())
    assert len(list((tmp_path / "field").glob("*.npz"))) == 2                 # 현장 장면이 남았다
    seen = {}

    def runner(c, p):
        seen["boundary"] = c["inspection"]["box_boundary"]
        return runner_for(c, adapter)

    again = run_cycle.main(["--replay", str(tmp_path / "field"), "--set", "inspection.box_boundary=특"],
                           make_runner=runner)
    assert seen["boundary"] == "특" and len(again) == len(live)               # 같은 장면, 값만 바꿔 재실행
    with pytest.raises(SystemExit, match="오타"):
        run_cycle.main(["--replay", str(tmp_path / "field"), "--set", "inspection.no_such_key=1"],
                       make_runner=runner)


def test_real_m1_modules_through_from_config(config):
    """#13 의 실제 M1 (검출·3D 위치·상자·파지 후보) 로 같은 합성 장면을 돌린다."""
    config["workcell"]["T_base_camera"] = T_BASE_CAM.tolist()
    adapter = RecordingMotionAdapter()
    runner = SceneRunner.from_config(config, PickPlacePipeline(config, adapter))
    plan = runner.plan(Scene()())
    assert len(plan.apples) == 3
    for a, truth in zip(sorted(plan.apples, key=lambda a: a.center_m[0]), APPLES):
        assert np.allclose(a.center_m, to_base(truth.center), atol=0.003)       # 3D 위치
        assert a.diameter_m == pytest.approx(2 * RADIUS, abs=0.003)
    ratios = [g.ratio for g in sorted(plan.grades, key=lambda g: g.apple_id)]
    assert ratios == sorted(ratios)                                            # 왼쪽부터 착색 적음 → 많음
    # 상자 위치(origin_m)가 설정 전이면 상자를 재지 않는다 → 놓을 칸 없음 → 실행하지 않고 P1 로 남긴다
    records = runner.records(plan.snapshot, plan, runner.execute(plan))
    assert adapter.calls == [] and all(r.fail_stage == FailStage.P1 and not r.success for r in records)
