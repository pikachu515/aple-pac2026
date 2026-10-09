"""Mission supervision tests with explicit perception and hardware-proof doubles."""
from dataclasses import dataclass, replace, field
from types import SimpleNamespace
import json
import pytest
from aple.config import load_config
from aple.orchestration.cycle_manager import CycleManager
from aple.orchestration.mission_controller import MissionController
from aple.orchestration.recovery import RecoveryPolicy
from aple.orchestration.scene_runner import ScenePlan
from aple.execution.pick_place_pipeline import SceneWorkPlan
from aple.schema import ExecResult, FailureCode, Phase, TaskAction, TaskItem, InspectionPolicy
from tests.test_cycle_manager import snapshot, record
from tests.test_pick_place_pipeline import apple


@dataclass(frozen=True)
class Item:
    apple: object
    task: object
    placement: object = None
    execution: object = None
    timings: dict = field(default_factory=dict)


class Runner:
    def __init__(self, cfg, scenes, outcomes=()):
        self.scenes = iter(scenes)
        self.outcomes = iter(outcomes)
        self._recovery = RecoveryPolicy(cfg)
        self._params = SimpleNamespace(variety='fuji')
        self._policy, self._backend = InspectionPolicy.NEVER, 'scripted'
        self._execution_halted = False
        self.last_verified_snapshot = None
        self.calls = []
        self.releases = []
        self.pipeline = SimpleNamespace(release_placement_reservation=lambda *a: self.releases.append(a))

    def plan(self, snap):
        ids, action = next(self.scenes)
        apples = tuple(apple(i, scene_id=snap.scene_id) for i in ids)
        items = tuple(Item(a, TaskItem(snap.scene_id, a.apple_id, action)) for a in apples)
        return ScenePlan(snap, None, (), apples, (), (), SceneWorkPlan(snap.scene_id, items))

    def execute(self, plan):
        self.calls.append(plan.work.items[0].apple.apple_id)
        out = next(self.outcomes, ExecResult(True, FailureCode.NONE, Phase.RETREAT))
        if isinstance(out, Exception):
            raise out
        return replace(plan.work, items=(replace(plan.work.items[0], execution=out),))

    def records(self, snap, plan, result):
        return [replace(record(snap.scene_id, i.apple.apple_id), task_action=i.task.action,
            success=i.execution.success if i.execution else False,
            failure_code=i.execution.failure_code if i.execution else FailureCode.NONE)
            for i in result.items]


@pytest.fixture
def setup(tmp_path):
    cfg = load_config('configs')
    cfg['workcell']['expected_apple_count'] = 2
    cfg['orchestration']['logging']['csv_dir'] = str(tmp_path)
    counter = iter(range(2, 100))
    return cfg, lambda: snapshot(next(counter))


def mission(setup, scenes, outcomes=(), **kwargs):
    cfg, capture = setup
    runner = Runner(cfg, scenes, outcomes)
    return MissionController(cfg, runner, capture, **kwargs), runner


def run(m):
    return m.execute(m.plan(snapshot(1)))


def test_incomplete_rescan_then_place_all(setup):
    m, r = mission(setup, [([1], TaskAction.PICK_PLACE), ([1, 2], TaskAction.PICK_PLACE), ([2], TaskAction.PICK_PLACE)])
    result = run(m)
    assert not result.halted and r.calls == [1, 2]
    assert {row.scene_id for row in result.records} == {1}
    assert any(e['kind'] == 'SCENE_NOT_READY' for e in m.events)


def test_deferred_reassessed_and_bounded(setup):
    cfg, _ = setup
    cfg['orchestration']['max_rescan'] = 1
    m, r = mission(setup, [([1, 2], TaskAction.DEFER), ([1, 2], TaskAction.DEFER)])
    result = run(m)
    assert result.halted and not r.calls
    assert any(not row.success for row in result.records)


def test_deferred_becomes_executable(setup):
    m, r = mission(setup, [([1, 2], TaskAction.DEFER), ([1, 2], TaskAction.PICK_PLACE), ([2], TaskAction.PICK_PLACE)])
    assert not run(m).halted
    assert r.calls == [1, 2]


def test_id_replacement_blocks_motion(setup):
    cfg, _ = setup
    cfg['orchestration']['max_rescan'] = 1
    m, r = mission(setup, [([1, 2], TaskAction.PICK_PLACE), ([3], TaskAction.PICK_PLACE), ([3], TaskAction.PICK_PLACE)])
    assert run(m).halted and r.calls == [1]


def test_failure_retry_requires_proof(setup):
    bad = ExecResult(False, FailureCode.IK_FAIL, Phase.APPROACH)
    m, r = mission(setup, [([1, 2], TaskAction.PICK_PLACE)] * 2 + [([2], TaskAction.PICK_PLACE)], [bad], prepare_recovery=lambda item: True)
    result = run(m)
    assert not result.halted and r.calls == [1, 1, 2]
    assert [row.success for row in result.records] == [False, True, True]


def test_unsafe_recovery_and_drop_are_observed_without_scan(setup):
    observations = []
    m, r = mission(setup, [([1, 2], TaskAction.PICK_PLACE)],
        [ExecResult(False, FailureCode.DROP, Phase.TRANSIT)],
        prepare_recovery=lambda item: False,
        observe_drop=lambda: observations.append(True) or snapshot(2))
    assert run(m).halted and r.calls == [1] and observations == [True]
    assert any(e['kind'] == 'DROP_OBSERVED' for e in m.events)


def test_persistent_failure_has_retry_limit(setup):
    cfg, _ = setup
    cfg['orchestration']['max_retry'] = 1
    bad = ExecResult(False, FailureCode.IK_FAIL, Phase.APPROACH)
    m, r = mission(setup, [([1, 2], TaskAction.PICK_PLACE)] * 2, [bad, bad], prepare_recovery=lambda item: True)
    assert run(m).halted and r.calls == [1, 1]


def test_execution_exception_is_recorded_and_stops_next_cycle(setup, tmp_path):
    m, r = mission(setup, [([1, 2], TaskAction.PICK_PLACE)], [RuntimeError('connection lost')])
    manager = CycleManager(setup[0], capture=lambda: snapshot(1), plan_scene=m.plan,
        execute_scene=m.execute, make_records=m.records, failure_records=m.failure_records,
        stop_requested=lambda: m.halted)
    rows = manager.run(3)
    assert rows and not rows[0].success and r.calls == [1]
    assert (tmp_path/'cycles.csv').exists()
    events = [json.loads(line) for line in (tmp_path/'mission_events.jsonl').read_text().splitlines()]
    assert events[-1]['kind'] == 'ALERT'
    assert events[-1]['mission_scene_id'] == 1


def test_capture_exception_produces_sentinel_row(setup):
    m, _ = mission(setup, [])
    def broken():
        raise TimeoutError('camera')
    manager = CycleManager(setup[0], capture=broken, plan_scene=m.plan, execute_scene=m.execute,
        make_records=m.records, failure_records=m.failure_records, stop_requested=lambda: m.halted)
    rows = manager.run(2)
    assert len(rows) == 1 and rows[0].scene_id == 0 and rows[0].failure_code == FailureCode.TIMEOUT


def test_skipped_apples_finish_without_grasp(setup):
    m, r = mission(setup, [([1, 2], TaskAction.SKIP)])
    assert not run(m).halted and not r.calls and m.skipped == {1, 2}


def test_new_explicit_cycle_clears_previous_mission_ids(setup):
    m, r = mission(setup, [([1, 2], TaskAction.PICK_PLACE), ([2], TaskAction.PICK_PLACE),
        ([3, 4], TaskAction.PICK_PLACE), ([4], TaskAction.PICK_PLACE)])
    assert not run(m).halted
    assert not m.execute(m.plan(snapshot(2))).halted
    assert r.calls == [1, 2, 3, 4]


def test_event_sink_error_does_not_erase_mission_records(setup):
    def broken(event):
        raise RuntimeError('display disconnected')
    m, r = mission(setup, [([1, 2], TaskAction.SKIP)], event_sink=broken)
    assert not run(m).halted and m.events


def test_stale_drop_frame_is_rejected(setup):
    m, r = mission(setup, [([1, 2], TaskAction.PICK_PLACE)],
        [ExecResult(False, FailureCode.DROP, Phase.TRANSIT)],
        prepare_recovery=lambda item: True, observe_drop=lambda: snapshot(1))
    assert run(m).halted and r.calls == [1]
    assert not any(e['kind'] == 'DROP_OBSERVED' for e in m.events)


def test_occluded_scene_never_executes(setup):
    cfg, _ = setup
    cfg['orchestration']['max_rescan'] = 0
    m, r = mission(setup, [([1, 2], TaskAction.PICK_PLACE)])
    plan = m.plan(replace(snapshot(1), tray_clear=False))
    assert m.execute(plan).halted and not r.calls


def test_lighting_fault_does_not_move_without_recalibration(setup):
    m, r = mission(setup, [([1, 2], TaskAction.PICK_PLACE)])
    plan = m.plan(snapshot(1))
    plan = replace(plan, scene=replace(plan.scene, blocked='lighting'))
    assert m.execute(plan).halted and not r.calls
    assert not any(e['kind'] == 'RESCAN' for e in m.events)


def test_pregrasp_reservation_released_only_after_proof(setup):
    cfg, _ = setup
    cfg['workcell']['expected_apple_count'] = 1
    m, r = mission(setup, [([1], TaskAction.PICK_PLACE)] * 2,
        [ExecResult(False, FailureCode.IK_FAIL, Phase.APPROACH)], prepare_recovery=lambda i: True)
    original = r.plan
    def with_placement(snap):
        plan = original(snap)
        item = replace(plan.work.items[0], placement=SimpleNamespace(feasible=True, box_id=1, cell_index=2))
        return replace(plan, work=replace(plan.work, items=(item,)))
    r.plan = with_placement
    assert not run(m).halted and r.releases == [(1, 2)]


def test_plan_exception_logged_before_exit(setup):
    m, _ = mission(setup, [])
    manager = CycleManager(setup[0], capture=lambda: snapshot(1), plan_scene=m.plan,
        execute_scene=m.execute, make_records=m.records, failure_records=m.failure_records,
        stop_requested=lambda: m.halted)
    rows = manager.run(2)
    assert len(rows) == 1 and not rows[0].success and m.halted


def test_operator_stop_prevents_capture(setup):
    m, _ = mission(setup, [])
    m.stop()
    manager = CycleManager(setup[0], capture=lambda: pytest.fail('captured after stop'),
        plan_scene=m.plan, execute_scene=m.execute, make_records=m.records,
        failure_records=m.failure_records, stop_requested=lambda: m.halted)
    assert manager.run() == []


def test_real_scene_runner_verifies_each_apple_and_reuses_post_frames(tmp_path):
    from tests.test_pick_place_pipeline import cfg as cfg_fixture
    from tests.test_review_fixes import verified_runner
    # Use the fixture's calibrated test inputs, never production physical values.
    config = cfg_fixture.__wrapped__()
    config['workcell']['expected_apple_count'] = 3
    config['orchestration']['logging']['csv_dir'] = str(tmp_path)
    runner, initial, adapter = verified_runner(config)
    def unnecessary_scan():
        pytest.fail('successful verification frames should be reused')
    m = MissionController(config, runner, unnecessary_scan)
    result = m.execute(m.plan(initial.snapshot))
    assert not result.halted and m.placed == {1, 2, 3}
    assert len(result.records) == 3 and all(r.success for r in result.records)
    assert all(r.scene_id == initial.snapshot.scene_id for r in result.records)
    assert sum(r.t_scan_s for r in result.records) == pytest.approx(sum(p.timings['t_scan_s'] for p in m._plans))
    assert sum(r.t_plan_s for r in result.records) == pytest.approx(sum(p.timings['t_plan_s'] for p in m._plans))


def test_retry_exhaustion_can_skip_with_proven_empty_state(setup):
    cfg, _ = setup
    cfg['orchestration']['max_retry'] = 0
    m, r = mission(setup, [([1, 2], TaskAction.PICK_PLACE)] * 2,
        [ExecResult(False, FailureCode.IK_FAIL, Phase.APPROACH)], prepare_recovery=lambda i: True)
    assert not run(m).halted and r.calls == [1, 2]
    assert m.skipped == {1} and m.placed == {2}
    assert any(e['kind'] == 'SKIP_ALERT' for e in m.events)
