"""Regression checks for the 2026-10-09 review; no hardware access."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import math
import time

import numpy as np
import pytest

from aple.config import load_config
from aple.execution.goal_builder import Pose3D, compose
from aple.execution.inspect_skill import build_inspection_plan
from aple.execution.pick_place_pipeline import PickPlacePipeline
from aple.execution.testing import RecordingMotionAdapter
from aple.execution.timed_motion import MotionLimits, time_parameterize
from aple.hal.in_hand_simulation import inspection_poses
from aple.hal.live_runtime import build_live_runtime, missing_for_live_execution
from aple.hal.live_snapshot import PiperLiveCamera
from aple.hal.simulation import simulation_config
from aple.inspection.in_hand_provider import pose_to_matrix
from aple.planning.grasp_profile import select
from aple.schema import FailureCode, Phase, TaskAction
from tests.test_pick_place_pipeline import cfg, apple, grade, box, grasp_candidate  # noqa: F401
from tests.test_real_world import CALIB, ENC
from tests.test_scene_runner import Scene, runner_for, estimate_boxes, estimate_apples
from tests.test_segment_validator import validator, q


def test_timed_command_velocity_acceleration_and_path_corners():
    limits = MotionLimits((.8,)*6, (1.2,)*6, .01, .05, .1)
    start = (0.,)*6
    corner, end = (.4, 0, 0, 0, 0, 0), (.4, .3, 0, 0, 0, 0)
    fast = time_parameterize(start, [(.2, 0, 0, 0, 0, 0), corner, end], limits, speed_scale=1.)
    slow = time_parameterize(start, [corner, end], limits, speed_scale=.2)
    assert slow[-1].time_s > fast[-1].time_s
    positions = np.array([start]+[g.position_rad for g in fast])
    velocity = np.diff(positions, axis=0)/limits.period_s
    acceleration = np.diff(velocity, axis=0)/limits.period_s
    assert np.abs(velocity).max() <= .8 + 1e-8
    assert np.abs(acceleration).max() <= 1.2 + 1e-8
    assert any(np.allclose(p, corner) for p in positions)
    # Cannot shortcut a checked right-angle path with a diagonal.
    assert all(p[1] == 0 or p[0] == .4 for p in positions)


@pytest.mark.parametrize('bad', [0., -1., math.nan, True])
def test_timed_limits_reject_invalid_measurements(bad):
    with pytest.raises(ValueError):
        MotionLimits((bad,)*6, (1.,)*6, .01, .1, .1)


def test_inspection_moves_view_direction_and_preserves_apple_center():
    offset = Pose3D((.01, -.02, .08), (.0, .0, math.sin(.2), math.cos(.2)))
    settings = dict(inspection_center_m=[.3, 0, .2], tilt_rad=math.pi/2,
                    rotation_angles_rad=[0, math.pi/2, math.pi, 3*math.pi/2], settle_time_s=.1)
    plan = build_inspection_plan(offset, tuple(settings['inspection_center_m']),
        tilt_rad=settings['tilt_rad'], rotation_angles_rad=settings['rotation_angles_rad'], settle_time_s=.1)
    assert inspection_poses(settings, offset) == plan.target_tool_poses
    directions = []
    for pose in plan.target_tool_poses:
        held = compose(pose, offset)
        assert held.position_m == pytest.approx(settings['inspection_center_m'])
        directions.append(pose_to_matrix(held)[:3, :3].T @ np.array([0, 0, 1]))
    assert np.linalg.norm(directions[0]-directions[2]) > 1.9


def test_opening_margin_exceeding_actual_travel_is_infeasible(cfg):
    cfg['piper_web']['gripper_width_range_m'] = [0., .100]
    a = replace(apple(), diameter_m=.098)
    profile = select(a, cfg)
    assert profile.open_width_m == pytest.approx(.103)
    assert not profile.feasible
    cfg['piper_web']['gripper_width_range_m'] = None
    assert not select(a, cfg).feasible


def test_repeated_previews_do_not_consume_real_cells(cfg):
    pipeline = PickPlacePipeline(cfg)
    cells = [pipeline.plan_scene([apple(i)], [grade(i)], [box()], [grasp_candidate(i)]).items[0].placement.cell_index
             for i in range(1, 15)]
    assert len(set(cells)) == 1
    assert not pipeline._placement_planner.reserved
    # Two apples within the same preview still get different cells.
    work = pipeline.plan_scene([apple(), apple(2)], [grade(), grade(2)], [box()], [grasp_candidate(), grasp_candidate(2)])
    assert len({i.placement.cell_index for i in work.items}) == 2


@pytest.mark.parametrize('stamp_kind', ['before', 'future', 'old'])
def test_single_frame_capture_rejects_bad_exposure(stamp_kind):
    wall = 10_000_000_000
    stamp = {'before':wall-1, 'future':wall+1, 'old':wall-2_000_000_000}[stamp_kind]
    reader = lambda: (np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.uint8), stamp)
    cam = PiperLiveCamera(CALIB, ENC, reader, frames=1, timeout_s=.012, wall_clock_ns=lambda:wall)
    with pytest.raises(TimeoutError):
        cam(capture_after_ns=0 if stamp_kind == 'old' else wall)
    assert cam._next_id == 1


def test_capture_never_assigns_new_id_to_same_exposure():
    stamp = [10_000_000_001]
    reader = lambda: (np.zeros((4, 4, 3), np.uint8), np.ones((4, 4), np.uint8), stamp[0])
    cam = PiperLiveCamera(CALIB, ENC, reader, frames=1, timeout_s=.012, wall_clock_ns=lambda:10_000_000_002)
    assert cam(capture_after_ns=10_000_000_000).scene_id == 1
    with pytest.raises(TimeoutError):
        cam(capture_after_ns=10_000_000_000)
    stamp[0] += 1
    assert cam(capture_after_ns=10_000_000_000).scene_id == 2


def test_held_apple_volume_blocks_path_clear_for_robot_alone():
    guard = validator(obstacles=[[[-.04, .04], [-.04, .04], [.58, .62]]])
    assert guard.is_segment_safe(q(-.5), q(.5))
    guard.set_payload(Pose3D((0, 0, .1), (0, 0, 0, 1)), .035)
    assert not guard.is_segment_safe(q(-.5), q(.5))
    guard.clear_payload()
    assert guard.is_segment_safe(q(-.5), q(.5))


def test_observed_neighbor_remains_collision_obstacle():
    guard = validator()
    guard._wall_thickness = .005
    target = replace(apple(), center_m=(-.5, 0, .5))
    other = replace(apple(2), center_m=(0, 0, .5))
    guard.update_scene([target, other], [], target_apple_id=target.apple_id)
    assert not guard.is_segment_safe(q(-.5), q(.5))
    guard.update_scene([target], [], target_apple_id=target.apple_id)
    assert guard.is_segment_safe(q(-.5), q(.5))


def verified_runner(cfg, *, placement_ok=True, stale=False):
    adapter = RecordingMotionAdapter()
    runner = runner_for(cfg, adapter)
    scene = Scene()
    initial = runner.plan(scene())
    state = dict(done=set(), heights={}, active=None)
    def before(item, apples, boxes):
        state['active'] = item
    def boxes(snap):
        return [replace(b, heights_m=state['heights'].get(b.box_id, list(b.heights_m)),
                        occupied=[h > 0 for h in state['heights'].get(b.box_id, b.heights_m)]) for b in estimate_boxes(snap)]
    def capture():
        if stale:
            return initial.snapshot
        item = state['active']
        state['done'].add(item.apple.apple_id)
        if placement_ok:
            heights = state['heights'].setdefault(item.placement.box_id, [0.]*6)
            heights[item.placement.cell_index] += item.apple.diameter_m
        return scene()
    def apples(snap, detections, prev):
        return [replace(a, center_m=(a.center_m[0]+.01, *a.center_m[1:]))
                for a in estimate_apples(snap, detections, prev) if a.apple_id not in state['done']]
    runner._before_execute = before
    runner._capture_for_verification = capture
    runner._boxes, runner._apples = boxes, apples
    return runner, initial, adapter


def test_verified_loop_refreshes_remaining_coordinates_and_frees_reservations(cfg):
    runner, initial, adapter = verified_runner(cfg)
    result = runner.execute(initial)
    assert all(i.execution.success and i.placement_verification.success for i in result.items)
    assert not runner.pipeline._placement_planner.reserved
    assert result.items[1].apple.scene_id > initial.snapshot.scene_id
    old = {a.apple_id:a for a in initial.apples}
    changed = [i for i in result.items if i.apple.scene_id > initial.snapshot.scene_id]
    assert changed and all(i.apple.center_m[0] == pytest.approx(old[i.apple.apple_id].center_m[0]+.01) for i in changed)
    rows = runner.records(initial.snapshot, initial, result)
    assert sum(r.t_scan_s for r in rows) == pytest.approx(initial.timings['t_scan_s'])
    assert all(math.isfinite(r.settle_error_m) for r in rows)
    assert all(math.isnan(r.grasp_force_set_n) for r in rows) # fixed torque is not N


@pytest.mark.parametrize('stale', [False, True])
def test_failed_or_stale_placement_scan_halts_next_apple(cfg, stale):
    runner, initial, adapter = verified_runner(cfg, placement_ok=False, stale=stale)
    result = runner.execute(initial)
    executed = [i for i in result.items if i.execution is not None]
    assert len(executed) == 1 and not executed[0].execution.success
    assert executed[0].execution.failure_code == FailureCode.ABORTED
    assert runner._execution_halted and runner.pipeline._placement_planner.reserved
    count = len(adapter.calls)
    runner.execute(initial)
    assert len(adapter.calls) == count


def measured_fixture(tmp_path):
    # Synthetic inputs only: this function never writes deployable YAML.
    cfg = simulation_config(load_config(), 'configs/simulation/piper.yaml')
    cfg['execution']['motion'] = dict(velocity_limits_rad_s=[.5]*6, acceleration_limits_rad_s2=[1.]*6,
        command_period_s=.05, tracking_tolerance_rad=.05, velocity_tolerance_rad_s=.1,
        capture_joint_tolerance_rad=.005)
    cfg['execution']['gripper'].update(closing_compression_m=.001, contact_settle_s=.1,
        contact_width_tolerance_m=.003, contact_stability_m=.0005)
    cfg['safety']['segment_validation'].update(wall_thickness_m=.005,
        self_collision_pairs=[], payload_self_links=[])
    cfg['piper_web'].update(enabled=True, robotd_safety_confirmed=True, iface='fake_follower')
    card = tmp_path/'fake-card.json'
    card.write_text('{}') # factory doesn't load color card: CardMonitor does
    cfg['color']['card_json'] = str(card)
    return cfg


def test_live_factory_checks_measurements_before_any_studio_construction(monkeypatch):
    import aple.hal.piper_studio as module
    def forbidden(*args, **kwargs):
        pytest.fail('no robot/client construction allowed before preflight succeeds')
    monkeypatch.setattr(module, 'PiperStudio', forbidden)
    missing = missing_for_live_execution(load_config())
    assert any('velocity_limits' in s for s in missing)
    with pytest.raises(ValueError, match='configuration incomplete'):
        build_live_runtime(load_config(), forbidden, forbidden)


def test_live_factory_wires_adapter_capture_verification_and_common_fk(tmp_path, monkeypatch):
    import aple.hal.piper_studio as studio_module
    import aple.hal.piper_motion as motion_module
    import aple.execution.trajectory as trajectory_module
    import aple.execution.segment_validator as guard_module
    cfg = measured_fixture(tmp_path)
    assert missing_for_live_execution(cfg) == []
    adapter = RecordingMotionAdapter()
    fk = lambda q: Pose3D((.3, 0, .3), (0, 0, 0, 1))
    studio = SimpleNamespace(joint_state=lambda:(0.,)*6)
    guard = SimpleNamespace(clear_payload=lambda:None, update_scene=lambda *a, **kw:None)
    captured = {}
    monkeypatch.setattr(studio_module, 'PiperStudio', lambda *a, **kw:studio)
    monkeypatch.setattr('aple.hal.live_runtime.build_piper_kinematics', lambda c:('ik', fk))
    monkeypatch.setattr(guard_module, 'build_piper_segment_validator', lambda c, f:guard)
    def build(s, c, ik, g, **kwargs):
        captured.update(kwargs)
        return adapter
    monkeypatch.setattr(motion_module, 'build_piper_motion_adapter', build)
    runtime = build_live_runtime(cfg, Scene(), lambda snap: __import__('tests.test_scene_runner', fromlist=['OK']).OK)
    assert runtime.adapter is adapter
    assert captured['tool_pose_provider'] is fk
    assert captured['in_hand_capture_provider']._tool_pose_fn is fk
    assert callable(captured['holding_verifier'])
    assert runtime.runner._capture_for_verification is runtime.capture
    assert runtime.runner._capture_for_retry is runtime.capture


def test_retry_grasp_planner_receives_all_observed_neighbors(cfg):
    runner = runner_for(cfg, RecordingMotionAdapter())
    scene = Scene()
    initial = runner.plan(scene())
    seen = []
    old = runner._grasps
    def grasps(apples):
        seen.extend(a.apple_id for a in apples)
        return old(apples)
    runner._grasps = grasps
    runner._capture_for_retry = scene
    runner._retry_item(initial, initial.work.items[0])
    assert set(seen) == {a.apple_id for a in initial.apples}
    assert len(seen) == 3


@pytest.mark.parametrize('provide_verifier', [True, False])
def test_command_ack_without_fruit_evidence_cannot_report_holding(provide_verifier):
    from tests.test_piper_motion import make_adapter, pose
    from aple.execution.goal_builder import MotionTargets
    from aple.schema import GraspProfile, PlacementPlan, Grade
    adapter, transport, _ = make_adapter()
    transport.holding_confirmed = False
    if not provide_verifier:
        transport.confirm_holding = None
    a = apple()
    targets = MotionTargets(pose(.2), pose(0, .3), pose(0), pose(.4), pose(.4, .3), 0.)
    result = adapter.execute_phase_a(a, GraspProfile(1, 1, 0, .2, .09, 0, 0, 1000, True), None,
        PlacementPlan(1, 1, Grade.UPPER, 0, (.3, 0, .2), .3, .2, .01, True), targets,
        backend='scripted', effort=1000, task_text='', speed_scale=.4, timeout_s=3.)
    assert not result.success and not result.holding
    assert result.failure_code == FailureCode.NOT_HOLDING
    assert result.phase_reached == Phase.LIFT
    assert all(c[1][0] < .4 for c in transport.calls if c[0] == 'move')


def test_support_tolerance_only_applies_to_contact_lift():
    guard = validator()
    guard._support_tolerance = .001
    offset = Pose3D((0, 0, -.4605), (0, 0, 0, 1))
    guard.set_payload(offset, .04, allow_support_contact=True)
    assert guard.is_segment_safe(q(0), q(.1)) # 0.5mm apparent penetration from measurement error
    guard.set_payload(offset, .04)
    assert not guard.is_segment_safe(q(0), q(.1))
    guard.set_payload(Pose3D((0, 0, -.47), (0, 0, 0, 1)), .04, allow_support_contact=True)
    assert not guard.is_segment_safe(q(0), q(.1)) # true 10mm penetration is still blocked


def test_configured_tray_wall_blocks_mid_path():
    guard = validator()
    guard._wall_thickness = .005
    guard._tray = dict(center_m=[0, 0, .4], size_m=[.2, .2], wall_height_m=.2)
    guard.update_scene([], [], target_apple_id=1)
    assert not guard.is_segment_safe(q(-.5), q(.5))


def test_color_depth_freshness_uses_earliest_exposure():
    from aple.hal.shm_frames import read_pair
    color, depth = np.zeros((2, 2, 3), np.uint8), np.zeros((2, 2), np.uint8)
    for color_time, depth_time in ((1_010_000_000, 1_000_000_000), (1_000_000_000, 1_010_000_000)):
        c = SimpleNamespace(read=lambda:(color, 1, color_time))
        d = SimpleNamespace(read=lambda:(depth, 1, depth_time))
        assert read_pair(c, d)[2] == 1_000_000_000


def test_initial_scan_capture_time_is_allocated_once(cfg, monkeypatch, tmp_path):
    from tests.test_scene_runner import manager
    cfg['orchestration']['logging']['csv_dir'] = str(tmp_path)
    runner = runner_for(cfg, RecordingMotionAdapter())
    runner._clock = lambda:0.
    samples = iter([10., 12.])
    monkeypatch.setattr('aple.orchestration.cycle_manager.time.monotonic', lambda:next(samples))
    rows = manager(cfg, runner, Scene(), dry_run=True).run(1)
    assert sum(r.t_scan_s for r in rows) == pytest.approx(2.)


@pytest.mark.parametrize('mutate', [
    lambda c:c['safety']['segment_validation'].update(clearance_m=-.1),
    lambda c:c['safety']['segment_validation'].update(obstacle_aabbs_m=[[[0,0]]*3]),
    lambda c:c['safety']['segment_validation'].update(self_collision_pairs=[['link2','link2']]),
    lambda c:c['execution']['motion'].update(velocity_limits_rad_s=[True]*6),
])
def test_invalid_live_measurements_report_missing_without_robot(tmp_path, mutate):
    c = measured_fixture(tmp_path)
    mutate(c)
    assert missing_for_live_execution(c)
