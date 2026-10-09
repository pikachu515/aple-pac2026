"""Central apples sorted by colour into the left (red) / right (yellow) box, simulation only."""
import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from aple.config import load_config
from aple.hal.color_sort_trial import COLORS, RealtimeClock, load_sort, sort_config
from aple.hal.gripper_physics import assumed_config, load_assumptions, physics_scene
from aple.hal.simulation import simulation_config

SORT = Path('configs/simulation/color_sort_trial.yaml')


def build(sort=None, seed=1):
    a = load_assumptions('configs/simulation/gripper_physics.yaml')
    cfg = assumed_config(simulation_config(load_config(), 'configs/simulation/piper.yaml'), a)
    sort = sort or load_sort(SORT)
    return sort_config(cfg, sort, sort['base_scene'], seed=seed), a


def test_central_apples_random_colours_and_large_side_boxes():
    sort = load_sort(SORT)
    seen = set()
    for seed in range(8):
        out, a = build(sort, seed)
        apples = out['simulation']['apples']
        assert len(apples) == sort['apples']['count']
        region = np.asarray(sort['apples']['region_xy_m'])
        for i, apple in enumerate(apples):
            assert (region[:, 0] <= apple['center_m'][:2]).all() and (apple['center_m'][:2] <= region[:, 1]).all()
            assert all(np.linalg.norm(np.subtract(apple['center_m'][:2], b['center_m'][:2]))
                       >= sort['apples']['min_distance_m'] for b in apples[:i])
            assert apple['rgba'] == COLORS[apple['color']]
        seen.update(out['simulation']['color_sort']['colors'])
        boxes = out['workcell']['boxes']
        red, yellow = boxes[sort['boxes']['red']['grade']], boxes[sort['boxes']['yellow']['grade']]
        assert red['grid'] == sort['boxes']['red']['grid'] and red['grid'][0]*red['grid'][1] >= 3
        assert red['origin_m'][1] > 0 > yellow['origin_m'][1] + yellow['grid'][0]*yellow['cell_pitch_m']
    assert seen == {'red', 'yellow'}
    assert out == build(sort, 7)[0]
    rgba = {o['id']: o['rgba'] for o in physics_scene(out, a)['objects'] if o['shape'] == 'sphere'}
    assert rgba['aple_apple_1'] == COLORS[out['simulation']['apples'][0]['color']]


def test_invalid_sort_settings(tmp_path):
    import yaml
    sort = load_sort(SORT)
    bad = copy.deepcopy(sort)
    bad['boxes']['yellow']['grade'] = bad['boxes']['red']['grade']
    path = tmp_path/'bad.yaml'
    path.write_text(yaml.safe_dump(dict(color_sort_trial=bad)))
    with pytest.raises(ValueError, match='different boxes'):
        load_sort(path)
    bad = copy.deepcopy(sort)
    bad['apples']['region_xy_m'] = [[0.30, 0.31], [-0.01, 0.01]]
    with pytest.raises(ValueError, match='graspable apples'):
        build(bad)


def test_realtime_clock_adds_planning_time_as_holding():
    steps = []
    transport = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=.002)),
                                step=lambda s: steps.append(s),
                                move_joint_goal=lambda *a, **k: True,
                                set_gripper_width=lambda *a, **k: True)
    now = [100.]
    import aple.hal.color_sort_trial as module
    clock = RealtimeClock()
    original = module.time.perf_counter
    module.time.perf_counter = lambda: now[0]
    try:
        clock.attach(transport)
        clock.start()
        now[0] += .5                      # 0.5 s of planning, no physics
        transport.move_joint_goal((0,)*6)
        assert steps == [pytest.approx(.5)] and clock.waited_s == pytest.approx(.5)
        transport.set_gripper_width(.1, 0)   # no new planning time
        assert len(steps) == 1
    finally:
        module.time.perf_counter = original


def test_three_apples_sorted_in_contact_physics():
    pytest.importorskip('mujoco')
    pytest.importorskip('piper_robot.kinematics')
    pytest.importorskip('piper_sim')
    from aple.hal.color_sort_trial import run_sort
    from aple.hal.pick_place_trial import apply_overrides, split_overrides
    from aple.hal.simulation_optimization import optimization_config
    sort = load_sort(SORT)
    out, a = build(sort, 1)
    from aple.hal.color_sort_trial import use_hardware_motion
    cfg = use_hardware_motion(optimization_config(out, sort['speed_profile']), load_config())
    a.update(cfg['simulation']['in_hand_scene']['gripper_overrides'])
    apply_overrides(cfg, a, split_overrides(sort['set'])[1])
    report, _ = run_sort(cfg, a)
    assert report['success'], report['errors']
    assert report['motion_path'] == 'hardware_adapter_pick_place'
    for item in report['apples']:
        assert item['box'] == sort['boxes'][item['color']]['grade']
    assert len({(i['box'], i['cell_index']) for i in report['apples']}) == 3
    assert all(o['placed_in_destination_bin'] for o in report['outcomes']) and not report['slips']
