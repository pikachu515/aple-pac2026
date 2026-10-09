"""Single-apple pick → box → return timing trial (simulation only)."""
import copy
from pathlib import Path

import numpy as np
import pytest

from aple.config import load_config
from aple.hal.gripper_physics import assumed_config, load_assumptions, physics_scene
from aple.hal.pick_place_trial import (PARK_X_M, apply_overrides, load_trial, load_trial_dict,
                                       split_overrides, trial_config)
from aple.hal.simulation import simulation_config

TRIAL = Path('configs/simulation/pick_place_trial.yaml')


def config():
    a = load_assumptions('configs/simulation/gripper_physics.yaml')
    return assumed_config(simulation_config(load_config(), 'configs/simulation/piper.yaml'), a), a


def build(trial=None, seed=1):
    cfg, _ = config()
    trial = trial or load_trial(TRIAL)
    return trial_config(cfg, trial, trial['base_scene'], seed=seed)


def box_center(box):
    return np.array(box['origin_m'][:2]) + np.array(box['grid'][::-1])*box['cell_pitch_m']/2


def test_assumed_apple_and_reproducible_random_box():
    trial = load_trial(TRIAL)
    out = build(trial)
    assert out == build(trial)
    assert out['workcell']['boxes'] != build(trial, seed=2)['workcell']['boxes']
    (apple,) = out['simulation']['apples']
    assert apple['center_m'][:2] == trial['apple']['center_m']
    assert apple['center_m'][2] == pytest.approx(trial['apple']['diameter_m']/2)
    for seed in range(10):
        out = build(trial, seed)
        target = box_center(out['workcell']['boxes'][trial['target_box']])
        assert trial['box']['reach_radius_m'][0] <= np.linalg.norm(target) <= trial['box']['reach_radius_m'][1]
        assert np.linalg.norm(target - apple['center_m'][:2]) >= trial['box']['min_distance_from_apple_m']
        others = [b for n, b in out['workcell']['boxes'].items() if n != trial['target_box']]
        assert all(box_center(b)[0] == pytest.approx(PARK_X_M) for b in others)


def test_scene_has_no_inspection_marker_and_keeps_real_config():
    cfg, a = config()
    before = copy.deepcopy(cfg)
    trial = load_trial(TRIAL)
    out = trial_config(cfg, trial, trial['base_scene'], seed=3)
    assert cfg == before
    ids = [o['id'] for o in physics_scene(out, a)['objects']]
    assert 'aple_inspection_center' not in ids and ids.count('aple_apple_1') == 1
    scene = out['simulation']['in_hand_scene']
    assert scene['transfer_height_m'] == trial['transfer_height_m']
    assert scene['tilt_rad'] == trial['transfer_tilt_rad']
    assert scene['align_before_box_transfer'] is False


def test_random_other_boxes_do_not_overlap():
    trial = load_trial(TRIAL)
    trial['other_boxes'] = 'random'
    for seed in range(5):
        out = build(trial, seed)
        centers = [box_center(b) for b in out['workcell']['boxes'].values()]
        assert all(c[0] > 0 for c in centers)
        assert len({tuple(np.round(c, 4)) for c in centers}) == 3


def test_fixed_box_and_overlap_rejected():
    trial = load_trial(TRIAL)
    trial['box']['center_m'] = [0.30, 0.30]
    assert np.allclose(box_center(build(trial)['workcell']['boxes'][trial['target_box']]), [0.30, 0.30])
    trial['box']['center_m'] = list(trial['apple']['center_m'])
    with pytest.raises(ValueError, match='overlaps'):
        build(trial)


def test_overrides_split_and_reject_unknown_keys():
    trial_sets, config_sets = split_overrides(['trial.other_boxes=random', 'execution.soft_landing.slow_speed_scale=0.1'])
    trial = load_trial(TRIAL)
    apply_overrides(trial, None, trial_sets)
    assert load_trial_dict(trial)['other_boxes'] == 'random'
    out, a = build(), config()[1]
    apply_overrides(out, a, config_sets + ['assumed_gripper.gripper_settle_s=0.2'])
    assert out['execution']['soft_landing']['slow_speed_scale'] == 0.1
    assert a['gripper_settle_s'] == 0.2
    with pytest.raises(KeyError):
        apply_overrides(out, a, ['execution.soft_landing.typo=1'])
    with pytest.raises(ValueError, match='transfer_height_m'):
        apply_overrides(out, a, ['simulation.in_hand_scene.transfer_height_m=0'])


def test_invalid_trial_values():
    trial = load_trial(TRIAL)
    bad = copy.deepcopy(trial)
    bad['other_boxes'] = 'hide'
    with pytest.raises(ValueError, match='other_boxes'):
        load_trial_dict(bad)
    bad = copy.deepcopy(trial)
    bad['apple']['center_m'] = [0.3, 0.0, 0.04]
    with pytest.raises(ValueError, match='center_m'):
        load_trial_dict(bad)
    bad = copy.deepcopy(trial)
    bad['target_box'] = 'TOP'
    with pytest.raises(ValueError, match='target_box'):
        build(bad)


def test_one_pick_place_return_cycle_in_contact_physics():
    pytest.importorskip('mujoco')
    pytest.importorskip('piper_robot.kinematics')
    pytest.importorskip('piper_sim')
    from aple.hal.pick_place_trial import run_trial
    from aple.hal.simulation_optimization import optimization_config
    trial = load_trial(TRIAL)
    cfg = optimization_config(build(trial, seed=1), trial['speed_profile'])
    a = load_assumptions('configs/simulation/gripper_physics.yaml')
    a.update(cfg['simulation']['in_hand_scene']['gripper_overrides'])
    report, transport = run_trial(cfg, a)
    assert report['success'], report['errors']
    assert report['outcome']['placed_in_destination_bin'] and not report['slips']
    assert {'pick', 'place', 'return_home'} <= set(report['stages'])
    assert 0 < report['cycle_time_s'] < report['simulated_time_s']
    home = np.array(cfg['workcell']['poses']['home_joint_rad'])
    assert np.max(abs(np.array(transport.joint_state()) - home)) <= a['joint_arrival_tolerance_rad']


def test_trial_set_list_and_route_choice_validation():
    trial = load_trial(TRIAL)
    assert all('=' in item for item in trial['set'])
    bad = copy.deepcopy(trial)
    bad['set'] = 'execution.soft_landing.slow_speed_scale=0.05'
    with pytest.raises(ValueError, match='set must be'):
        load_trial_dict(bad)
    from aple.hal.simulation_optimization import optimization_config
    out = optimization_config(build(trial), trial['speed_profile'])
    assert out['simulation']['in_hand_scene']['optimization']['transfer_route_choice'] == 'shortest'
    apply_overrides(out, None, ['simulation.in_hand_scene.optimization.transfer_route_choice=first_valid'])
    assert out['simulation']['in_hand_scene']['optimization']['transfer_route_choice'] == 'first_valid'
