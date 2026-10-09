"""Virtual layout integrity and fresh planning for reproducible scenes."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from aple.config import load_config
from aple.hal.simulation import simulation_config
from aple.hal.gripper_physics import assumed_config, load_assumptions, run_physics
from aple.hal.random_scene import randomize_scene, load_random_settings
from tools.run_studio_gripper_physics import main

SETTINGS = Path('configs/simulation/random_scene.yaml')


def cfg():
    return assumed_config(simulation_config(load_config(), 'configs/simulation/piper.yaml'),
                          load_assumptions('configs/simulation/gripper_physics.yaml'))


def test_seed_reproduces_layout_without_mutating_real_config():
    original = cfg()
    before = copy.deepcopy(original)
    settings = load_random_settings(SETTINGS)
    a = randomize_scene(original, settings, seed=42)
    b = randomize_scene(original, settings, seed=42)
    c = randomize_scene(original, settings, seed=7)
    assert a == b and a != c and original == before
    assert a['simulation']['apples'] != original['simulation']['apples']
    assert all(a['workcell']['boxes'][key]['origin_m'] != box['origin_m']
               for key, box in original['workcell']['boxes'].items())


def test_many_seeds_keep_apples_and_bins_separated_and_on_table():
    config = cfg()
    settings = load_random_settings(SETTINGS)
    for seed in range(20):
        sample = randomize_scene(config, settings, seed=seed)
        fruit = sample['simulation']['apples']
        for i, apple in enumerate(fruit):
            assert apple['center_m'][2] == sample['simulation']['table_z_m'] + apple['radius_m']
            for other in fruit[:i]:
                assert np.linalg.norm(np.array(apple['center_m'])[:2] - np.array(other['center_m'])[:2]) >= settings['apple_min_separation_m']
        bins = list(sample['workcell']['boxes'].values())
        centers = [np.array(b['origin_m'][:2]) + np.array(b['grid'][::-1])*b['cell_pitch_m']/2 for b in bins]
        for i, center in enumerate(centers):
            for other in centers[:i]:
                assert (np.abs(center-other) >= .094 + settings['object_clearance_m']).any()
            for apple in fruit:
                distance = np.maximum(np.abs(np.array(apple['center_m'][:2]) - center) - .047, 0)
                assert np.linalg.norm(distance) >= apple['radius_m'] + settings['object_clearance_m']


def test_impossible_layout_fails_instead_of_using_fixed_positions():
    settings = load_random_settings(SETTINGS)
    settings['apple_center_xy_m'] = [[.29, .30], [0, .01]]
    settings['max_layout_attempts'] = 3
    with pytest.raises(ValueError, match='apple layout'):
        randomize_scene(cfg(), settings, seed=42)


@pytest.mark.parametrize('seed', [-1, True])
def test_invalid_seed_is_rejected(seed):
    with pytest.raises(ValueError, match='seed'):
        randomize_scene(cfg(), load_random_settings(SETTINGS), seed=seed)


def test_seed_cannot_silently_run_fixed_scene(tmp_path):
    with pytest.raises(SystemExit):
        main(['--seed', '42', '--plan-only', '--output', str(tmp_path)])


def test_random_plan_only_never_contacts_studio_and_reaches_new_bin_targets(tmp_path, monkeypatch):
    pytest.importorskip('piper_robot.kinematics')
    import tools.run_studio_gripper_physics as runner

    def forbidden(*args, **kwargs):
        raise AssertionError('plan-only must never construct a Studio client')

    monkeypatch.setattr(runner, 'StudioSimulationReplay', forbidden)
    assert main(['--randomize', '--seed', '42', '--plan-only', '--output', str(tmp_path)]) == 0
    output = tmp_path/'seed-42'
    virtual = yaml.safe_load((output/'virtual-config.yaml').read_text())
    trace = json.loads((output/'planned-trace.json').read_text())
    boxes = virtual['workcell']['boxes']
    expected = {tuple(np.array(b['origin_m'][:2])+np.array(b['grid'][::-1])*b['cell_pitch_m']/2) for b in boxes.values()}
    actual = [apple['center_m'][:2] for apple in trace['apples']]
    assert len(actual) == 3
    assert all(min(np.linalg.norm(np.array(p)-target) for target in expected) < .001 for p in actual)
    assert sum(e['action'] == 'release' for e in trace['events']) == 3
    if pytest.importorskip('mujoco') and pytest.importorskip('piper_sim'):
        report = run_physics(virtual, trace, load_assumptions('configs/simulation/gripper_physics.yaml'))
        assert not report['transport_errors']
        assert all(o['lifted'] for o in report['apples'])
