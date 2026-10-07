"""Physical transport and independent failure criteria for assumed adapters."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from aple.config import load_config
from aple.hal.simulation import simulation_config
from aple.hal.gripper_physics import load_assumptions, assumed_config, run_physics, evaluate_outcomes
from tools import run_cycle

ASSUMPTIONS = Path('configs/simulation/gripper_physics.yaml')


def config():
    return assumed_config(simulation_config(load_config(), 'configs/simulation/piper.yaml'), load_assumptions(ASSUMPTIONS))


def test_virtual_assumptions_preserve_hardware_configuration():
    original = load_config()
    before = copy.deepcopy(original)
    virtual = assumed_config(original, load_assumptions(ASSUMPTIONS))
    assert original == before and original['workcell']['tcp_offset_m'] is None
    assert virtual['piper_web']['gripper_width_range_m'] == pytest.approx([.044, .112])


@pytest.mark.parametrize('key,value', [('cartesian_step_m', float('nan')), ('finger_force_limit_n', -1), ('lift_height_m', True)])
def test_invalid_assumptions_are_rejected(tmp_path, key, value):
    data = yaml.safe_load(ASSUMPTIONS.read_text())
    data['assumed_gripper'][key] = value
    path = tmp_path/'assumptions.yaml'
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=key):
        load_assumptions(path)


def test_bin_transport_is_distinct_from_precision_and_empty_bin_failure():
    cfg = config()
    a = load_assumptions(ASSUMPTIONS)
    initial = np.array([apple['center_m'] for apple in cfg['simulation']['apples']])
    expected = np.array([[.2, 0, .032], [.2, .12, .032], [.2, -.12, .032]])
    trace = {'apples': [{'center_m': p} for p in expected]}
    displaced = expected.copy()
    displaced[:, 0] += .014
    displaced[:, 2] = .034
    outcomes = evaluate_outcomes(cfg, trace, a, initial, initial[:, 2]+.08, displaced)
    assert all(o['success'] and not o['center_accurate'] for o in outcomes)
    displaced[0, 0] += .1
    outcomes = evaluate_outcomes(cfg, trace, a, initial, initial[:, 2]+.08, displaced)
    assert not outcomes[0]['success']
    outcomes = evaluate_outcomes(cfg, trace, a, initial, initial[:, 2], expected)
    assert not any(o['success'] for o in outcomes)  # destination alone is insufficient


def test_incomplete_trace_is_rejected_before_importing_physics():
    with pytest.raises(ValueError, match='complete planned cycle'):
        run_physics(config(), {'apples': [], 'events': []}, load_assumptions(ASSUMPTIONS))


def test_mujoco_contacts_move_apples_and_stock_fingers_do_not(tmp_path):
    pytest.importorskip('mujoco')
    pytest.importorskip('piper_sim')
    pytest.importorskip('piper_robot.kinematics')
    cfg = config()
    cfg['orchestration']['logging']['csv_dir'] = str(tmp_path)
    overlay = tmp_path/'config.yaml'
    overlay.write_text(yaml.safe_dump(cfg))
    path = tmp_path/'trace.json'
    rows = run_cycle.main(['--simulate', '--simulation-config', str(overlay), '--trace', str(path)])
    assert len(rows) == 3 and all(r.success for r in rows)
    trace = json.loads(path.read_text())
    a = load_assumptions(ASSUMPTIONS)
    result = run_physics(cfg, trace, a)
    assert result['success'] and not result['transport_errors']
    assert all(o['lifted'] and o['placed_in_destination_bin'] for o in result['apples'])
    control = run_physics(cfg, trace, a, with_attachment=False)
    assert not control['success']
    assert not any(o['lifted'] for o in control['apples'])
