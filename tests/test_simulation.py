"""Optional official Piper math checks plus ordinary simulation config tests.

Set PYTHONPATH to Piper Studio's robot directory to exercise the real model.
"""
import copy
import json
from pathlib import Path

import pytest

from aple.config import load_config
from aple.hal.simulation import simulation_config, build_simulation, check_piper_roundtrip
from aple.execution.trajectory import build_piper_kinematics, TrajectoryPlanningError
from tools import run_cycle

OVERLAY = Path(__file__).resolve().parents[1] / 'configs/simulation/piper.yaml'


def test_simulation_overlay_preserves_measured_configuration():
    cfg = load_config()
    before = copy.deepcopy(cfg)
    virtual = simulation_config(cfg, OVERLAY)
    assert cfg == before and cfg['workcell']['tcp_offset_m'] is None
    assert virtual['workcell']['tcp_offset_m'] == [0, 0, .16]
    assert not virtual['piper_web']['enabled']
    assert virtual['execution']['motion']['empty_speed_scale'] is None


def test_simulation_cannot_be_combined_with_hardware_execution():
    with pytest.raises(SystemExit, match='--execute'):
        run_cycle.main(['--simulate', '--execute'])


def test_empty_finger_preopening_is_allowed_but_missed_apple_close_fails():
    pytest.importorskip('piper_robot.kinematics')
    cfg = simulation_config(load_config(), OVERLAY)
    cfg['piper_web']['gripper_width_range_m'] = [.044, .112]
    cfg['execution']['gripper']['opening_margin_m'] = .02
    world, _, _ = build_simulation(cfg)
    assert world.set_gripper_width(.084, 1000, timeout_s=1)
    assert world.held is None
    assert not world.set_gripper_width(.064, 1000, timeout_s=1)
    assert world.held is None


def test_actual_piper_roundtrip_with_zero_and_nonzero_offset():
    module = pytest.importorskip('piper_robot.kinematics')
    cfg = simulation_config(load_config(), OVERLAY)
    for offset in ([0, 0, 0], [0, 0, .16]):
        cfg['workcell']['tcp_offset_m'] = offset
        ik, fk = build_piper_kinematics(cfg, kinematics_module=module)
        results = check_piper_roundtrip(ik, fk, cfg)
        assert len(results) == 3
        assert max(r['position_error_m'] for r in results) < .0015


def test_actual_piper_cycle_places_all_three_apples(tmp_path, monkeypatch):
    pytest.importorskip('piper_robot.kinematics')
    monkeypatch.setattr(run_cycle, 'load_config', load_config)
    trace = tmp_path / 'trace.json'
    rows = run_cycle.main(['--simulate', '--trace', str(trace), '--set',
                          f'orchestration.logging.csv_dir={tmp_path}'])
    assert len(rows) == 3 and all(r.success for r in rows)
    data = json.loads(trace.read_text())
    assert sum(e['action'] == 'grasp' for e in data['events']) == 3
    releases = [e for e in data['events'] if e['action'] == 'release']
    assert len(releases) == 3 and all(0 <= e['drop_m'] <= .03 for e in releases)
    targets = {(r.target_box.name, r.target_cell) for r in rows}
    assert targets == {('UPPER', 0), ('MIDDLE', 0), ('REJECT', 0)}
    assert sorted(a['center_m'][0] for a in data['apples']) == pytest.approx([.2] * 3, abs=.001)
    assert sorted(a['center_m'][1] for a in data['apples']) == pytest.approx([-.12, 0, .12], abs=.001)
    assert (tmp_path / 'cycles.csv').exists()
    moves = [e for e in data['events'] if e['action'] == 'move']
    assert any(e['speed_scale'] == .1 for e in moves)


def test_simulated_transport_blocks_unsafe_motion_and_unreachable_ik():
    pytest.importorskip('piper_robot.kinematics')
    cfg = simulation_config(load_config(), OVERLAY)
    world, adapter, _ = build_simulation(cfg)
    original = world.q
    unsafe = list(world.q)
    unsafe[0] = 10
    assert not world.move_joint_goal(unsafe, speed_scale=.1, timeout_s=10)
    assert world.q == original and not world.events
    from aple.execution.trajectory import ToolPose
    with pytest.raises(TrajectoryPlanningError, match='IK failed'):
        adapter._planner.plan([ToolPose((10, 0, .1), (0, 1, 0, 0))], world.q)
