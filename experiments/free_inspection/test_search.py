"""Check search constraints and disposable planning state, without hardware."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from aple.execution.goal_builder import Pose3D
from aple.execution.segment_validator import build_piper_segment_validator
from aple.execution.trajectory import build_piper_kinematics
from aple.hal.in_hand_simulation import ContactTransport, scene_config
from aple.hal.simulation_optimization import optimization_config
from aple.inspection.coloration import fibonacci_sphere, params_from_config
from experiments.free_inspection.adapter import FreeInspectionAdapter

EXPERIMENT = Path(__file__).resolve().parent


@pytest.fixture
def adapter():
    pytest.importorskip('mujoco')
    pytest.importorskip('piper_sim')
    from tests.test_in_hand_simulation import config
    cfg, assumptions = config()
    overrides = yaml.safe_load((EXPERIMENT/'scene.yaml').read_text())['in_hand_scene']['gripper_overrides']
    assumptions.update(overrides)
    cfg = scene_config(cfg, EXPERIMENT/'scene.yaml', seed=42)
    cfg = optimization_config(cfg, EXPERIMENT/'optimization.yaml')
    cfg['free_inspection'] = yaml.safe_load((EXPERIMENT/'search.yaml').read_text())['free_inspection']
    assumptions['opaque_cup_radius_m'] = cfg['simulation']['in_hand_scene']['optimization']['opaque_cup_radius_m']
    ik, fk = build_piper_kinematics(cfg)
    transport = ContactTransport(cfg, assumptions, fk)
    # Search geometry fixture only. The end-to-end runner measures this pose
    # from the actual contact grasp instead of substituting a nominal pose.
    transport.active = 0
    transport.apple_in_tcp = Pose3D((0., 0., 0.), (0., 0., 0., 1.))
    return FreeInspectionAdapter(transport, cfg, ik, build_piper_segment_validator(cfg, fk))


def test_joint_search_keeps_jaws_horizontal_and_varies_xyz(adapter):
    joints, report = adapter.candidates()
    limits = np.array(adapter._cfg['safety']['joint_limits_rad'])
    assert np.all(joints >= limits[:, 0]) and np.all(joints <= limits[:, 1])
    matrices = adapter.matrices(joints)
    assert np.all(np.abs(matrices[:, 2, 1]) <= .02)
    assert np.all(np.ptp(matrices[:, :3, 3], axis=0) > .1)
    assert np.all(np.ptp(joints, axis=0) > .1)
    assert report['geometric_candidates'] > report['direction_candidates']


def test_visibility_probe_preserves_real_world_state(adapter):
    transport = adapter._transport
    joints, _ = adapter.candidates()
    snapshot = (transport.data.qpos.copy(), transport.data.qvel.copy(), transport.data.time)
    probe = transport.mj.MjData(transport.model)
    probe.qpos[:] = transport.data.qpos
    surface = fibonacci_sphere(params_from_config(adapter._cfg).surface_bins)
    visible = adapter.visibility(joints[0], probe, surface)
    assert visible.dtype == np.bool_ and visible.shape == (len(surface),)
    np.testing.assert_array_equal(transport.data.qpos, snapshot[0])
    np.testing.assert_array_equal(transport.data.qvel, snapshot[1])
    assert transport.data.time == snapshot[2]


def test_held_apple_floor_clearance_is_checked_separately(adapter, monkeypatch):
    joints = np.array([adapter._transport.joint_state()])
    adapter._planner = SimpleNamespace(plan_joint_goal=lambda goal, start: SimpleNamespace(points_rad=joints))
    monkeypatch.setattr('experiments.free_inspection.adapter.native_route_clear', lambda transport, points: True)
    # Put the predicted center below the floor without moving the real world.
    matrix = adapter.matrices(joints)[0]
    relative = matrix[:3, :3].T @ (np.array([.3, 0., -.2]) - matrix[:3, 3])
    adapter._transport.apple_in_tcp = Pose3D(tuple(relative), (0., 0., 0., 1.))
    before = adapter._transport.data.qpos.copy()
    with pytest.raises(ValueError, match='floor'):
        adapter.safe_path(joints[0], joints[0])
    np.testing.assert_array_equal(adapter._transport.data.qpos, before)
