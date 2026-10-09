"""Optional simulation tuning must preserve real configs and physical safeguards."""
import copy
from pathlib import Path

import numpy as np
import pytest

from aple.hal.simulation_optimization import optimization_config, native_route_clear

PROFILE = Path('configs/simulation/in_hand_optimization.yaml')


def test_fast_profile_preserves_contact_descent_and_input_config():
    from tests.test_in_hand_simulation import config,SETTINGS
    from aple.hal.in_hand_simulation import scene_config
    cfg,_=config();cfg=scene_config(cfg,SETTINGS,seed=42)
    snapshot=copy.deepcopy(cfg)
    stable=optimization_config(cfg,PROFILE)
    fast=optimization_config(cfg,Path('configs/simulation/in_hand_fast.yaml'))
    assert cfg==snapshot
    old=stable['simulation']['in_hand_scene'];new=fast['simulation']['in_hand_scene']
    assert new['empty_joint_velocity_rad_s']>old['empty_joint_velocity_rad_s']
    assert new['loaded_joint_velocity_rad_s']>old['loaded_joint_velocity_rad_s']
    assert new['rotation_joint_velocity_rad_s']>old['rotation_joint_velocity_rad_s']
    assert new['loaded_joint_velocity_rad_s']*fast['execution']['soft_landing']['slow_speed_scale']==pytest.approx(
        old['loaded_joint_velocity_rad_s']*stable['execution']['soft_landing']['slow_speed_scale'])
    assert fast['inspection']['views']==stable['inspection']['views']
    assert fast['workcell']==stable['workcell']
    assert fast['piper_web']['poll_interval_s']<stable['piper_web']['poll_interval_s']


@pytest.mark.parametrize('key,value', [('loaded_joint_velocity_rad_s',0),('rotation_joint_velocity_rad_s',2),('studio_poll_interval_s',0)])
def test_invalid_fast_speed_settings_fail(tmp_path,key,value):
    import yaml
    from tests.test_in_hand_simulation import config,SETTINGS
    from aple.hal.in_hand_simulation import scene_config
    values=yaml.safe_load(Path('configs/simulation/in_hand_fast.yaml').read_text())
    values['optimization'][key]=value
    path=tmp_path/'invalid-fast.yaml';path.write_text(yaml.safe_dump(values))
    cfg,_=config()
    with pytest.raises(ValueError):
        optimization_config(scene_config(cfg,SETTINGS,seed=42),path)


def transport():
    pytest.importorskip('mujoco')
    pytest.importorskip('piper_sim')
    from tests.test_in_hand_simulation import config, SETTINGS
    from aple.hal.in_hand_simulation import scene_config, ContactTransport
    from aple.execution.trajectory import build_piper_kinematics
    cfg, assumptions = config()
    cfg = optimization_config(scene_config(cfg, SETTINGS, seed=42), PROFILE)
    assumptions.update(cfg['simulation']['in_hand_scene']['gripper_overrides'])
    _, fk = build_piper_kinematics(cfg)
    return ContactTransport(cfg, assumptions, fk)


def test_real_configuration_rejected_and_virtual_source_preserved():
    from aple.config import load_config
    from tests.test_in_hand_simulation import config, SETTINGS
    from aple.hal.in_hand_simulation import scene_config
    real = load_config()
    before = copy.deepcopy(real)
    with pytest.raises(ValueError, match='disabled-hardware'):
        optimization_config(real, PROFILE)
    assert real == before
    cfg, _ = config()
    cfg = scene_config(cfg, SETTINGS, seed=42)
    original = copy.deepcopy(cfg)
    optimized = optimization_config(cfg, PROFILE)
    assert cfg == original
    assert optimized['inspection'] == cfg['inspection']
    assert optimized['simulation']['apples'] == cfg['simulation']['apples']
    assert optimized['workcell'] == cfg['workcell']
    assert not optimized['piper_web']['enabled']


def test_native_fk_matches_official_fk_across_joint_states():
    from aple.inspection.in_hand_provider import pose_to_matrix
    t = transport()
    rng = np.random.default_rng(12)
    limits = np.array(t.cfg['safety']['joint_limits_rad'])
    for q in rng.uniform(limits[:,0], limits[:,1], size=(20,6)):
        for name, value in zip(t.arm, q):
            t.data.qpos[t.joints.qadr[name]] = value
        t.mj.mj_forward(t.model, t.data)
        assert np.allclose(t.native_tcp_matrix(), pose_to_matrix(t.tool_pose()), atol=1e-5, rtol=0)


def test_optimized_physics_still_stops_on_slip():
    from aple.execution.goal_builder import Pose3D, relative_pose
    t = transport()
    t.active, t.holding = 0, True
    t.apple_in_tcp = relative_pose(t.tool_pose(), Pose3D(tuple(t.positions()[0]),(0,0,0,1)))
    joint = t.model.body_jntadr[t.ids[0]]
    t.data.qpos[t.model.jnt_qposadr[joint]] += .05
    t.mj.mj_forward(t.model, t.data)
    with pytest.raises(RuntimeError, match='slipped'):
        t.step(t.model.opt.timestep)
    assert len(t.slips) == 1


def test_route_probe_rejects_robot_apple_contact_without_mutating_physics():
    t = transport()
    body = t.mj.mj_name2id(t.model, t.mj.mjtObj.mjOBJ_BODY, 'link4')
    joint = t.model.body_jntadr[t.ids[0]]
    t.data.qpos[t.model.jnt_qposadr[joint]:t.model.jnt_qposadr[joint]+3] = t.data.xpos[body]
    t.mj.mj_forward(t.model, t.data)
    before = t.data.qpos.copy()
    time_before = float(t.data.time)
    assert not native_route_clear(t, [t.joint_state()])
    assert np.array_equal(before, t.data.qpos)
    assert t.data.time == time_before


def test_route_selection_executes_only_safe_candidate(monkeypatch):
    from types import SimpleNamespace
    import aple.hal.simulation_optimization as opt
    executed = []
    t = SimpleNamespace(settings=dict(loaded_joint_velocity_rad_s=.35, empty_joint_velocity_rad_s=1.5),
        holding=False, route_choices=[], data=SimpleNamespace(time=0.),
        joint_state=lambda:(0,)*6,
        move_joint_goal=lambda point, **kwargs:executed.append(point) or True)
    adapter = SimpleNamespace(_transport=t, _deadline=lambda _:100., _remaining=lambda _:100.)
    unsafe, safe = [(0.01,)*6], [(0.1,)*6]
    monkeypatch.setattr(opt, 'planned_points', lambda adapter, poses:poses)
    monkeypatch.setattr(opt, 'native_route_clear', lambda transport, points:points != unsafe)
    opt.execute_shorter_transfer(adapter, [('short_unsafe', unsafe), ('long_safe',safe)], speed_scale=1., timeout_s=10.)
    assert executed == safe
    assert t.route_choices[0]['route']=='long_safe'
    assert t.route_choices[0]['rejected'][0]['route']=='short_unsafe'


@pytest.mark.parametrize('hover_z,expected', [(.11,.03), (.06,.02)])
def test_simulation_final_slow_distance_is_measured_from_release(monkeypatch, hover_z, expected):
    from types import SimpleNamespace
    from aple.execution.goal_builder import Pose3D, PlaceTargets
    from aple.execution.soft_landing import build_soft_landing_plan
    from aple.hal.in_hand_simulation import CentralInspectionAdapter
    from aple.hal.piper_motion import PiperTrajectoryMotionAdapter
    adapter = object.__new__(CentralInspectionAdapter)
    adapter._transport = SimpleNamespace(optimization={'exact_final_descent':True}, landing_segments=[])
    adapter._sim_place_targets = PlaceTargets(Pose3D((.3,0,.04),(0,0,0,1)),
        Pose3D((.3,0,hover_z),(0,0,0,1)), Pose3D((0,0,0),(0,0,0,1)))
    landing = build_soft_landing_plan(adapter._sim_place_targets, final_descent_m=.03,
                                      slow_speed_scale=.2, approach_speed_scale=1.)
    captured = []
    monkeypatch.setattr(PiperTrajectoryMotionAdapter,'execute_phase_b',
        lambda self, apple, profile, plan, targets, landing, **kwargs:captured.append(landing))
    adapter.execute_phase_b(SimpleNamespace(apple_id=1), None,None,None,landing,
        final_descent_m=.03,slow_speed_scale=.2)
    assert captured[0].slow_descent_start_z_m-.04 == pytest.approx(expected)
    assert adapter._transport.landing_segments[0]['slow_distance_m'] == pytest.approx(expected)
