"""Simulation-only layouts, fruit physics and empty/loaded speed contracts."""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from aple.config import load_config
from aple.hal.gripper_physics import assumed_config, load_assumptions, physics_scene
from aple.hal.simulation import simulation_config
from aple.hal.in_hand_simulation import scene_config

SETTINGS = Path('configs/simulation/in_hand_scene.yaml')


def config():
    a = load_assumptions('configs/simulation/gripper_physics.yaml')
    return assumed_config(simulation_config(load_config(), 'configs/simulation/piper.yaml'), a), a


def test_mixed_layout_is_reproducible_and_keeps_real_config_unchanged():
    cfg, _ = config()
    before = copy.deepcopy(cfg)
    a = scene_config(cfg, SETTINGS, seed=42)
    assert a == scene_config(cfg, SETTINGS, seed=42)
    assert a != scene_config(cfg, SETTINGS, seed=7)
    assert cfg == before
    assert a['inspection']['policy'] == 'ALWAYS'


def test_separation_camera_center_and_representative_sizes_many_seeds():
    cfg, _ = config()
    for seed in range(12):
        out = scene_config(cfg, SETTINGS, seed=seed)
        settings = out['simulation']['in_hand_scene']
        center = np.array(settings['inspection_center_m'])
        assert np.allclose(center[:2], np.array(settings['map_xy_m']).mean(axis=1))
        fruit = out['simulation']['apples']
        points = [np.array(a['center_m'][:2]) for a in fruit]
        points += [np.array(b['origin_m'][:2])+np.array(b['grid'])*b['cell_pitch_m']/2 for b in out['workcell']['boxes'].values()]
        for i,p in enumerate(points):
            assert np.linalg.norm(p-center[:2]) >= settings['inspection_clear_radius_m']
            assert all(np.linalg.norm(p-q) >= settings['object_min_distance_m'] for q in points[:i])
        assert [a['mass_kg'] for a in fruit] == [.252, .2971, .337]
        assert np.allclose([2*a['radius_m'] for a in fruit], [.0826, .0873, .0919])
        assert settings['empty_joint_velocity_rad_s'] > settings['loaded_joint_velocity_rad_s'] >= settings['rotation_joint_velocity_rad_s']


def test_scene_builder_preserves_individual_masses():
    cfg,a = config()
    out = scene_config(cfg, SETTINGS, seed=42)
    apples = [o for o in physics_scene(out,a)['objects'] if o['shape']=='sphere']
    assert [o['mass'] for o in apples] == [.252,.2971,.337]


@pytest.mark.parametrize('seed', [-1, True])
def test_invalid_seed(seed):
    cfg,_ = config()
    with pytest.raises(ValueError, match='seed'):
        scene_config(cfg, SETTINGS, seed=seed)


def test_camera_must_be_at_actual_map_midpoint(tmp_path):
    cfg,_ = config()
    values = yaml.safe_load(SETTINGS.read_text())
    values['in_hand_scene']['inspection_center_m'][0] += .02
    path = tmp_path/'invalid.yaml'
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match='midpoint'):
        scene_config(cfg,path,seed=42)


def contact_transport():
    pytest.importorskip('mujoco')
    pytest.importorskip('piper_robot.kinematics')
    pytest.importorskip('piper_sim')
    from aple.execution.trajectory import build_piper_kinematics
    from aple.hal.in_hand_simulation import ContactTransport
    cfg,a = config()
    cfg = scene_config(cfg, SETTINGS, seed=42)
    a.update(cfg['simulation']['in_hand_scene']['gripper_overrides'])
    _,fk = build_piper_kinematics(cfg)
    return ContactTransport(cfg,a,fk)


def test_perception_id_does_not_select_another_physical_apple_and_skin_rotates():
    from types import SimpleNamespace
    from aple.execution.goal_builder import Pose3D, relative_pose
    transport = contact_transport()
    transport.active = 1
    relative = relative_pose(transport.tool_pose(), Pose3D(tuple(transport.positions()[1]), (0,0,0,1)))
    transport.begin(SimpleNamespace(apple_id=99), relative)
    assert transport.active == 1
    before = transport.capture_in_hand(99,0,timeout_s=10)
    assert before.snapshot.apple_id == 99
    assert before.mask.any()
    joint = transport.model.body_jntadr[transport.ids[1]]
    address = transport.model.jnt_qposadr[joint]
    transport.data.qpos[address+3:address+7] = [0,1,0,0]  # 180 deg about apple-local X
    transport.mj.mj_forward(transport.model,transport.data)
    after = transport.capture_in_hand(99,1,timeout_s=10)
    assert not np.allclose(before.rotation_camera_apple,after.rotation_camera_apple)
    assert np.count_nonzero(before.snapshot.color != after.snapshot.color) > 100


def test_physical_displacement_stops_motion_and_records_slip():
    from aple.execution.goal_builder import Pose3D, relative_pose
    transport = contact_transport()
    transport.active = 0
    transport.holding = True
    transport.apple_in_tcp = relative_pose(transport.tool_pose(), Pose3D(tuple(transport.positions()[0]),(0,0,0,1)))
    joint = transport.model.body_jntadr[transport.ids[0]]
    address = transport.model.jnt_qposadr[joint]
    transport.data.qpos[address] += .05  # Inject a slip into the test's physical state.
    transport.mj.mj_forward(transport.model, transport.data)
    with pytest.raises(RuntimeError, match='slipped'):
        transport.step(transport.model.opt.timestep)
    assert len(transport.slips) == 1
    assert transport.slips[0]['displacement_m'] > transport.settings['max_slip_m']


def test_web_replay_rejects_failed_offline_run_before_network(tmp_path):
    import json
    from tools.replay_in_hand_studio import main
    (tmp_path/'report.json').write_text(json.dumps(dict(success=False)))
    with pytest.raises(ValueError,match='successful offline'):
        main([str(tmp_path)])


def test_web_replay_model_mismatch_does_not_change_scene_or_start_jog(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from tools.replay_in_hand_studio import main
    from aple.hal.piper_studio import PiperWebError
    cfg,a = config()
    cfg = scene_config(cfg, SETTINGS, seed=42)
    (tmp_path/'report.json').write_text(json.dumps(dict(success=True,assumptions=a,apples=[])))
    (tmp_path/'virtual-config.yaml').write_text(yaml.safe_dump(cfg))
    (tmp_path/'trace.json').write_text(json.dumps(dict(events=[dict(action='move',joints_rad=[0]*6,joint_velocity_rad_s=1)],apples=[])))
    class GuardedReplay:
        def __init__(self,*args,**kwargs):
            self.client = SimpleNamespace(_normalize_joint_target=lambda q:{})
        def _request(self,method,path):
            assert method=='GET' and path=='/api/sim/scenes/live/objects'
            return dict(objects=[dict(aple_assumptions_sha256='wrong model')])
        def prepare(self,*args,**kwargs):
            pytest.fail('Model mismatch must not mutate the scene')
        def session(self):
            pytest.fail('Model mismatch must not acquire motion')
    monkeypatch.setattr('aple.hal.studio_simulation.StudioSimulationReplay',GuardedReplay)
    with pytest.raises(PiperWebError,match='assumptions differ'):
        main([str(tmp_path)])


def test_virtual_top_camera_is_above_the_exact_marked_map_center():
    transport = contact_transport()
    cid = transport.mj.mj_name2id(transport.model, transport.mj.mjtObj.mjOBJ_CAMERA,'top')
    assert np.allclose(transport.model.cam_pos[cid,:2], transport.settings['inspection_center_m'][:2])
