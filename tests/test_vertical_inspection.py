"""Vertical inspection keeps the apple center and the jaw-closing axis fixed."""
import copy
from pathlib import Path

import numpy as np
import pytest

from aple.execution.goal_builder import Pose3D, compose
from aple.hal.in_hand_simulation import inspection_poses, scene_config
from aple.hal.simulation_optimization import optimization_config
from aple.inspection.in_hand_provider import pose_to_matrix


PROFILE = Path('configs/simulation/in_hand_vertical.yaml')


def test_pitch_keeps_center_and_fingers_horizontal_without_local_roll():
    settings = dict(inspection_center_m=[.325,0,.24], tilt_rad=np.pi/2,
        rotation_angles_rad=[.872664626,.581776417,.290888209,0.],
        optimization={'inspection_rotation_mode':'vertical_pitch'})
    relative = Pose3D((.004,-.002,.003), (0,0,0,1))
    poses = inspection_poses(settings,relative)
    assert len(poses)==4
    for pose in poses:
        assert np.allclose(compose(pose,relative).position_m,settings['inspection_center_m'])
        rotation=pose_to_matrix(pose)[:3,:3]
        assert np.allclose(rotation[:,1],[0,1,0])
    assert not np.allclose(pose_to_matrix(poses[0])[:3,2],pose_to_matrix(poses[-1])[:3,2])


def test_vertical_profile_retains_real_config_and_four_views():
    from tests.test_in_hand_simulation import config, SETTINGS
    cfg,_=config()
    cfg=scene_config(cfg,SETTINGS,seed=42)
    before=copy.deepcopy(cfg)
    result=optimization_config(cfg,PROFILE)
    assert cfg==before
    assert len(result['inspection']['views']['rotation_angles_rad'])==4
    assert result['inspection']['policy']=='ALWAYS'
    assert result['workcell']==cfg['workcell']


def test_opaque_surface_blocks_camera_rays_and_does_not_add_contact_mass():
    mj=pytest.importorskip('mujoco')
    pytest.importorskip('piper_sim')
    from tests.test_in_hand_simulation import config, SETTINGS
    from aple.hal.gripper_physics import build_assumed_model, physics_scene
    cfg,a=config();cfg=scene_config(cfg,SETTINGS,seed=42)
    a.update(cfg['simulation']['in_hand_scene']['gripper_overrides'])
    # Exercise archived solid-disk behavior independently of today's ring.
    a.pop('opaque_cup_outer_diameter_mm',None)
    a.pop('opaque_cup_inner_diameter_mm',None)
    baseline=build_assumed_model(physics_scene(cfg,a),a)
    a['opaque_cup_radius_m']=.04
    opaque=build_assumed_model(physics_scene(cfg,a),a)
    assert np.allclose(baseline.body_mass,opaque.body_mass)
    data=mj.MjData(opaque);mj.mj_forward(opaque,data)
    gid=mj.mj_name2id(opaque,mj.mjtObj.mjOBJ_GEOM,'aple_opaque_finger_l')
    assert gid>=0 and opaque.geom_rgba[gid,3]==1.
    assert opaque.geom_contype[gid]==0 and opaque.geom_conaffinity[gid]==0
    center=data.geom_xpos[gid].copy()
    normal=data.geom_xmat[gid].reshape(3,3)[:,2].copy()
    # Avoid the existing contact pad at the center of the optical disk.
    center += data.geom_xmat[gid].reshape(3,3)[:,:2]@np.array([.01,.01])
    hit=np.empty(1,np.int32)
    # Ray along the disk normal, starting outside its front face.
    distance=mj.mj_ray(opaque,data,center+normal*.01,-normal,None,1,-1,hit)
    assert distance>=0 and hit[0]==gid


def test_partial_replay_rejects_invalid_subset_even_with_explicit_option(tmp_path):
    import json
    from tools.replay_in_hand_studio import main
    (tmp_path/'report.json').write_text(json.dumps(dict(success=True,full_scene_complete=False,
        tested_apple_count=1,apples=[dict(apple=0)])))
    (tmp_path/'virtual-config.yaml').write_text('{}')
    (tmp_path/'trace.json').write_text(json.dumps(dict(apples=[{}],events=[])))
    with pytest.raises(ValueError,match='Invalid tested apple IDs'):
        main([str(tmp_path),'--allow-partial'])
