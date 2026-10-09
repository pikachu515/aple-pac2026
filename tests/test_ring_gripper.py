"""Measured aperture must remain open to MuJoCo rays, on both fingers."""
import numpy as np
import pytest

from aple.hal.gripper_physics import optical_ring_dimensions


@pytest.mark.parametrize('values', [(78.664,78.664),(78.664,80),(78.664,None),(True,48.92)])
def test_invalid_ring_dimensions_fail(values):
    with pytest.raises(ValueError):
        optical_ring_dimensions(dict(zip(('opaque_cup_outer_diameter_mm','opaque_cup_inner_diameter_mm'),values)))


def test_measured_ring_rays_and_contact_mass():
    mj=pytest.importorskip('mujoco')
    pytest.importorskip('piper_sim')
    from tests.test_in_hand_simulation import config,SETTINGS
    from aple.hal.in_hand_simulation import scene_config
    from aple.hal.gripper_physics import build_assumed_model,physics_scene
    cfg,a=config();cfg=scene_config(cfg,SETTINGS,seed=42)
    outer,inner=optical_ring_dimensions(a)
    assert outer==pytest.approx(.039332) and inner==pytest.approx(.02446)
    model=build_assumed_model(physics_scene(cfg,a),a)
    baseline=a.copy()
    baseline.pop('opaque_cup_outer_diameter_mm');baseline.pop('opaque_cup_inner_diameter_mm')
    plain=build_assumed_model(physics_scene(cfg,baseline),baseline)
    np.testing.assert_allclose(model.body_mass,plain.body_mass)
    data=mj.MjData(model);mj.mj_forward(model,data)
    model.geom_group[:]=2
    for geom in range(model.ngeom):
        name=mj.mj_id2name(model,mj.mjtObj.mjOBJ_GEOM,geom) or ''
        if name.startswith(('aple_ring_','aple_pad_','aple_shaft_')):
            model.geom_group[geom]=1
        if name.startswith('aple_ring_'):
            assert model.geom_contype[geom]==model.geom_conaffinity[geom]==0
    groups=np.array([0,1,0,0,0,0],np.uint8)
    for sign,name in [(1,'finger_l'),(-1,'finger_r')]:
        body=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,name)
        rotation=data.xmat[body].reshape(3,3)
        center=data.xpos[body]+rotation@np.array([0,sign*(a['closed_contact_gap_m']/2-.008+.0015),a['flange_tcp_m'][2]-.065])
        normal=rotation@np.array([0,sign,0])
        hit=np.empty(1,np.int32)
        # Center and most of the hole, avoiding the opposite ring too.
        for x in [0.,.01,.0244]:
            point=center+rotation@np.array([x,0,0])
            assert mj.mj_ray(model,data,point+normal*.01,-normal,groups,1,-1,hit)<0
        point=center+rotation@np.array([.031,0,.005])
        assert mj.mj_ray(model,data,point+normal*.01,-normal,groups,1,-1,hit)>=0
