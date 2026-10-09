"""The whole finite table must fit both synthetic and native top cameras."""
from pathlib import Path

import numpy as np
import pytest

from aple.hal.in_hand_simulation import scene_config
from aple.hal.gripper_physics import build_assumed_model, physics_scene


PROFILE=Path('configs/simulation/in_hand_full_floor.yaml')


@pytest.mark.parametrize('failure_stage',['read','send'])
def test_virtual_opening_propagates_invalid_state_and_ownership_loss(failure_stage):
    from types import SimpleNamespace
    from aple.hal.piper_studio import PiperWebError
    from tools.replay_in_hand_studio import open_virtual_gripper
    commands=[]
    def read():
        if failure_stage=='read':
            raise PiperWebError('encoder outside calibration')
        return (0,)*6,20.
    def send(target,keepalive,timeout):
        commands.append(target)
        raise PiperWebError('ownership lost')
    sim=SimpleNamespace(client=SimpleNamespace(_read_raw_state=read,
        _normalize_gripper_width=lambda width:94.),_send=send)
    with pytest.raises(PiperWebError):
        open_virtual_gripper(sim,.108,None)
    assert len(commands)==(0 if failure_stage=='read' else 1)


def test_all_floor_corners_fit_and_native_intrinsics_match():
    mj=pytest.importorskip('mujoco')
    pytest.importorskip('piper_sim')
    from tests.test_in_hand_simulation import config
    cfg,a=config()
    cfg=scene_config(cfg,PROFILE,seed=42)
    a.update(cfg['simulation']['in_hand_scene']['gripper_overrides'])
    model=build_assumed_model(physics_scene(cfg,a),a)
    camera=mj.mj_name2id(model,mj.mjtObj.mjOBJ_CAMERA,'top')
    table=mj.mj_name2id(model,mj.mjtObj.mjOBJ_GEOM,'table')
    assert np.allclose(model.cam_quat[camera],[1,0,0,0])
    assert np.allclose(model.cam_pos[camera],np.array(cfg['workcell']['T_base_camera'])[:3,3])
    height,width=cfg['simulation']['image_size']
    fx,fy,cx,cy=cfg['simulation']['intrinsics']
    assert fy==pytest.approx(height/(2*np.tan(np.deg2rad(model.cam_fovy[camera]/2))))
    transform=np.array(cfg['workcell']['T_base_camera'])
    center=model.geom_pos[table]
    half=model.geom_size[table]
    for x in (center[0]-half[0],center[0]+half[0]):
        for y in (center[1]-half[1],center[1]+half[1]):
            point=transform[:3,:3].T@(np.array([x,y,0])-transform[:3,3])
            pixel=[fx*point[0]/point[2]+cx,fy*point[1]/point[2]+cy]
            assert 0<pixel[0]<width and 0<pixel[1]<height


def test_camera_profile_keeps_scene_layout():
    from tests.test_in_hand_simulation import config, SETTINGS
    cfg,_=config()
    original=scene_config(cfg,SETTINGS,seed=42)
    wide=scene_config(cfg,PROFILE,seed=42)
    assert original['simulation']['apples']==wide['simulation']['apples']
    assert original['workcell']['boxes']==wide['workcell']['boxes']
    assert 'simulation' not in cfg or 'in_hand_scene' not in cfg['simulation']


def test_initial_virtual_reader_waits_for_real_valid_state(monkeypatch):
    from types import SimpleNamespace
    from tools.replay_in_hand_studio import initial_virtual_reader
    from aple.hal.piper_studio import PiperWebError
    import tools.replay_in_hand_studio as replay
    monkeypatch.setattr(replay.time,'sleep',lambda _:None)
    readings=[PiperWebError('Piper Web returned a gripper encoder value outside calibration'), ((0,)*6,98.)]
    def reader():
        value=readings.pop(0)
        if isinstance(value,Exception):
            raise value
        return value
    assert initial_virtual_reader(reader,SimpleNamespace(check=lambda:None))()==((0,)*6,98.)


def test_initial_virtual_reader_does_not_hide_invalid_arm_state():
    from types import SimpleNamespace
    from tools.replay_in_hand_studio import initial_virtual_reader
    from aple.hal.piper_studio import PiperWebError
    def reader():
        raise PiperWebError('invalid arm state')
    with pytest.raises(PiperWebError,match='invalid arm state'):
        initial_virtual_reader(reader,SimpleNamespace(check=lambda:None))()
