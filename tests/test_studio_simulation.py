"""Sim-only destination checks and jog cleanup, without hardware or HTTP."""
import copy
from types import SimpleNamespace

import pytest

from aple.config import load_config
from aple.hal.simulation import simulation_config
from aple.hal.piper_studio import PiperStudio, PiperWebError
from aple.hal.studio_simulation import StudioSimulationReplay, scene_from_config


def config():
    return simulation_config(load_config(), 'configs/simulation/piper.yaml')


class FakeStudio(PiperStudio):
    def __init__(self):
        cfg = config()
        cfg['piper_web']['iface'] = 'sim_follower1'
        super().__init__(cfg)
        self.calls = []
        self.transport = 'sim'
        self.running = False
        self.target = self._normalize_joint_target((0, 1, -1, 0, 0, 0))
        self.arrive = True
        self.heartbeat_stopped = False
        self.fail_goal = False
        self.beat = SimpleNamespace(check=lambda: None, stop=self.stop_heartbeat)

    def stop_heartbeat(self):
        self.heartbeat_stopped = True

    def _start_estop_keepalive(self):
        return self.beat

    def _ensure_authenticated(self, **kwargs):
        pass

    def _request(self, method, path, body=None, **kwargs):
        self.calls.append((method, path, copy.deepcopy(body)))
        if path == '/api/robots/ports':
            return {'sim': []}
        if path == '/api/cameras/scan':
            return [{'id': camera_id, 'cam_type': 'sim', 'connected': False}
                    for camera_id in ('sim:top', 'sim:front')]
        if path == '/api/cameras/connect':
            return {'id': body['id'], 'connected': True, 'cam_type': 'sim'}
        if path == '/api/robots/current':
            return {'arms': [{'iface': 'sim_follower1', 'transport': self.transport,
                             'connected': True, 'role': 'follower'}]}
        if path == '/api/robots/jog/status':
            return {'running': self.running, 'iface': 'sim_follower1' if self.running else None}
        if path == '/api/robots/jog/start':
            self.running = True
            return {'running': True, 'iface': body['iface'], 'status': 'started'}
        if path == '/api/robots/jog/goal':
            if self.fail_goal:
                raise PiperWebError('failed goal')
            if self.arrive:
                self.target.update(body['values'])
            return {'status': 'ok'}
        if path == '/api/robots/jog/stop':
            self.running = False
            return {'status': 'stopped'}
        if path == '/api/sim/scenes':
            return {'current': 'my-old-scene'}
        if path == '/api/sim/scenes/my-old-scene':
            return {'name': 'Existing scene', 'objects': []}
        if path == '/api/sim/scenes/import':
            return {'id': 'aple-planned-trajectory'}
        if path.endswith('/apply'):
            return {'ok': True}
        raise AssertionError(path)

    def _read_raw_state(self):
        return self._denormalize_joint_target(self.target), self.target.get('gripper', 100)


def trace():
    return [{'action': 'move', 'joints_rad': [0, 1.2, -1.1, 0, .1, 0]},
            {'action': 'grasp', 'apple_index': 0},
            {'action': 'gripper', 'width_m': .064, 'effort': 1000}]


def test_hardware_interface_is_rejected_before_any_io():
    with pytest.raises(ValueError, match='hardware'):
        StudioSimulationReplay(config(), iface='can0')
    client = FakeStudio()
    client._iface = 'can0'
    with pytest.raises(ValueError, match='sim_follower1'):
        StudioSimulationReplay(config(), client=client)
    assert client.calls == []


def test_declared_can_transport_is_rejected_before_jog_start():
    client = FakeStudio()
    client.transport = 'can'
    replay = StudioSimulationReplay(config(), client=client)
    with pytest.raises(PiperWebError, match='transport=sim'):
        replay.replay(trace(), minimum_step_s=0)
    assert not any(c[0] == 'POST' for c in client.calls)


def test_replay_sends_only_sim_goals_and_closes_session():
    client = FakeStudio()
    replay = StudioSimulationReplay(config(), client=client)
    assert replay.replay(trace(), minimum_step_s=0) == 2
    goals = [c[2] for c in client.calls if c[1].endswith('/jog/goal')]
    assert all(g['iface'] == 'sim_follower1' for g in goals)
    assert goals[-1]['values']['gripper'] == pytest.approx(.064 / .07 * 100)
    assert client.heartbeat_stopped and not client.running
    assert not any('gripper-effort' in c[1] for c in client.calls)


def test_server_failure_stops_owned_jog_and_heartbeat():
    client = FakeStudio()
    client.fail_goal = True
    with pytest.raises(PiperWebError, match='failed goal'):
        StudioSimulationReplay(config(), client=client).replay(trace(), minimum_step_s=0)
    assert client.heartbeat_stopped and not client.running


def test_busy_jog_is_not_stopped_or_taken_over():
    client = FakeStudio()
    client.running = True
    with pytest.raises(PiperWebError, match='already active'):
        StudioSimulationReplay(config(), client=client).replay(trace(), minimum_step_s=0)
    assert client.running
    assert not any(c[0] == 'POST' for c in client.calls)


def test_prepare_imports_new_scene_and_preserves_previous(tmp_path):
    import json
    client = FakeStudio()
    path = tmp_path / 'previous.json'
    replay = StudioSimulationReplay(config(), client=client)
    assert replay.prepare(scene_from_config(config()), backup_path=path) == 'aple-planned-trajectory'
    assert json.loads(path.read_text())['previous_scene_id'] == 'my-old-scene'
    assert not any(c[0] in ('PUT', 'DELETE') for c in client.calls)
    assert not any('/jog/start' in c[1] for c in client.calls)


def test_invalid_trace_cannot_start_a_motion_session():
    client = FakeStudio()
    events = trace() + [{'action': 'move', 'joints_rad': [float('nan')] * 6}]
    with pytest.raises(ValueError):
        StudioSimulationReplay(config(), client=client).replay(events, minimum_step_s=0)
    assert not any(c[0] == 'POST' for c in client.calls)


def test_initial_gripper_command_preserves_parked_arm_joints():
    client = FakeStudio()
    original = client._denormalize_joint_target(client.target)
    events = [{'action': 'gripper', 'width_m': .07}]
    assert StudioSimulationReplay(config(), client=client).replay(events, minimum_step_s=0) == 1
    goals = [c[2]['values'] for c in client.calls if c[1].endswith('/jog/goal')]
    assert goals == [{'gripper': 100}]
    assert client._denormalize_joint_target(client.target) == original


def test_virtual_cameras_only_are_connected():
    client = FakeStudio()
    StudioSimulationReplay(config(), client=client).connect_virtual_cameras()
    connected = [c[2]['id'] for c in client.calls if c[1] == '/api/cameras/connect']
    assert connected == ['sim:top', 'sim:front']


def test_changed_client_destination_cannot_make_requests():
    client = FakeStudio()
    replay = StudioSimulationReplay(config(), client=client)
    client._iface = 'can0'
    with pytest.raises(PiperWebError, match='destination changed'):
        replay.replay(trace(), minimum_step_s=0)
    assert not client.calls


def test_scene_uses_si_spheres_and_box_cell_centers():
    cfg = config()
    scene = scene_from_config(cfg)
    apples = [o for o in scene['objects'] if o['shape'] == 'sphere']
    assert len(apples) == 3 and apples[0]['size'] == [.032]
    middle = next(o for o in scene['objects'] if o['id'] == 'aple_middle')
    assert middle['pos'] == pytest.approx([.2, 0, 0])
