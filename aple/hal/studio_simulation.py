"""Studio 0.5.12 MuJoCo trajectory replay, restricted to sim_follower1.

Replays planned arm/gripper commands. It does not assert physical grasp success:
Studio's stock fingers do not represent the configured handmade TCP/adapters.
"""
from __future__ import annotations

import copy
import math
import time
from contextlib import contextmanager
from pathlib import Path

from aple.hal.piper_studio import PiperStudio, PiperWebError

SIM_IFACE = 'sim_follower1'


def scene_from_config(cfg):
    """A separate Studio scene of virtual apples and destination bins (SI)."""
    objects = []
    for i, apple in enumerate(cfg['simulation']['apples']):
        red = float(apple['red_fraction'])
        objects.append(dict(id=f'aple_apple_{i+1}', label=f'Apple {i+1}', shape='sphere',
                            size=[float(apple['radius_m'])], pos=list(apple['center_m']),
                            rgba=[.2 + .6 * red, .75 - .6 * red, .1, 1], mass=.1, movable=True))
    for grade, box in cfg['workcell']['boxes'].items():
        pitch = float(box['cell_pitch_m'])
        rows, cols = box['grid']
        x, y, z = box['origin_m']
        objects.append(dict(id=f'aple_{grade.lower()}', label=grade, shape='preset:bin',
                            pos=[x + pitch * rows / 2, y + pitch * cols / 2, z], movable=False,
                            params=dict(inner=[pitch * rows / 2, pitch * cols / 2],
                                        wall_t=.002, wall_h=float(box['wall_height_m']) / 2,
                                        floor_t=.001)))
    return dict(version=1, name='APLE planned trajectory', cameras={'top': {'z': .8}}, objects=objects)


class StudioSimulationReplay:
    def __init__(self, cfg, *, base_url=None, iface=SIM_IFACE, client=None):
        if iface != SIM_IFACE:
            raise ValueError('Studio replay permits only sim_follower1; hardware interfaces are forbidden')
        config = copy.deepcopy(cfg)
        config['piper_web']['iface'] = SIM_IFACE
        config['piper_web']['enabled'] = True
        if base_url is not None:
            config['piper_web']['base_url'] = base_url
        self.client = client or PiperStudio(config)
        if self.client._iface != SIM_IFACE:
            raise ValueError('client must target sim_follower1')
        self._owned = False

    def _request(self, method, path, body=None):
        if self.client._iface != SIM_IFACE:
            raise PiperWebError('Replay client destination changed; refusing request')
        return self.client._request(method, path, body)

    def _arm(self):
        payload = self._request('GET', '/api/robots/current')
        if not isinstance(payload, dict) or not isinstance(payload.get('arms'), list):
            raise PiperWebError('Invalid Studio arm registry')
        matches = [a for a in payload['arms'] if isinstance(a, dict) and a.get('iface') == SIM_IFACE]
        if len(matches) != 1 or matches[0].get('transport') != 'sim':
            raise PiperWebError('sim_follower1 must be declared transport=sim by Studio')
        return matches[0]

    def prepare(self, scene, *, backup_path):
        # Read authentication rules from the server. The sim-only path supports
        # Studio's local password-disabled mode; hardware HAL rules are unchanged.
        self.client._ensure_authenticated(require_control=False)
        self._idle()
        self._request('GET', '/api/robots/ports')  # sync simd restart/disconnection facts
        registry = self._request('GET', '/api/robots/current')
        if not any(a.get('iface') == SIM_IFACE for a in registry.get('arms', [])):
            self._request('GET', '/api/robots/can')  # Studio discovery includes simd
        arm = self._arm()
        if arm.get('connected') is not True:
            result = self._request('POST', '/api/robots/attach', {'iface': SIM_IFACE})
            if not isinstance(result, dict) or result.get('transport') != 'sim':
                raise PiperWebError('Studio did not confirm simulated arm attachment')
        arm = self._arm()
        if arm.get('connected') is not True or arm.get('role') != 'follower':
            raise PiperWebError('Simulated arm must be connected as follower')
        listing = self._request('GET', '/api/sim/scenes')
        previous = listing.get('current')
        backup = {'previous_scene_id': previous, 'previous_scene': None}
        if previous:
            backup['previous_scene'] = self._request('GET', f'/api/sim/scenes/{previous}')
        import json
        path = Path(backup_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        original = path
        number = 2
        while path.exists():
            path = original.with_name(f'{original.stem}-{number}{original.suffix}')
            number += 1
        with path.open('x', encoding='utf-8') as out:
            out.write(json.dumps(backup, ensure_ascii=False, indent=2))
        result = self._request('POST', '/api/sim/scenes/import',
                               {'text': json.dumps(scene), 'name': scene['name']})
        sid = result.get('id') if isinstance(result, dict) else None
        if not isinstance(sid, str) or not sid or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789_-' for c in sid):
            raise PiperWebError('Invalid imported Studio scene id')
        self._request('POST', f'/api/sim/scenes/{sid}/apply', {})
        return sid

    def connect_virtual_cameras(self, camera_ids=('sim:top', 'sim:front')):
        if any(camera_id not in ('sim:top', 'sim:front', 'sim:side') for camera_id in camera_ids):
            raise ValueError('Only approved simulated viewing cameras may be connected')
        cameras = self._request('GET', '/api/cameras/scan')
        if not isinstance(cameras, list):
            raise PiperWebError('Invalid Studio camera registry')
        for camera_id in camera_ids:
            matches = [c for c in cameras if isinstance(c, dict) and c.get('id') == camera_id]
            if len(matches) != 1 or matches[0].get('cam_type') != 'sim':
                raise PiperWebError(f'{camera_id} must be declared as a simulated camera')
            if matches[0].get('connected') is not True:
                result = self._request('POST', '/api/cameras/connect', {'id': camera_id})
                if result.get('connected') is not True or result.get('cam_type') != 'sim':
                    raise PiperWebError(f'Studio did not confirm simulated camera {camera_id}')

    def _idle(self):
        status = self._request('GET', '/api/robots/jog/status')
        if not isinstance(status, dict) or status.get('running') is not False:
            raise PiperWebError('Studio jog is already active; refusing to take over')

    @contextmanager
    def session(self):
        self._idle()
        arm = self._arm()
        if arm.get('connected') is not True or arm.get('role') != 'follower':
            raise PiperWebError('Simulated follower is not connected')
        keepalive = self.client._start_estop_keepalive()
        try:
            result = self._request('POST', '/api/robots/jog/start', {'iface': SIM_IFACE})
            self._owned = isinstance(result, dict) and result.get('iface') == SIM_IFACE and result.get('running') is True
            if not self._owned or result.get('status') != 'started':
                raise PiperWebError('Studio did not confirm simulated jog ownership')
            yield keepalive
        finally:
            try:
                if self._owned:
                    status = self._request('GET', '/api/robots/jog/status')
                    if status.get('iface') != SIM_IFACE:
                        raise PiperWebError('Jog ownership changed; refusing to stop another arm')
                    result = self._request('POST', '/api/robots/jog/stop', {})
                    if result.get('status') != 'stopped':
                        raise PiperWebError('Studio did not confirm jog stop')
            finally:
                self._owned = False
                keepalive.stop()
                keepalive.check()

    def _send(self, target, keepalive, timeout_s):
        if not self._owned:
            raise PiperWebError('No owned simulated jog session')
        self._arm()  # transport must still be sim immediately before a command
        status = self._request('GET', '/api/robots/jog/status')
        if status.get('iface') != SIM_IFACE or status.get('running') is not True:
            raise PiperWebError('Simulated jog ownership lost')
        result = self._request('POST', '/api/robots/jog/goal', {'iface': SIM_IFACE, 'values': target})
        if result.get('status') != 'ok':
            raise PiperWebError('Studio rejected simulated goal')
        deadline = time.monotonic() + timeout_s
        target_rad = self.client._denormalize_joint_target(target) if 'joint1' in target else None
        while True:
            keepalive.check()
            q, gripper = self.client._read_raw_state()
            arm_ok = target_rad is None or all(abs(a-b) <= self.client._arrival_tolerance_rad for a, b in zip(q, target_rad))
            grip_ok = 'gripper' not in target or abs(gripper-target['gripper']) <= self.client._gripper_arrival_tolerance
            if arm_ok and grip_ok:
                return q
            if time.monotonic() >= deadline:
                raise PiperWebError('Simulated goal did not arrive; contact/limits/daemon state need inspection')
            time.sleep(self.client._poll_interval_s)

    def replay(self, events, *, timeout_s=15, minimum_step_s=.1, on_progress=None):
        if not math.isfinite(timeout_s) or timeout_s <= 0 or not math.isfinite(minimum_step_s) or minimum_step_s < 0:
            raise ValueError('Replay timing must be finite with positive timeout and nonnegative delay')
        # Validate all commands before obtaining any motion session.
        targets = []
        normalized = {}
        for event in events:
            if event.get('action') == 'move':
                normalized = self.client._normalize_joint_target(tuple(event['joints_rad']))
                targets.append(dict(normalized))
            elif event.get('action') == 'gripper':
                width = self.client._normalize_gripper_width(event['width_m'])
                targets.append({**normalized, 'gripper': width})
        if not targets:
            raise ValueError('Trace has no arm/gripper commands')
        with self.session() as keepalive:
            for i, target in enumerate(targets, 1):
                self._send(target, keepalive, timeout_s)
                if on_progress:
                    on_progress(i, len(targets))
                time.sleep(minimum_step_s)
        return len(targets)
