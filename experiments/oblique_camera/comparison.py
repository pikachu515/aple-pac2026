"""Camera-only comparisons on identical real simulated contact-grasp states."""
import copy
import math
from pathlib import Path

import numpy as np
import yaml

from aple.hal.in_hand_simulation import ContactTransport
from aple.inspection.coloration import InHandView, accumulate_views, params_from_config
from aple.inspection.geometry import Sphere

SETTINGS = yaml.safe_load((Path(__file__).parent/'settings.yaml').read_text())['camera_audit']


def camera_transform(tilt_rad):
    c, s = math.cos(tilt_rad), math.sin(tilt_rad)
    a = SETTINGS['azimuth_rad']
    ca, sa = math.cos(a), math.sin(a)
    azimuth = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1.]])
    transform = np.eye(4)
    transform[:3, :3] = azimuth @ np.array([[c, 0, -s], [0, -1, 0], [-s, 0, -c]])
    transform[:3, 3] = np.array(SETTINGS['target_m']) + SETTINGS['standoff_m'] * (azimuth @ np.array([s, 0., c]))
    return transform


class CameraComparisonTransport(ContactTransport):
    def __init__(self, *args, **kwargs):
        self.camera_views = {tilt: [] for tilt in SETTINGS['tilt_rad']}
        self.camera_capture_stats = {tilt: [] for tilt in SETTINGS['tilt_rad']}
        super().__init__(*args, **kwargs)

    def capture_in_hand(self, apple_id, view_index, *, timeout_s):
        # Baseline settles physics exactly once. The alternate cameras do not
        # advance physics, change the robot, or repaint the apple's body pattern.
        baseline = super().capture_in_hand(apple_id, view_index, timeout_s=timeout_s)
        saved_transform = copy.deepcopy(self.cfg['workcell']['T_base_camera'])
        saved_settle = self.optimization.get('inspection_physics_settle_s', 0.)
        saved_callback = self.capture_callback
        self.optimization['inspection_physics_settle_s'] = 0.
        self.capture_callback = None
        event_count, view_count, scene_id = len(self.events), len(self.views), self.scene_id
        try:
            for tilt in SETTINGS['tilt_rad']:
                transform = camera_transform(tilt)
                self.cfg['workcell']['T_base_camera'] = transform.tolist()
                capture = baseline if tilt == 0. else super().capture_in_hand(apple_id, view_index, timeout_s=timeout_s)
                center = (self.positions()[self.active] - transform[:3, 3]) @ transform[:3, :3]
                radius = self.cfg['simulation']['apples'][self.active]['radius_m']
                snap = capture.snapshot
                # Ground-truth sphere for visibility audit, not for M1 estimates.
                self.camera_views[tilt].append(InHandView(snap.color, snap.depth_m, snap.intrinsics,
                    capture.mask, capture.rotation_camera_apple, Sphere(center, radius)))
                self.camera_capture_stats[tilt].append(dict(view=view_index, visible_pixels=int(capture.mask.sum())))
        finally:
            self.cfg['workcell']['T_base_camera'] = saved_transform
            self.optimization['inspection_physics_settle_s'] = saved_settle
            self.capture_callback = saved_callback
            del self.events[event_count:]
            del self.views[view_count:]
            self.scene_id = scene_id
        return baseline

    def camera_report(self):
        params = params_from_config(self.cfg)
        results = []
        for tilt, views in self.camera_views.items():
            ratio = accumulate_views(views, params) if views else None
            results.append(dict(tilt_deg=math.degrees(tilt), T_base_camera=camera_transform(tilt).tolist(),
                captured_views=len(views), coverage=ratio.coverage if ratio else None,
                captures=self.camera_capture_stats[tilt]))
        return results
