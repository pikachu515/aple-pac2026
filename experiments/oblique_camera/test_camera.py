"""Camera geometry and current M1 behavior under tilted viewpoints."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from aple.hal.in_hand_simulation import scene_config
from aple.hal.simulation import KinematicWorld
from aple.perception.pose_estimator import estimate
from aple.schema import AppleDetection2D
from experiments.oblique_camera.comparison import camera_transform


@pytest.mark.parametrize('degrees', [0, 30, 45, 60])
def test_correctly_calibrated_tilt_preserves_sphere_center(degrees):
    from tests.test_in_hand_simulation import config
    cfg, _ = config()
    cfg = scene_config(cfg, 'configs/simulation/in_hand_full_floor.yaml', seed=42)
    transform = camera_transform(np.radians(degrees))
    cfg['workcell']['T_base_camera'] = transform.tolist()
    # This legacy analytic test fixture requires every image ray to intersect
    # the table. Narrow its field of view to keep the 60-degree case below the
    # horizon; the separate MuJoCo audit retains the original 55-degree fovy.
    cfg['simulation']['intrinsics'] = [600., 600., 320., 240.]
    truth = np.array([.325, 0., .0413])
    cfg['simulation']['apples'] = [dict(center_m=truth.tolist(), radius_m=.0413, red_fraction=.5)]
    world = KinematicWorld(cfg, None, SimpleNamespace(is_segment_safe=lambda *args: True))
    snap = world.capture()
    h, w = snap.depth_m.shape
    v, u = np.indices((h, w))
    fx, fy, cx, cy = snap.intrinsics
    rays = np.stack(((u-cx)/fx, (v-cy)/fy, np.ones_like(u)), axis=-1) @ transform[:3, :3].T
    table_depth = -transform[2, 3]/rays[..., 2]
    mask = snap.depth_m < table_depth-.0001
    detection = AppleDetection2D(snap.scene_id, 1, cv2.boundingRect(mask.astype(np.uint8)), 1., mask)
    measured = estimate(snap, [detection], transform, None, cfg)
    assert len(measured) == 1
    assert np.linalg.norm(np.array(measured[0].center_m)-truth) < .003
    # Characterize the existing support fallback; this is not a production fix.
    snap.depth_m[mask] = 0
    fallback = estimate(snap, [detection], transform, None, cfg)
    assert len(fallback) == 1 and fallback[0].depth_valid_ratio == 0.
    error = np.linalg.norm(np.array(fallback[0].center_m)-truth)
    if degrees == 0:
        assert error < .003
    elif degrees >= 45:
        assert error > .01


def test_comparison_does_not_change_physics_or_baseline_events():
    from experiments.oblique_camera.audit_perception import setup
    from experiments.oblique_camera.comparison import CameraComparisonTransport
    cfg, original = setup(42)
    cfg['simulation']['in_hand_scene']['optimization']['inspection_physics_settle_s'] = 0.
    transport = CameraComparisonTransport(cfg, original.a, original.fk)
    transport.active = 0
    before = transport.data.qpos.copy(), transport.data.time
    transform = np.array(cfg['workcell']['T_base_camera'])
    events, views, scene = len(transport.events), len(transport.views), transport.scene_id
    transport.capture_in_hand(1, 0, timeout_s=10)
    np.testing.assert_array_equal(transport.data.qpos, before[0])
    assert transport.data.time == before[1]
    np.testing.assert_array_equal(cfg['workcell']['T_base_camera'], transform)
    assert len(transport.events) == events+1
    assert len(transport.views) == views+1
    assert transport.scene_id == scene+1
    assert all(len(v) == 1 for v in transport.camera_views.values())
