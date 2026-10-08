"""모드 B 촬영 공급자 — 핑거가 가린 손 안 사과의 마스크와, 재판정까지 이어지는 흐름.

합성 장면에 핑거(사과 표면보다 12 mm 앞, 양옆 띠)를 그려 넣고 Piper Studio 8bit 깊이 양자화를 적용한다.
"""

from dataclasses import replace

import numpy as np
import pytest

from aple.config import load_config
from aple.execution.goal_builder import Pose3D
from aple.inspection.geometry import Sphere, pixel_rays, ray_sphere
from aple.inspection.in_hand_provider import (InHandCaptureProvider, in_hand_mask, mask_params_from_config,
                                              pose_to_matrix)
from aple.inspection.inhand import InHandSlip, apple_center_cam, rot_cam_apple
from aple.schema import CaptureMode, SceneSnapshot, Source
from tests.synthetic import INTR, Apple, full_truth, render, side_band
from tests.test_inhand import APPLE_IN_TCP, POSES, T_BASE_CAM, tcp_pose
from tests.test_inspection import P
from tests.test_pick_place_pipeline import apple, cfg  # noqa: F401
from tests.test_regrade_pose_sphere import RADIUS_M, held_with_captures
from tests.test_slip_check import piper_depth

PARAMS = mask_params_from_config(load_config())
PATTERN = side_band(0.3)
FINGER_GAP_M = 0.012


def scene_with_fingers(true_center, R=np.eye(3), **render_kw):
    """→ (color, depth, 보이는 사과 정답 마스크, 핑거 마스크)."""
    c = np.asarray(true_center, float)
    color, depth, masks = render([Apple(tuple(c), RADIUS_M, PATTERN, R)], **render_kw)
    fx, fy, cx, cy = INTR
    cu, cv = fx * c[0] / c[2] + cx, fy * c[1] / c[2] + cy
    r_px = fx * RADIUS_M / c[2]
    v, u = np.indices(depth.shape)
    dx = np.abs(u - cu) / r_px
    finger = (dx >= 0.55) & (dx <= 1.25) & (np.abs(v - cv) <= 0.4 * r_px)
    inside = finger & masks[0]
    depth = depth.copy()
    depth[inside] -= FINGER_GAP_M                         # 사과 앞을 가린 부분
    depth[finger & ~masks[0]] = c[2] - 0.005              # 실루엣 밖의 핑거 몸통
    color = color.copy()
    color[finger] = (90, 90, 90)
    depth = np.where(depth > 0, piper_depth(depth), 0.0).astype(np.float32)
    return color, depth, masks[0] & ~finger, finger


def judged(mask, center):
    """색 판정에 실제로 쓰이는 픽셀 (시선–법선 cosθ ≥ cos_theta_min, coloration 과 같은 기준)."""
    vs, us = np.nonzero(mask)
    rays = pixel_rays(us, vs, INTR)
    t, hit = ray_sphere(rays, Sphere(np.asarray(center, float), RADIUS_M))
    n = rays * t[:, None] - center
    cos = -(n / np.linalg.norm(n, axis=1, keepdims=True) * rays).sum(axis=1)
    out = np.zeros_like(mask)
    out[vs, us] = hit & (cos >= P.cos_theta_min)
    return out


def test_pose_to_matrix_from_quaternion():
    s = np.sqrt(0.5)
    T = pose_to_matrix(Pose3D((0.1, 0.2, 0.3), (0.0, 0.0, s, s)))   # z 축 90°
    assert np.allclose(T[:3, :3], [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-9)
    assert np.allclose(T[:3, 3], [0.1, 0.2, 0.3])
    assert pose_to_matrix(T) is not None


def test_mask_excludes_fingers_and_keeps_visible_apple():
    for R_base_tcp in POSES:
        T_tcp = tcp_pose(R_base_tcp)
        center = apple_center_cam(T_BASE_CAM, T_tcp, APPLE_IN_TCP)
        _, depth, truth, finger = scene_with_fingers(center, rot_cam_apple(T_BASE_CAM, T_tcp))
        mask = in_hand_mask(depth, INTR, center, RADIUS_M, PARAMS)
        assert (judged(mask, center) & finger).sum() == 0                  # 색 판정 영역에는 핑거가 하나도 없다
        assert (mask & finger).sum() / mask.sum() < 0.002                  # 실루엣 테두리(cosθ≈0) 몇 픽셀뿐
        assert (mask & truth).sum() / truth.sum() > 0.9                    # 보이는 사과는 거의 다 잡는다


def test_mask_follows_slipped_apple():
    predicted = np.array([0.0, 0.0, 0.45])
    actual = predicted + [0.0, 0.015, 0.0]                                 # 15 mm 밀림
    _, depth, truth, _ = scene_with_fingers(actual)
    mask = in_hand_mask(depth, INTR, predicted, RADIUS_M, PARAMS)
    assert (mask & truth).sum() / truth.sum() > 0.9
    assert (mask & ~truth).sum() / mask.sum() < 0.01


def provider_for(shifts=None, *, dropout=0.0, calls=None, **kwargs):
    """POSES 순서로 장면을 찍는 공급자. joint_state 로 자세 번호를 넘긴다. shifts: {뷰: 실제 사과 이동(m)}."""
    shifts = shifts or {}
    state = {}

    def tool_pose_fn(view):
        state["view"] = view
        return tcp_pose(POSES[view])

    def snapshot_fn():
        if calls is not None:
            calls.append(state["view"])
        T_tcp = tcp_pose(POSES[state["view"]])
        center = apple_center_cam(T_BASE_CAM, T_tcp, APPLE_IN_TCP) + np.asarray(shifts.get(state["view"], 0.0))
        color, depth, _, _ = scene_with_fingers(center, rot_cam_apple(T_BASE_CAM, T_tcp), depth_dropout=dropout)
        return SceneSnapshot(0, Source.TOP, CaptureMode.FULL, color, depth, INTR, 0.0)

    provider = InHandCaptureProvider(snapshot_fn, tool_pose_fn, T_BASE_CAM, PARAMS, **kwargs)
    provider.begin(apple(), Pose3D(tuple(APPLE_IN_TCP), (0.0, 0.0, 0.0, 1.0)))
    return provider


def regrade_with(cfg, provider):  # noqa: F811
    cfg["workcell"]["T_base_camera"] = T_BASE_CAM.tolist()
    pipeline, held = held_with_captures(cfg)
    captures = tuple(provider(1, i, i, timeout_s=5.0) for i in range(len(POSES)))
    held = replace(held, inspection=replace(held.inspection, captures=captures))
    return captures, pipeline.regrade_held_apple(held, P)


def test_provider_feeds_regrade_end_to_end(cfg):  # noqa: F811
    captures, held = regrade_with(cfg, provider_for())
    for i, cap in enumerate(captures):
        s = cap.snapshot
        assert (s.source, s.capture_mode, s.apple_id, s.view_index) == (Source.IN_HAND, CaptureMode.FULL, 1, i)
        assert np.allclose(s.tool_pose, tcp_pose(POSES[i]))
    ids = [c.snapshot.scene_id for c in captures]
    assert ids == sorted(set(ids)) and ids[0] > apple().scene_id
    assert held.regraded_grade is not None


def test_provider_masks_give_correct_ratio(cfg, monkeypatch):  # noqa: F811
    from aple.execution import pick_place_pipeline as pipeline_module
    seen = {}
    real = pipeline_module.accumulate_views

    def recording(views, params, **kw):
        seen["ratio"] = real(views, params, **kw)
        return seen["ratio"]

    monkeypatch.setattr(pipeline_module, "accumulate_views", recording)
    regrade_with(cfg, provider_for())
    r, truth = seen["ratio"], full_truth(PATTERN)
    assert r.ratio_lo <= truth <= r.ratio_hi
    assert abs(r.ratio - truth) <= (1 - r.coverage) + 0.02
    assert r.coverage < 0.85                                               # 핑거 가림이 커버리지에 반영된다


def test_slip_is_caught_through_the_provider(cfg):  # noqa: F811
    with pytest.raises(InHandSlip, match="view 2"):
        regrade_with(cfg, provider_for({2: [0.0, 0.02, 0.0]}))


def test_missing_depth_retakes_then_leaves_decision_to_regrade(cfg):  # noqa: F811
    calls = []
    provider = provider_for(dropout=0.995, calls=calls)
    cap = provider(1, 0, 0, timeout_s=5.0)
    assert calls == [0] * (PARAMS.retries + 1)                             # 같은 자세에서 다시 찍었다
    assert cap is not None and cap.mask.sum() < PARAMS.min_pixels


def test_provider_requires_begin_and_uses_scene_id_source():
    provider = InHandCaptureProvider(lambda: None, lambda j: np.eye(4), T_BASE_CAM, PARAMS)
    with pytest.raises(ValueError, match="begin"):
        provider(1, 0, None, timeout_s=1.0)
    ids = iter([40, 41])
    provider = provider_for(scene_id_fn=lambda: next(ids))
    assert [provider(1, i, i, timeout_s=5.0).snapshot.scene_id for i in range(2)] == [40, 41]


def test_from_config_requires_calibration():
    with pytest.raises(ValueError, match="T_base_camera"):
        InHandCaptureProvider.from_config(load_config(), lambda: None, lambda j: np.eye(4))


def test_ring_finger_face_on_keeps_apple_seen_through_the_hole():
    """링형 핑거(10/7 설계 변경)가 카메라를 향할 때: 고리는 빼고, 가운데 구멍으로 보이는 사과는 잡는다."""
    center = np.array([0.0, 0.0, 0.45])
    color, depth, masks = render([Apple(tuple(center), RADIUS_M, PATTERN, np.eye(3))])
    fx, fy, cx, cy = INTR
    r_px = fx * RADIUS_M / center[2]
    v, u = np.indices(depth.shape)
    d = np.hypot(u - cx, v - cy) / r_px
    ring = (d >= 0.45) & (d <= 0.75)                                       # 고리 (구멍 반지름 0.45 r)
    hole = (d < 0.45) & masks[0]
    depth = depth.copy()
    depth[ring & masks[0]] -= 0.008                                       # 고리는 사과 표면보다 8 mm 앞
    depth = np.where(depth > 0, piper_depth(depth), 0.0).astype(np.float32)
    mask = in_hand_mask(depth, INTR, center, RADIUS_M, PARAMS)
    assert (judged(mask, center) & ring).sum() < 0.001 * mask.sum()        # 판정 영역에 고리 픽셀 없음 (깊이 잡음 꼬리 몇 개)
    assert (mask & hole).sum() / hole.sum() > 0.95                         # 구멍으로 보이는 사과는 잡는다
    outer = masks[0] & (d > 0.75)
    assert (mask & outer).sum() / outer.sum() > 0.85                       # 고리 바깥 사과도 잡는다
