"""붙어 놓인 사과 (10/9 실기: 파란 트레이에 사과를 한 줄로 붙여 놓자 깊이 분할이 38~105 mm 로 틀렸고,
가운데 사과는 핑거 틈이 없다고 보류됐다).

- 검출: 사과색 마스크를 모양(거리 변환)으로 나눈다 (segment: color) — 깊이가 흔들려도 직경이 맞다
- 검출: 집는 영역 경계에 걸친 사과는 넓힌 영역에서 통째로 찾고 중심이 영역 안인 것만 남긴다
- 파지: 틈이 좁으면 한쪽 여유를 줄여 핑거를 틈에 넣고, 그 폭만큼만 벌린다 (adaptive_open)
"""

import copy

import numpy as np
import pytest

from aple.config import load_config
from aple.execution.pick_place_pipeline import _with_candidate_opening
from aple.perception import detector
from aple.perception.grasp_planner import plan_grasps
from aple.perception.sim_scene import INTR, SIZE, camera_looking_down
from aple.perception.sim_scene import render as sim_render
from aple.schema import AppleState, CaptureMode, GraspProfile, SceneSnapshot, Source

BLUE_BGR = (180, 130, 40)
RED_BGR = (40, 40, 200)
T = camera_looking_down((0.30, 0.0), 0.65)
R = 0.04


def scene(spheres, depth_noise_m=0.0):
    depth, who, _ = sim_render(T, spheres, noise_m=depth_noise_m, seed=3)
    color = np.empty(SIZE + (3,), np.uint8)
    color[:] = BLUE_BGR
    color[who >= 0] = RED_BGR
    return SceneSnapshot(1, Source.TOP, CaptureMode.FULL, color, depth, INTR, 0.0, tray_clear=True), who


def cfg_color(**detect):
    c = copy.deepcopy(load_config())
    c["workcell"]["T_base_camera"] = T.tolist()
    c["perception"]["detect"].update(segment="color", min_apple_color=0.3, **detect)
    return c


ROW = [((0.30, y, R), R) for y in (-2 * R, 0.0, 2 * R)]          # 표면이 맞닿은 세 사과


def diam_px_m(det, depth):
    z = float(np.median(depth[det.mask & (depth > 0)]))
    return 2 * np.sqrt(det.mask.sum() / np.pi) * z / INTR[0]


def test_color_segmentation_splits_touching_row_despite_noisy_depth():
    snap, who = scene(ROW, depth_noise_m=0.02)                    # 깊이 잡음 2 cm — 깊이로는 못 나눈다
    dets = detector.detect(snap, cfg_color())
    assert len(dets) == 3
    for d in dets:
        assert diam_px_m(d, snap.depth_m) == pytest.approx(2 * R, rel=0.12)


def test_apple_straddling_pick_area_edge_is_found_whole():
    snap, who = scene([((0.30, 0.0, R), R)])
    c = cfg_color()
    # 집는 영역의 한 변이 사과 중심 근처를 지난다 (영역 x 0.20~0.31) — 넓히지 않으면 사과가 잘린다
    c["workcell"]["tray"].update(center_m=[0.255, 0.0, 0.0], size_m=[0.11, 0.30])
    cut = detector.detect(snap, c)
    c["perception"]["detect"].update(roi_margin_m=0.06, keep_margin_m=0.03)
    whole = detector.detect(snap, c)
    assert len(whole) == 1 and diam_px_m(whole[0], snap.depth_m) == pytest.approx(2 * R, rel=0.1)
    assert not cut or diam_px_m(cut[0], snap.depth_m) < 0.85 * 2 * R
    c["workcell"]["tray"].update(center_m=[0.10, 0.0, 0.0], size_m=[0.11, 0.30])   # 영역 밖 사과는 남기지 않는다
    assert detector.detect(snap, c) == []


def apple(i, x, y, r=R):
    return AppleState(scene_id=1, apple_id=i, det_id=i, center_m=(x, y, r), diameter_m=2 * r,
                      top_surface_z_m=2 * r, clearance_m=0.0, fit_residual_m=0.0, confidence=1.0,
                      depth_valid_ratio=1.0)


def grasp_cfg(adaptive):
    c = copy.deepcopy(load_config())
    c["workcell"]["tray"].update(center_m=[0.30, 0.0, 0.0], size_m=[0.10, 0.50])   # x 쪽은 벽이 가깝다
    c["grasp_planner"].update(min_open_margin_m=0.003 if adaptive else 0.01, adaptive_open=adaptive,
                              tilt_start_m=None, yaw_step_deg=90)   # yaw 0°(x)·90°(y) 두 방향만
    return c


GAP = 0.026                                                       # 이웃 사과 표면까지 26 mm (핑거 두께·폭 12·20 mm)
NEIGHBORS = [apple(1, 0.30, 0.0), apple(2, 0.30, 2 * R + GAP), apple(3, 0.30, -(2 * R + GAP))]


def test_fingers_go_into_narrow_gap_with_reduced_opening():
    fixed = {g.apple_id: g for g in plan_grasps(NEIGHBORS, grasp_cfg(False))}
    adaptive = {g.apple_id: g for g in plan_grasps(NEIGHBORS, grasp_cfg(True))}
    assert not fixed[1].collision_free                            # 한쪽 여유 10 mm 로는 틈에 안 들어간다
    g = adaptive[1]
    assert g.collision_free
    assert abs(np.cos(g.grasp_yaw_rad)) < 0.3                     # 이웃 쪽(±y)으로 닫는다 — x 는 벽
    assert 2 * R + 2 * 0.003 - 1e-9 <= g.open_width_m < 2 * R + 2 * 0.01   # 기본보다 좁게 벌린다


def test_open_tray_side_lets_fingers_cross_paper_edge():
    a = [apple(1, 0.225, 0.0)]                                    # 집는 영역 x_min 변(종이 쪽)에 붙은 사과
    c = grasp_cfg(True)
    c["workcell"]["tray"].update(size_m=[0.16, 0.08])
    assert not plan_grasps(a, c)[0].collision_free
    c["grasp_planner"]["tray_open_sides"] = ["x_min"]
    assert plan_grasps(a, c)[0].collision_free


def test_profile_takes_candidate_opening_within_gripper_range():
    c = load_config()
    c["piper_web"]["gripper_width_range_m"] = [0.058, 0.12]
    prof = GraspProfile(scene_id=1, apple_id=1, size_bin=2, est_mass_kg=0.2, open_width_m=0.09,
                        grasp_height_offset_m=0.0, grasp_force_n=0.0, effort=1000, feasible=True)
    cand = type("C", (), {"open_width_m": 0.086})()
    assert _with_candidate_opening(prof, cand, c).open_width_m == pytest.approx(0.086)
    assert _with_candidate_opening(prof, type("C", (), {"open_width_m": 0.2})(), c).open_width_m == pytest.approx(0.12)
    assert _with_candidate_opening(prof, None, c).open_width_m == pytest.approx(0.09)
