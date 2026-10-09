"""M1 인식 · 3D 위치 · 상자 점유 · 카메라–로봇 변환 (역할 A).

정답을 아는 장면을 base_link 기준으로 그린다: 위에서 내려다보는 카메라, 테이블(z=0) 위 구형 사과,
상자(바닥 + 벽 윗면). 깊이는 카메라 z (D435 와 같은 의미)이며 잡음·결측을 넣을 수 있다.
"""

from __future__ import annotations

import copy
import math
import shutil

import numpy as np
import pytest

from aple.config import CONFIG_DIR, load_config
from aple.hal.calibration import (base_to_pixel, fit_rigid_transform, load_T_base_camera, pixel_to_base,
                                  save_T_base_camera, solve_eye_to_hand, validate_T)
from aple.perception import box_occupancy, detector, pose_estimator
from aple.perception.sim_scene import INTR, SIZE, box_tops, camera_looking_down
from tests.conftest import pristine_copy
from aple.perception.sim_scene import render as sim_render
from aple.planning.placement_planner import PlacementPlanner
from aple.schema import (CaptureMode, Grade, GradeResult, InspectMode, RejectReason, SceneSnapshot, Source)

CAM_XY = (0.30, 0.0)
CAM_H = 0.65


# ================================================================ 장면

def make_T(tilt_rad: float = 0.0) -> np.ndarray:
    """카메라를 (0.30, 0, 0.65) 에 아래로 향하게. 영상 오른쪽 = base −y, 영상 아래 = base −x."""
    return camera_looking_down(CAM_XY, CAM_H, tilt_rad)


def render(T, spheres=(), tops=(), **kw):
    """→ depth_m (H×W), who (H×W, 사과 번호 또는 −1)."""
    depth, who, _ = sim_render(T, spheres, tops, **kw)
    return depth, who


def snap_of(depth, scene_id=1, tray_clear=True):
    color = np.full(SIZE + (3,), 128, np.uint8)
    return SceneSnapshot(scene_id, Source.TOP, CaptureMode.FULL, color, depth, INTR, 0.0, tray_clear=tray_clear)


def apple_on_table(x, y, r):
    return ((x, y, r), r)


@pytest.fixture
def cfg():
    c = copy.deepcopy(load_config())
    c["workcell"]["T_base_camera"] = make_T().tolist()
    return c


def iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


# ================================================================ 카메라–로봇 변환

def test_pixel_base_roundtrip():
    T = make_T(0.08)
    p = pixel_to_base(400.0, 120.0, 0.6, INTR, T)
    uv, z = base_to_pixel(p[None], INTR, T)
    assert np.allclose(uv[0], [400.0, 120.0]) and np.isclose(z[0], 0.6)
    with pytest.raises(ValueError):
        pixel_to_base(10, 10, 0.0, INTR, T)


def test_validate_rejects_non_rigid():
    bad = np.eye(4)
    bad[0, 0] = 1.1
    with pytest.raises(ValueError):
        validate_T(bad)
    with pytest.raises(ValueError):
        validate_T(np.eye(3))


def test_save_and_load_T_in_workcell_yaml(tmp_path):
    path = pristine_copy("workcell.yaml", tmp_path / "workcell.yaml")
    with pytest.raises(ValueError, match="캘리브레이션"):
        load_T_base_camera(str(path))
    T = make_T(0.05)
    save_T_base_camera(str(path), T)
    assert np.allclose(load_T_base_camera(str(path)), T, atol=1e-6)
    assert "expected_apple_count" in path.read_text(encoding="utf-8")      # 다른 줄은 그대로


def test_fit_rigid_transform_recovers_camera_pose():
    rng = np.random.default_rng(3)
    T = make_T(0.1)
    base = rng.uniform([0.15, -0.25, 0.0], [0.45, 0.25, 0.1], size=(8, 3))
    cam = (base - T[:3, 3]) @ T[:3, :3] + rng.normal(0, 0.001, base.shape)
    T_fit, rms = fit_rigid_transform(cam, base)
    assert rms < 0.002
    assert np.allclose(T_fit[:3, 3], T[:3, 3], atol=0.003)
    with pytest.raises(ValueError):
        fit_rigid_transform(np.outer(np.arange(4), [1, 0, 0]), np.outer(np.arange(4), [0, 1, 0]))


def test_solve_eye_to_hand_recovers_camera_pose():
    rng = np.random.default_rng(1)
    T_bc = make_T(0.07)
    T_gt = np.eye(4)
    T_gt[:3, 3] = [0.0, 0.02, 0.05]                        # 그리퍼에 물린 보드

    def pose(rv, t):
        import cv2
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(np.asarray(rv, float))[0]
        T[:3, 3] = t
        return T

    T_bg = [pose(rng.uniform(-0.6, 0.6, 3), rng.uniform([0.2, -0.2, 0.15], [0.4, 0.2, 0.35])) for _ in range(12)]
    T_ct = [np.linalg.inv(T_bc) @ Tb @ T_gt for Tb in T_bg]
    T = solve_eye_to_hand(T_bg, T_ct)
    assert np.allclose(T, T_bc, atol=1e-4)


# ================================================================ 검출

APPLES3 = [apple_on_table(0.25, 0.10, 0.038), apple_on_table(0.33, -0.05, 0.035), apple_on_table(0.20, -0.15, 0.042)]


def test_detects_separated_apples_with_good_masks(cfg):
    depth, who = render(make_T(), APPLES3)
    dets = detector.detect(snap_of(depth, scene_id=7), cfg)
    assert len(dets) == 3
    assert [d.det_id for d in dets] == [1, 2, 3]
    assert all(d.scene_id == 7 for d in dets)
    for i in range(3):
        best = max(iou(d.mask, who == i) for d in dets)
        assert best > 0.9


def test_splits_touching_apples(cfg):
    pair = [apple_on_table(0.30, 0.04, 0.037), apple_on_table(0.30, -0.034, 0.037)]   # 표면 간격 0
    depth, who = render(make_T(), pair)
    dets = detector.detect(snap_of(depth), cfg)
    assert len(dets) == 2
    for i in range(2):
        assert max(iou(d.mask, who == i) for d in dets) > 0.8


def test_splits_touching_apples_of_different_size(cfg):
    pair = [apple_on_table(0.30, 0.045, 0.045), apple_on_table(0.30, -0.032, 0.032)]
    depth, who = render(make_T(), pair)
    dets = detector.detect(snap_of(depth), cfg)
    assert len(dets) == 2
    for i in range(2):
        assert max(iou(d.mask, who == i) for d in dets) > 0.85


def test_detection_survives_glossy_depth_holes(cfg):
    depth, _ = render(make_T(), APPLES3, noise_m=0.0015, dropout=0.3)
    assert len(detector.detect(snap_of(depth), cfg)) == 3


def test_yolo_boxes_without_raised_object_are_rejected(cfg):
    depth, who = render(make_T(), APPLES3[:2])
    boxes = []
    for i in range(2):
        vs, us = np.nonzero(who == i)
        boxes.append((us.min() - 3, vs.min() - 3, us.max() + 3, vs.max() + 3, 0.9 - 0.1 * i))
    boxes.append((20, 20, 80, 80, 0.95))                    # 테이블만 있는 곳 — 오검출
    boxes.append((300, 300, 340, 340, 0.1))                 # conf 미달
    dets = detector.detect(snap_of(depth), cfg, model=lambda img: boxes)
    assert sorted(round(d.confidence, 2) for d in dets) == [0.8, 0.9]
    for i in range(2):
        assert max(iou(d.mask, who == i) for d in dets) > 0.9


def test_tall_post_is_not_an_apple(cfg):
    """의자 다리·로봇 팔처럼 직경보다 훨씬 높이 솟은 물체는 사과가 아니다 (2026-10-07 실측 오검출)."""
    post = [(0.40, 0.435, 0.12, 0.155, 0.20)]               # 3.5 cm 각 기둥, 높이 20 cm
    depth, _ = render(make_T(), APPLES3, tops=post)
    dets = detector.detect(snap_of(depth), cfg)
    assert len(dets) == 3


def test_apple_cut_by_image_border_is_dropped(cfg):
    T = make_T()
    edge = pixel_to_base(2, 240, CAM_H, INTR, T)            # 영상 왼쪽 끝 열이 지나는 테이블 위치
    depth, _ = render(T, APPLES3 + [apple_on_table(edge[0], edge[1], 0.036)])
    dets = detector.detect(snap_of(depth), cfg)
    assert len(dets) == 3
    assert not any(d.bbox[0] == 0 for d in dets)


def test_coco_weights_keep_only_apple_class(cfg, monkeypatch):
    ultralytics = pytest.importorskip("ultralytics")
    seen = {}

    class FakeYOLO:
        def __init__(self, weights):
            seen["weights"] = weights

        def predict(self, img, **kw):
            seen.update(kw)
            raise StopIteration

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    run = detector.load_yolo({"weights": "fake_coco.pt", "classes": [47], "conf": 0.4})
    with pytest.raises(StopIteration):
        run(np.zeros((8, 8, 3), np.uint8))
    assert seen["classes"] == [47]
    detector._MODEL_CACHE.pop("fake_coco.pt", None)


def test_tray_roi_limits_detection(cfg):
    cfg["workcell"]["tray"]["center_m"] = [0.25, 0.0, 0.0]
    cfg["workcell"]["tray"]["size_m"] = [0.20, 0.22]        # x 0.15–0.35, y −0.11–0.11
    depth, _ = render(make_T(), APPLES3)                    # 세 번째 사과는 y = −0.15 → 트레이 밖
    dets = detector.detect(snap_of(depth), cfg)
    assert len(dets) == 2


def test_apples_inside_boxes_are_not_tray_apples(cfg):
    origin = (0.20, 0.12, 0.01)
    cfg["workcell"]["boxes"]["UPPER"].update(grid=[2, 3], cell_pitch_m=0.06, origin_m=list(origin))
    spheres = [apple_on_table(0.30, -0.10, 0.036), ((0.23, 0.15, 0.01 + 0.03), 0.03)]   # 두 번째는 상자 안
    depth, _ = render(make_T(), spheres, tops=box_tops(origin, 2, 3, 0.06, 0.08))
    dets = detector.detect(snap_of(depth), cfg)
    assert len(dets) == 1


# ================================================================ 3D 위치 · 직경

def run_m1(cfg, depth, prev=None, scene_id=1, tray_clear=True):
    snap = snap_of(depth, scene_id=scene_id, tray_clear=tray_clear)
    dets = detector.detect(snap, cfg)
    return snap, dets, pose_estimator.estimate(snap, dets, make_T(), prev, cfg)


@pytest.mark.parametrize("noise, dropout, step", [(0.0, 0.0, 0.0), (0.0015, 0.25, 0.0), (0.002, 0.3, 0.004)])
def test_center_and_diameter_accuracy(cfg, noise, dropout, step):
    """역할 A 목표: 직경 오차 ±3 mm (전략 4-4). 합성 장면에서는 ±2 mm 로 본다."""
    apples = [apple_on_table(0.22, 0.17, 0.032), apple_on_table(0.36, -0.02, 0.040),
              apple_on_table(0.18, -0.20, 0.045)]           # 영상 가장자리 쪽 포함 (원근 왜곡)
    depth, _ = render(make_T(), apples, noise_m=noise, dropout=dropout, seed=4)
    if step:                                                  # Piper Studio 8비트 깊이 인코딩 (한 단계 약 4 mm)
        depth = np.where(depth > 0, np.floor(depth / step) * step + step / 2, 0).astype(np.float32)
    _, _, states = run_m1(cfg, depth)
    assert len(states) == 3
    for (c, r) in apples:
        s = min(states, key=lambda s: np.linalg.norm(np.subtract(s.center_m, c)))
        assert abs(s.diameter_m - 2 * r) < 0.002
        assert np.linalg.norm(np.subtract(s.center_m, c)) < 0.003
        assert abs(s.top_surface_z_m - 2 * r) < 0.004
        assert s.confidence >= 0.5 and not s.occluded


def test_tilted_camera_still_gives_base_positions(cfg):
    T = make_T(0.12)
    cfg["workcell"]["T_base_camera"] = T.tolist()
    apples = [apple_on_table(0.25, 0.05, 0.038), apple_on_table(0.32, -0.10, 0.036)]
    depth, _ = render(T, apples)
    snap = snap_of(depth)
    states = pose_estimator.estimate(snap, detector.detect(snap, cfg), T, None, cfg)
    for (c, r) in apples:
        s = min(states, key=lambda s: np.linalg.norm(np.subtract(s.center_m, c)))
        assert np.linalg.norm(np.subtract(s.center_m, c)) < 0.003 and abs(s.diameter_m - 2 * r) < 0.002


def test_touching_apples_keep_their_size(cfg):
    pair = [apple_on_table(0.30, 0.04, 0.037), apple_on_table(0.30, -0.034, 0.037)]
    _, _, states = run_m1(cfg, render(make_T(), pair)[0])
    assert len(states) == 2
    for s in states:
        assert abs(s.diameter_m - 0.074) < 0.003
        assert s.clearance_m < 0.003                         # 맞닿아 있다


def test_clearance_between_neighbours(cfg):
    pair = [apple_on_table(0.30, 0.06, 0.035), apple_on_table(0.30, -0.06, 0.035)]   # 표면 간격 0.05
    _, _, states = run_m1(cfg, render(make_T(), pair)[0])
    assert all(abs(s.clearance_m - 0.05) < 0.004 for s in states)


def test_depth_loss_lowers_confidence_but_keeps_position(cfg):
    depth, _ = render(make_T(), APPLES3[:1], dropout=0.75, seed=2)
    _, _, states = run_m1(cfg, depth)
    s = states[0]
    assert s.depth_valid_ratio < 0.4 and s.confidence <= 0.5
    assert abs(s.diameter_m - 0.076) < 0.003


def test_no_depth_on_apple_falls_back_to_support_plane(cfg):
    """마스크는 있는데 사과 깊이가 전부 빠진 경우 (G1): 둘레 받침면 깊이로 중심을 추정한다."""
    from aple.schema import AppleDetection2D
    depth, who = render(make_T(), APPLES3[:1])
    depth[who == 0] = 0.0
    snap = snap_of(depth)
    det = AppleDetection2D(1, 1, (0, 0, 1, 1), 0.9, who == 0)
    (s,) = pose_estimator.estimate(snap, [det], make_T(), None, cfg)
    assert s.depth_valid_ratio == 0.0 and s.confidence <= 0.45
    assert abs(s.diameter_m - 0.076) < 0.004
    assert np.linalg.norm(np.subtract(s.center_m, APPLES3[0][0])) < 0.005


def test_tracking_keeps_ids_and_drops_picked_apples(cfg):
    _, _, first = run_m1(cfg, render(make_T(), APPLES3)[0], scene_id=1)
    moved = [apple_on_table(0.26, 0.10, 0.038), APPLES3[1]]   # 첫 사과 1 cm 이동, 세 번째는 집어 감
    _, _, second = run_m1(cfg, render(make_T(), moved)[0], prev=first, scene_id=2)
    assert len(second) == 2
    by_pos = {round(s.center_m[0], 2): s for s in second}
    id0 = min(first, key=lambda s: np.linalg.norm(np.subtract(s.center_m, APPLES3[0][0]))).apple_id
    s0 = by_pos[0.26]
    assert s0.apple_id == id0 and abs(s0.displacement_m - 0.01) < 0.003
    assert len({s.apple_id for s in second}) == 2


def test_hidden_apple_kept_as_occluded_when_arm_over_tray(cfg):
    _, _, first = run_m1(cfg, render(make_T(), APPLES3)[0], scene_id=1)
    _, _, second = run_m1(cfg, render(make_T(), APPLES3[:2])[0], prev=first, scene_id=2, tray_clear=False)
    hidden = [s for s in second if s.occluded]
    assert len(hidden) == 1 and hidden[0].det_id == 0 and hidden[0].scene_id == 2


def test_new_apple_gets_fresh_id(cfg):
    _, _, first = run_m1(cfg, render(make_T(), APPLES3[:2])[0], scene_id=1)
    _, _, second = run_m1(cfg, render(make_T(), APPLES3)[0], prev=first, scene_id=2)
    assert sorted(s.apple_id for s in second) == [1, 2, 3]


def test_scene_mismatch_is_rejected(cfg):
    snap, dets, _ = run_m1(cfg, render(make_T(), APPLES3)[0], scene_id=3)
    with pytest.raises(ValueError):
        pose_estimator.estimate(snap_of(snap.depth_m, scene_id=4), dets, make_T(), None, cfg)


# ================================================================ 상자 점유

@pytest.fixture
def box_cfg(cfg):
    cfg["workcell"]["boxes"]["UPPER"].update(grid=[2, 3], cell_pitch_m=0.06, wall_height_m=0.08,
                                              origin_m=[0.20, 0.08, 0.01])
    return cfg


def cell_center(k, origin=(0.20, 0.08, 0.01), cols=3, pitch=0.06):
    row, col = divmod(k, cols)
    return origin[0] + (col + 0.5) * pitch, origin[1] + (row + 0.5) * pitch


def box_scene(cells, r=0.028, scene_id=1, **kw):
    origin = (0.20, 0.08, 0.01)
    spheres = [((*cell_center(k), origin[2] + r), r) for k in cells]
    depth, _ = render(make_T(), spheres, tops=box_tops(origin, 2, 3, 0.06, 0.08), **kw)
    return snap_of(depth, scene_id=scene_id)


def test_box_grid_heights_and_occupancy(box_cfg):
    boxes = box_occupancy.estimate(box_scene([0, 4], noise_m=0.001), make_T(), box_cfg)
    assert len(boxes) == 1                                   # MIDDLE·REJECT 는 origin 미실측 → 빠짐
    b = boxes[0]
    assert b.box_id == Grade.UPPER and b.valid and (b.rows, b.cols) == (2, 3)
    assert b.occupied == [True, False, False, False, True, False]
    for k, h in enumerate(b.heights_m):
        expect = 0.01 + 0.056 if k in (0, 4) else 0.01
        assert abs(h - expect) < 0.006
    assert math.isclose(b.wall_height_m, 0.09)
    assert abs(b.wall_height_measured_m - 0.09) < 0.004


def test_box_cells_without_depth_are_treated_as_occupied(box_cfg):
    snap = box_scene([])
    T = make_T()
    cx, cy = cell_center(2)
    uv, _ = base_to_pixel(np.array([[cx, cy, 0.01]]), INTR, T)
    u, v = uv[0]
    snap.depth_m[int(v) - 30:int(v) + 30, int(u) - 30:int(u) + 30] = 0.0
    b = box_occupancy.estimate(snap, T, box_cfg)[0]
    assert b.occupied[2] and b.heights_m[2] == pytest.approx(b.wall_height_m)
    assert b.valid                                           # 6칸 중 1칸만 판단 불가


def test_box_out_of_view_is_invalid(box_cfg):
    box_cfg["workcell"]["boxes"]["UPPER"]["origin_m"] = [0.20, 0.60, 0.01]   # 시야 밖
    b = box_occupancy.estimate(box_scene([]), make_T(), box_cfg)[0]
    assert not b.valid


def test_placement_planner_uses_box_measurement(box_cfg):
    boxes = box_occupancy.estimate(box_scene([0, 1, 2]), make_T(), box_cfg)
    apples = pose_estimator.estimate(*_one_apple(box_cfg), box_cfg)
    a = apples[0]
    grade = GradeResult(a.scene_id, a.apple_id, Grade.UPPER, 0.6, 0.55, 0.65, 0.25, 0.15,
                        RejectReason.NONE, True, InspectMode.TOP_SINGLE, "fuji")
    boxes = [type(b)(**{**b.__dict__, "scene_id": a.scene_id}) for b in boxes]
    plan = PlacementPlanner(box_cfg).plan(a, grade, boxes)
    assert plan.feasible and plan.cell_index in (3, 4, 5)
    assert plan.hover_z_m > boxes[0].wall_height_m


def _one_apple(cfg):
    depth, _ = render(make_T(), [apple_on_table(0.30, -0.12, 0.035)])
    snap = snap_of(depth)
    return snap, detector.detect(snap, cfg), make_T(), None


def test_verify_placement(box_cfg):
    before = box_occupancy.estimate(box_scene([], scene_id=1), make_T(), box_cfg)[0]
    after = box_occupancy.estimate(box_scene([3], scene_id=2), make_T(), box_cfg)[0]
    ok, err = box_occupancy.verify_placement(before, after, 3, expected_top_z_m=0.01 + 0.056)
    assert ok and abs(err) < 0.006
    ok, _ = box_occupancy.verify_placement(before, after, 5, expected_top_z_m=0.066)
    assert not ok                                            # 다른 칸은 비어 있다
    bad = type(after)(**{**after.__dict__, "valid": False})
    ok, err = box_occupancy.verify_placement(before, bad, 3, expected_top_z_m=0.066)
    assert not ok and math.isnan(err)


# ================================================================ M1 → M2 연결

def test_m1_output_feeds_inspection_scene():
    """역할 C 의 inspect_scene 이 M1 결과(검출 + AppleState)를 그대로 받아 사과마다 판정한다."""
    from aple.inspection.coloration import params_from_config
    from aple.inspection.pipeline import inspect_scene
    from tests.synthetic import Apple, half_red, render as render_color

    apples = [Apple((-0.08, 0.02, 0.61), 0.038, half_red), Apple((0.09, -0.03, 0.613), 0.035, half_red)]
    color, depth, _ = render_color(apples)
    c = copy.deepcopy(load_config())
    c["workcell"]["T_base_camera"] = make_T().tolist()
    snap = SceneSnapshot(5, Source.TOP, CaptureMode.FULL, color, depth, INTR, 0.0)
    dets = detector.detect(snap, c)
    states = pose_estimator.estimate(snap, dets, make_T(), None, c)
    results = inspect_scene(snap, dets, states, params_from_config(c))
    assert sorted(r.apple_id for r in results) == sorted(s.apple_id for s in states)
    assert all(r.scene_id == 5 for r in results)
