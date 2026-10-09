"""ChArUco 보드 검출·자세·깊이 편향·eye-to-hand (aple/hal/charuco.py, tools/charuco_calib.py).

보드 그림을 알려진 자세로 카메라 영상에 그린다 (3배 해상도로 그린 뒤 평균 축소 — 실제 카메라처럼 앨리어싱 없이).
렌더 좌표계(그림 왼쪽 위 원점, x 오른쪽, y 아래, z 보드 안쪽)와 OpenCV 보드 좌표계는 고정 변환만큼 다르므로,
시험은 그 차이와 무관한 양(거리, 기울기, 자세 간 상대 변환, eye-to-hand 결과)으로 비교한다.
"""

from __future__ import annotations

import copy
import json
import re
import shutil

import cv2
import numpy as np
import pytest

from aple.config import CONFIG_DIR, load_config
from aple.hal import charuco
from aple.hal.calibration import load_T_base_camera, rpy_to_matrix
from aple.perception.sim_scene import INTR, camera_looking_down
from tools import charuco_calib

PX_PER_M = 6000.0


@pytest.fixture(scope="module")
def spec():
    return charuco.board_spec(load_config())


@pytest.fixture(scope="module")
def board_img(spec):
    return charuco.board_image(spec, PX_PER_M)


def board_pose(spec, rx, ry, rz, center):
    """보드 중심이 카메라 좌표 center 에 오고 RPY(deg)만큼 돈 T_camera_render."""
    W, H = spec.squares_x * spec.square_m, spec.squares_y * spec.square_m
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*np.radians([rx, ry, rz]))
    T[:3, 3] = np.asarray(center) - T[:3, :3] @ [W / 2, H / 2, 0]
    return T


def render(board_img, T_cam_render, ss: int = 3) -> np.ndarray:
    fx, fy, cx, cy = INTR
    K = np.array([[fx * ss, 0, (cx + 0.5) * ss - 0.5], [0, fy * ss, (cy + 0.5) * ss - 0.5], [0, 0, 1]])
    s = 1 / PX_PER_M
    Hm = K @ np.column_stack([T_cam_render[:3, 0], T_cam_render[:3, 1], T_cam_render[:3, 3]]) @ np.diag([s, s, 1])
    big = cv2.warpPerspective(board_img, Hm, (640 * ss, 480 * ss), flags=cv2.INTER_LINEAR, borderValue=200)
    return cv2.cvtColor(cv2.resize(big, (640, 480), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)


def plane_distance(T):
    return float(abs(T[:3, 2] @ T[:3, 3]))


# ================================================================ 보드 치수

def test_board_size_uses_measured_print_scale(spec):
    """100 mm 눈금이 81 mm 로 인쇄됨 → 칸 28.35 mm, 마커 21.06 mm (실측 칸 ≈ 28, 테두리 ≈ 3.5 mm 와 맞는다)."""
    assert spec.square_m == pytest.approx(0.02835)
    assert spec.marker_m == pytest.approx(0.02106)
    assert (spec.square_m - spec.marker_m) / 2 == pytest.approx(0.0036, abs=2e-4)
    assert (spec.squares_x, spec.squares_y, spec.dictionary) == (7, 5, "DICT_4X4_100")


def test_bad_scale_is_rejected():
    cfg = copy.deepcopy(load_config())
    cfg["charuco"]["print_scale"] = 81                 # 81 mm 를 배율 자리에 넣은 실수
    with pytest.raises(ValueError, match="print_scale"):
        charuco.board_spec(cfg)


def test_even_row_board_needs_new_opencv():
    cfg = copy.deepcopy(load_config())
    cfg["charuco"]["squares_y"] = 6
    s = charuco.board_spec(cfg)
    if charuco._NEW_API:
        charuco.make_board(s)
    else:
        with pytest.raises(RuntimeError, match="4.7"):
            charuco.make_board(s)


# ================================================================ 검출 · 자세

@pytest.mark.parametrize("rpy, center", [((0, 0, 0), (0.0, 0.0, 0.45)), ((20, -15, 30), (0.03, -0.02, 0.5)),
                                         ((-25, 10, -60), (-0.04, 0.03, 0.6))])
def test_pose_distance_and_tilt(spec, board_img, rpy, center):
    T = board_pose(spec, *rpy, center)
    p = charuco.estimate_board_pose(render(board_img, T), INTR, spec)
    assert p is not None and p.n_corners == 24 and p.reproj_rms_px < 1.0
    assert abs(p.distance_m - plane_distance(T)) < 0.002
    true_tilt = np.degrees(np.arccos(abs(T[2, 2])))
    assert abs(p.tilt_deg - true_tilt) < 1.0


def test_relative_pose_between_views(spec, board_img):
    T1, T2 = board_pose(spec, 15, 5, 10, (0.0, 0.01, 0.45)), board_pose(spec, -20, 15, -40, (0.03, -0.02, 0.55))
    P1 = charuco.estimate_board_pose(render(board_img, T1), INTR, spec)
    P2 = charuco.estimate_board_pose(render(board_img, T2), INTR, spec)
    est = P1.T_camera_board @ np.linalg.inv(P2.T_camera_board)
    true = T1 @ np.linalg.inv(T2)
    assert np.linalg.norm(est[:3, 3] - true[:3, 3]) < 0.003
    assert charuco_calib._angle_deg(est[:3, :3], true[:3, :3]) < 0.5


def test_unscaled_board_size_gives_wrong_distance(spec, board_img):
    """설계값(100%)으로 81% 인쇄물을 보면 거리가 1/0.81 배로 틀린다 — print_scale 이 필요한 이유."""
    T = board_pose(spec, 10, -10, 0, (0.0, 0.0, 0.5))
    design = charuco.BoardSpec(spec.dictionary, 7, 5, 0.035, 0.026)
    p = charuco.estimate_board_pose(render(board_img, T), INTR, design)
    assert p.distance_m == pytest.approx(plane_distance(T) / 0.81, rel=0.01)


def test_no_board_returns_none(spec):
    assert charuco.estimate_board_pose(np.full((480, 640, 3), 128, np.uint8), INTR, spec) is None


def test_depth_bias_against_board_plane(spec, board_img):
    T = board_pose(spec, 0, 0, 0, (0.0, 0.0, 0.5))
    p = charuco.estimate_board_pose(render(board_img, T), INTR, spec)
    depth = np.full((480, 640), 0.5 + 0.004, np.float32)          # 깊이가 4 mm 멀게 나오는 카메라
    depth[::7, ::5] = 0.0                                          # 결측 섞기
    err, n = charuco.board_depth_error(depth, p, INTR)
    assert n >= 20 and err == pytest.approx(0.004, abs=0.0015)


# ================================================================ eye-to-hand

def handeye_samples(spec, board_img, *, n=12, seed=0):
    """보드를 그리퍼에 고정하고 자세를 바꿔 찍은 표본. → (표본 목록, 정답 T_base_camera)."""
    rng = np.random.default_rng(seed)
    T_bc = camera_looking_down((0.30, 0.0), 0.65, tilt_rad=0.05)
    T_gR = np.eye(4)
    T_gR[:3, :3] = rpy_to_matrix(np.pi, 0.1, 0.4)                  # 그리퍼 → 보드 (모르는 고정값)
    T_gR[:3, 3] = [0.01, -0.02, 0.09]
    samples = []
    for k in range(n):
        rpy = rng.uniform([-25, -25, -70], [25, 25, 70])
        center = rng.uniform([-0.05, -0.04, 0.38], [0.05, 0.04, 0.5])
        T_cR = board_pose(spec, *rpy, center)
        T_bg = T_bc @ T_cR @ np.linalg.inv(T_gR)
        pose = charuco.estimate_board_pose(render(board_img, T_cR), INTR, spec)
        assert pose is not None
        samples.append({"name": f"S{k + 1}", "T_base_gripper": T_bg.tolist(),
                        "T_camera_board": pose.T_camera_board.tolist(), "n_corners": pose.n_corners,
                        "reproj_px": pose.reproj_rms_px})
    return samples, T_bc


def test_handeye_recovers_camera_pose(spec, board_img):
    samples, T_bc = handeye_samples(spec, board_img)
    r = charuco_calib.handeye_fit(samples, load_config())
    assert np.linalg.norm(r["T"][:3, 3] - T_bc[:3, 3]) < 0.003
    assert charuco_calib._angle_deg(r["T"][:3, :3], T_bc[:3, :3]) < 0.3
    assert r["warnings"] == [] and r["spread_m"].max() < 0.003


def test_handeye_flags_bad_sample(spec, board_img):
    samples, _ = handeye_samples(spec, board_img)
    T = np.asarray(samples[4]["T_base_gripper"])
    T[:3, 3] += [0.012, 0.0, 0.0]                                   # 잘못 받아 적은 로봇 자세
    samples[4]["T_base_gripper"] = T.tolist()
    r = charuco_calib.handeye_fit(samples, load_config())
    assert any(w.startswith("S5:") for w in r["warnings"])
    clean = charuco_calib.handeye_fit([s for s in samples if s["name"] != "S5"], load_config())
    assert clean["warnings"] == []


def test_handeye_warns_on_too_few_or_similar_poses(spec, board_img):
    samples, _ = handeye_samples(spec, board_img, n=4)
    w = " ".join(charuco_calib.handeye_fit(samples, load_config())["warnings"])
    assert "4개" in w


def test_flange_pose_entry():
    T = charuco_calib.flange_pose("300, 0, 250, 180, 0, 90")
    assert T[:3, 3] == pytest.approx([0.3, 0.0, 0.25])
    assert T[:3, 2] == pytest.approx([0, 0, -1], abs=1e-9)          # rx 180° → 플랜지 z 가 아래
    with pytest.raises(ValueError):
        charuco_calib.flange_pose("1 2 3")


# ================================================================ 도구

@pytest.fixture
def cfg_with_T(monkeypatch):
    cfg = copy.deepcopy(load_config())
    cfg["workcell"]["T_base_camera"] = camera_looking_down().tolist()
    monkeypatch.setattr(charuco_calib, "load_config", lambda: cfg)
    return cfg


def save_snapshot(path, color, depth):
    np.savez_compressed(path, color=color, depth_m=depth, intrinsics=np.array(INTR), stamp_s=0.0)


def test_check_command_on_board_lying_on_table(spec, board_img, cfg_with_T, tmp_path, capsys):
    T_cR = board_pose(spec, 0, 0, 0, (0.0, 0.0, 0.65))             # 카메라 0.65 m 아래 테이블 위에 눕힌 보드
    snap = tmp_path / "board.npz"
    save_snapshot(snap, render(board_img, T_cR), np.full((480, 640), 0.652, np.float32))
    pose = charuco_calib.main(["check", str(snap), "--draw", str(tmp_path / "check.png")])
    out = capsys.readouterr().out
    assert pose is not None
    bias_mm = float(re.search(r"깊이 − 보드 평면 = ([+-][\d.]+) mm", out).group(1))
    assert abs(bias_mm - 2.0) < 1.5                                 # 정면·0.65 m·보드 약 180 px → 거리 정밀도 ≈ 1 mm
    assert "보드 중심" in out and "수평 대비 0." in out
    assert (tmp_path / "check.png").exists()


def test_handeye_add_and_fit_saves_to_workcell(spec, board_img, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(charuco_calib, "load_config", lambda: copy.deepcopy(load_config()))
    T_bc = camera_looking_down((0.30, 0.0), 0.65, tilt_rad=0.05)
    session = tmp_path / "he.json"
    # add 는 스냅샷에서 보드를 다시 검출한다 → 자세마다 영상과 플랜지 자세 문자열을 만든다
    rng = np.random.default_rng(3)
    T_gR = np.eye(4)
    T_gR[:3, :3] = rpy_to_matrix(np.pi, 0.1, 0.4)
    T_gR[:3, 3] = [0.01, -0.02, 0.09]
    for k in range(10):
        rpy = rng.uniform([-25, -25, -70], [25, 25, 70])
        center = rng.uniform([-0.05, -0.04, 0.38], [0.05, 0.04, 0.5])
        T_cR = board_pose(spec, *rpy, center)
        T_bg = T_bc @ T_cR @ np.linalg.inv(T_gR)
        snap = tmp_path / f"he_{k}.npz"
        save_snapshot(snap, render(board_img, T_cR), np.zeros((480, 640), np.float32))
        x, y, z = T_bg[:3, 3] * 1000
        sy = -T_bg[2, 0]                                             # R = Rz·Ry·Rx → rx, ry, rz 복원
        ry = np.arcsin(np.clip(sy, -1, 1))
        rx = np.arctan2(T_bg[2, 1], T_bg[2, 2])
        rz = np.arctan2(T_bg[1, 0], T_bg[0, 0])
        pose_txt = f"{x:.3f} {y:.3f} {z:.3f} {np.degrees(rx):.4f} {np.degrees(ry):.4f} {np.degrees(rz):.4f}"
        charuco_calib.main(["handeye", "add", str(snap), "--pose", pose_txt, "--session", str(session)])
    assert len(json.loads(session.read_text(encoding="utf-8"))["samples"]) == 10
    workcell = tmp_path / "workcell.yaml"
    shutil.copy(CONFIG_DIR / "workcell.yaml", workcell)
    r = charuco_calib.main(["handeye", "fit", "--session", str(session), "--save", str(workcell)])
    assert r["warnings"] == []
    T_saved = load_T_base_camera(str(workcell))
    assert np.linalg.norm(T_saved[:3, 3] - T_bc[:3, 3]) < 0.003
