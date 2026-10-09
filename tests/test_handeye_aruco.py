"""ArUco 마커 eye-to-hand 캘리브레이션 (tools/handeye_aruco.py) — 합성 영상으로 검출·PnP·풀이까지 끝까지."""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from aple.config import CONFIG_DIR
from aple.hal.calibration import load_T_base_camera
from tools import handeye_aruco as he

INTR = (606.2, 605.1, 414.7, 259.3)                     # 10/9 실기 D435 컬러 848×480
SIZE = (480, 848)
MARKER_M = 0.06


def _rot(axis, deg):
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    a = np.deg2rad(deg)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


# 카메라: 작업 영역 위 0.95 m 에서 거의 아래를 본다 (10/9 실기처럼 몇 도 기울고 90° 돌아 있다)
T_BC = _T(_rot([0, 0, 1], 95) @ _rot([1, 0, 0], 177) @ _rot([0, 1, 0], -3), [0.34, -0.03, 0.95])
# 마커: 플랜지에서 툴 축 반대쪽(손목 윗면)으로 4 cm, 옆으로 1.5 cm — 툴이 아래를 볼 때 마커는 위를 본다
T_FM = _T(_rot([1, 0, 0], 180) @ _rot([0, 0, 1], 20), [0.015, 0.0, -0.04])


def flange_poses(n=12, seed=1):
    """작업 영역에 흩고 손목을 여러 방향으로 ±35° 기울인 플랜지 자세 (툴 아래 방향 기준)."""
    rng = np.random.default_rng(seed)
    down = _rot([1, 0, 0], 180)
    out = []
    for _ in range(n):
        axis = rng.normal(size=3)
        axis[2] *= 0.3
        R = _rot(axis, rng.uniform(10, 35)) @ _rot([0, 0, 1], rng.uniform(-60, 60)) @ down
        p = np.array([rng.uniform(0.22, 0.45), rng.uniform(-0.18, 0.12), rng.uniform(0.20, 0.32)])
        out.append(_T(R, p))
    return out


def render_marker(T_cm: np.ndarray, marker_id=0, size_m=MARKER_M, noise=2.0, seed=0) -> np.ndarray:
    """실제 ArUco 영상을 마커 자세대로 원근 변환해 회색 장면에 그린다 (흰 여백 포함)."""
    px, border = 400, 100
    tile = np.full((px + 2 * border,) * 2, 255, np.uint8)
    tile[border:border + px, border:border + px] = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(he.DICT), marker_id, px)
    s = size_m * (px + 2 * border) / px / 2                      # 여백까지 포함한 반변
    obj = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
    K = he._K(INTR)
    img_pts, _ = cv2.projectPoints(obj, cv2.Rodrigues(T_cm[:3, :3])[0], T_cm[:3, 3], K, None)
    src = np.float32([[0, 0], [tile.shape[1], 0], [tile.shape[1], tile.shape[0]], [0, tile.shape[0]]])
    H = cv2.getPerspectiveTransform(src, img_pts.reshape(4, 2).astype(np.float32))
    warped = cv2.warpPerspective(tile, H, (SIZE[1], SIZE[0]), flags=cv2.INTER_AREA, borderValue=0)
    mask = cv2.warpPerspective(np.full(tile.shape, 255, np.uint8), H, (SIZE[1], SIZE[0]), borderValue=0)
    img = np.full(SIZE, 110, np.uint8)
    img[mask > 0] = warped[mask > 0]
    img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float64)
    img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def synthetic_session(n=12, seed=1):
    flanges = flange_poses(n, seed)
    poses = []
    for i, F in enumerate(flanges):
        T_cm = np.linalg.inv(T_BC) @ F @ T_FM
        snap = SimpleNamespace(color=render_marker(T_cm, seed=i), depth_m=None, intrinsics=INTR, stamp_s=0.0)
        pose = he.record_pose(snap, np.zeros(6), lambda q, F=F: F, 0, MARKER_M, use_depth=False)
        assert pose is not None, f"자세 {i + 1}: 합성 마커를 못 찾았다"
        poses.append(pose)
    return {"marker_id": 0, "marker_m": MARKER_M, "intrinsics": list(INTR), "poses": poses}


def _pose_error(T_est, T_true):
    return (np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]),
            he.rotation_angle_deg(T_est[:3, :3], T_true[:3, :3]))


def test_detect_and_pnp_on_rendered_marker():
    T_cm = np.linalg.inv(T_BC) @ flange_poses(1)[0] @ T_FM
    corners = he.detect_marker(render_marker(T_cm), 0)
    assert corners is not None
    T_est, reproj, scale = he.marker_pose(corners, MARKER_M, INTR)
    assert reproj < 0.5 and scale is None
    pos, rot = _pose_error(T_est, T_cm)
    assert pos < 0.01 and rot < 3.0                               # 50 px 마커 한 장 — 거리 방향이 가장 흔들린다
    assert he.detect_marker(render_marker(T_cm), 5) is None       # 다른 id 는 무시


def test_depth_rescales_marker_distance():
    T_cm = np.linalg.inv(T_BC) @ flange_poses(1)[0] @ T_FM
    corners = he.detect_marker(render_marker(T_cm), 0)
    depth = np.zeros(SIZE)
    depth[:] = T_cm[2, 3] * 1.03                                  # 깊이가 3 % 멀다고 말하면 그만큼 축척
    T_est, _, scale = he.marker_pose(corners, MARKER_M, INTR, depth)
    T_pnp, _, _ = he.marker_pose(corners, MARKER_M, INTR)
    assert scale == pytest.approx(T_cm[2, 3] * 1.03 / T_pnp[2, 3], rel=1e-6)
    assert T_est[2, 3] == pytest.approx(T_cm[2, 3] * 1.03, rel=1e-3)
    depth[:] = T_cm[2, 3] * 1.5                                   # 터무니없는 깊이는 쓰지 않는다
    assert he.marker_pose(corners, MARKER_M, INTR, depth)[2] is None


def test_end_to_end_recovers_camera_pose():
    session = synthetic_session()
    F, C, _ = he._arrays(session)
    T_bc, T_fm = he.solve(F, C, MARKER_M, [p["corners_px"] for p in session["poses"]], INTR)
    pos, rot = _pose_error(T_bc, T_BC)
    assert pos < 0.003 and rot < 1.0
    res, _ = he.residuals(F, C, T_bc, T_fm)
    assert np.median(res) < 0.004
    assert he.rotation_spread_deg(F) >= 30


def test_solve_with_noisy_marker_poses():
    rng = np.random.default_rng(3)
    flanges = flange_poses(15, seed=7)
    cams = []
    for F in flanges:
        T = np.linalg.inv(T_BC) @ F @ T_FM
        T = T @ _T(_rot(rng.normal(size=3), rng.normal(0, 0.5)), rng.normal(0, [0.001, 0.001, 0.003]))
        cams.append(T)
    T_bc, T_fm = he.solve(flanges, cams, MARKER_M)
    pos, rot = _pose_error(T_bc, T_BC)
    assert pos < 0.003 and rot < 1.0
    assert np.linalg.norm(T_fm[:3, 3] - T_FM[:3, 3]) < 0.005


def test_warnings_for_few_poses_and_small_rotation():
    down = _rot([1, 0, 0], 180)
    flanges = [_T(down, [0.3 + 0.02 * i, 0.0, 0.25]) for i in range(5)]   # 손목을 안 기울였다
    cams = [np.linalg.inv(T_BC) @ F @ T_FM for F in flanges]
    warn = he.fit_warnings(len(flanges), he.rotation_spread_deg(flanges), np.zeros(5), T_BC)
    assert any("자세가 5개" in w for w in warn) and any("최대 회전" in w for w in warn)
    assert he.rotation_spread_deg(flanges) == pytest.approx(0, abs=1e-6)
    assert len(cams) == 5


def test_fit_command_saves_workcell(tmp_path):
    session_path = tmp_path / "handeye.json"
    he.save_session(session_path, synthetic_session())
    workcell = tmp_path / "workcell.yaml"
    shutil.copy(CONFIG_DIR / "workcell.yaml", workcell)
    T_bc, _ = he.main(["fit", "--session", str(session_path), "--save", str(workcell)])
    saved = load_T_base_camera(str(workcell))
    assert np.abs(saved - T_bc).max() < 1e-5
    pos, rot = _pose_error(saved, T_BC)
    assert pos < 0.003 and rot < 1.0
    fit = json.loads(session_path.read_text())["fit"]
    assert fit["warnings"] == [] and len(fit["used"]) == 12


def test_fit_refuses_save_with_warnings(tmp_path):
    session = synthetic_session(n=12)
    session["poses"] = session["poses"][:5]
    session_path = tmp_path / "handeye.json"
    he.save_session(session_path, session)
    workcell = tmp_path / "workcell.yaml"
    shutil.copy(CONFIG_DIR / "workcell.yaml", workcell)
    with pytest.raises(SystemExit, match="경고"):
        he.main(["fit", "--session", str(session_path), "--save", str(workcell)])
    with pytest.raises(ValueError):
        load_T_base_camera(str(workcell))                         # 저장 안 됨
    he.main(["fit", "--session", str(session_path), "--save", str(workcell), "--force"])
    assert load_T_base_camera(str(workcell)).shape == (4, 4)
