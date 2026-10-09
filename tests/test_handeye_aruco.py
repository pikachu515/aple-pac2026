"""ArUco 보드/마커 eye-to-hand 캘리브레이션 (tools/handeye_aruco.py) — 합성 영상으로 검출·PnP·풀이까지 끝까지."""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import cv2
import numpy as np

from tests.conftest import pristine_copy
import pytest

from aple.config import CONFIG_DIR
from aple.hal.calibration import load_T_base_camera
from tools import handeye_aruco as he

INTR = (606.2, 605.1, 414.7, 259.3)                     # 10/9 실기 D435 컬러 848×480
SIZE = (480, 848)
MARKER_M = 0.06
BOARD = he.Target("board")                               # 기본: 2×2, id 10~13, 35 mm, 간격 8 mm
SINGLE = he.Target("marker", marker_m=MARKER_M, marker_id=0)


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
# 표적: 플랜지에서 툴 축 반대쪽(손목 윗면)으로 4 cm, 옆으로 1.5 cm — 툴이 아래를 볼 때 표적은 위를 본다
T_FM = _T(_rot([1, 0, 0], 180) @ _rot([0, 0, 1], 20), [0.015, 0.0, -0.04])
# 보드는 원점이 좌상 모서리라 보드 중심이 마커와 같은 자리에 오게 옮긴다 (보드 z 는 보드 안쪽 → 뒤집는다)
_W = BOARD.size_m[0]
T_FB = T_FM @ _T(_rot([1, 0, 0], 180), [-_W / 2, _W / 2, 0])


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


def _tile(target: he.Target):
    """→ (표적 영상, 영상 네 모서리의 표적 좌표 4×3). 흰 여백 포함."""
    if target.kind == "board":
        px, margin = 860, 120
        w_px = px - 2 * margin
        s = target.size_m[0] / w_px                                # m/px — 원점 = 보드 좌상, y 아래
        img = target.board().generateImage((px, px), marginSize=margin)
        a, b = -margin * s, (px - margin) * s
        return img, np.array([[a, a, 0], [b, a, 0], [b, b, 0], [a, b, 0]])
    px, border = 400, 100
    img = np.full((px + 2 * border,) * 2, 255, np.uint8)
    img[border:border + px, border:border + px] = cv2.aruco.generateImageMarker(
        cv2.aruco.getPredefinedDictionary(he.DICT), target.marker_id, px)
    h = target.marker_m * (px + 2 * border) / px / 2             # 중심 원점, y 위
    return img, np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]])


def render_target(T_ct: np.ndarray, target: he.Target, noise=2.0, seed=0, hide_id=None) -> np.ndarray:
    """실제 ArUco 영상을 표적 자세대로 원근 변환해 회색 장면에 그린다. hide_id: 그 마커를 회색으로 가린다."""
    tile, corners_obj = _tile(target)
    if hide_id is not None:
        found = he.detect_target(tile, target)
        quads = found[1].reshape(-1, 4, 2) if found else []
        for q, i in zip(quads, found[2] if found else []):
            if i == hide_id:
                cv2.fillConvexPoly(tile, q.astype(np.int32), 128)
    K = he._K(INTR)
    img_pts, _ = cv2.projectPoints(corners_obj, cv2.Rodrigues(T_ct[:3, :3])[0], T_ct[:3, 3], K, None)
    src = np.float32([[0, 0], [tile.shape[1], 0], [tile.shape[1], tile.shape[0]], [0, tile.shape[0]]])
    H = cv2.getPerspectiveTransform(src, img_pts.reshape(4, 2).astype(np.float32))
    warped = cv2.warpPerspective(tile, H, (SIZE[1], SIZE[0]), flags=cv2.INTER_AREA, borderValue=0)
    mask = cv2.warpPerspective(np.full(tile.shape, 255, np.uint8), H, (SIZE[1], SIZE[0]), borderValue=0)
    img = np.full(SIZE, 110, np.uint8)
    img[mask > 0] = warped[mask > 0]
    img = cv2.GaussianBlur(img, (3, 3), 0).astype(np.float64)
    img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def synthetic_session(target=SINGLE, n=12, seed=1):
    T_ft = T_FB if target.kind == "board" else T_FM
    poses = []
    for i, F in enumerate(flange_poses(n, seed)):
        T_ct = np.linalg.inv(T_BC) @ F @ T_ft
        snap = SimpleNamespace(color=render_target(T_ct, target, seed=i), depth_m=None, intrinsics=INTR, stamp_s=0.0)
        pose = he.record_pose(snap, np.zeros(6), lambda q, F=F: F, target, use_depth=False)
        assert pose is not None, f"자세 {i + 1}: 합성 표적을 못 찾았다"
        poses.append(pose)
    return {"target": target.to_dict(), "intrinsics": list(INTR), "poses": poses}


def _pose_error(T_est, T_true):
    return (np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]),
            he.rotation_angle_deg(T_est[:3, :3], T_true[:3, :3]))


def _solve_session(session):
    F, C, keep = he._arrays(session)
    obj, img = he._points(session, keep)
    return he.solve(F, C, obj, img, session["intrinsics"])


# ---------------------------------------------------------------- 마커 하나

def test_detect_and_pnp_on_rendered_marker():
    T_cm = np.linalg.inv(T_BC) @ flange_poses(1)[0] @ T_FM
    corners = he.detect_marker(render_target(T_cm, SINGLE), 0)
    assert corners is not None
    T_est, reproj, scale = he.marker_pose(corners, MARKER_M, INTR)
    assert reproj < 0.5 and scale is None
    pos, rot = _pose_error(T_est, T_cm)
    assert pos < 0.01 and rot < 3.0                               # 50 px 마커 한 장 — 거리 방향이 가장 흔들린다
    assert he.detect_marker(render_target(T_cm, SINGLE), 5) is None   # 다른 id 는 무시


def test_depth_rescales_marker_distance():
    T_cm = np.linalg.inv(T_BC) @ flange_poses(1)[0] @ T_FM
    corners = he.detect_marker(render_target(T_cm, SINGLE), 0)
    T_pnp, _, _ = he.marker_pose(corners, MARKER_M, INTR)
    depth = np.full(SIZE, T_pnp[2, 3] * 1.03)                     # 깊이가 3 % 멀다고 말하면 그만큼 축척
    T_est, _, scale = he.marker_pose(corners, MARKER_M, INTR, depth)
    assert scale == pytest.approx(1.03, rel=1e-6)
    assert T_est[2, 3] == pytest.approx(T_pnp[2, 3] * 1.03, rel=1e-6)
    depth[:] = T_pnp[2, 3] * 1.5                                  # 터무니없는 깊이는 쓰지 않는다
    assert he.marker_pose(corners, MARKER_M, INTR, depth)[2] is None


def test_end_to_end_single_marker():
    session = synthetic_session(SINGLE)
    T_bc, T_fm = _solve_session(session)
    pos, rot = _pose_error(T_bc, T_BC)
    assert pos < 0.003 and rot < 1.0
    F, C, _ = he._arrays(session)
    res, _ = he.residuals(F, C, T_bc, T_fm)
    assert np.median(res) < 0.004
    assert he.rotation_spread_deg(F) >= 30


def test_solve_with_noisy_marker_poses():
    rng = np.random.default_rng(3)
    flanges = flange_poses(15, seed=7)
    cams = []
    for F in flanges:
        T = np.linalg.inv(T_BC) @ F @ T_FM
        cams.append(T @ _T(_rot(rng.normal(size=3), rng.normal(0, 0.5)), rng.normal(0, [0.001, 0.001, 0.003])))
    T_bc, T_fm = he.solve(flanges, cams)
    pos, rot = _pose_error(T_bc, T_BC)
    assert pos < 0.003 and rot < 1.0
    assert np.linalg.norm(T_fm[:3, 3] - T_FM[:3, 3]) < 0.005


# ---------------------------------------------------------------- 보드 (기본)

def test_board_detects_all_markers_and_pose():
    T_cb = np.linalg.inv(T_BC) @ flange_poses(1)[0] @ T_FB
    obj, img, ids = he.detect_target(render_target(T_cb, BOARD), BOARD)
    assert sorted(ids) == [10, 11, 12, 13] and obj.shape == (16, 3) and img.shape == (16, 2)
    T_est, reproj, _ = he.target_pose(obj, img, INTR)
    assert reproj < 1.0                                           # 마커 35 px — 하나(50 px)보다 작다
    pos, rot = _pose_error(T_est, T_cb)
    assert pos < 0.008 and rot < 2.0


def test_board_with_one_marker_hidden_still_works():
    T_cb = np.linalg.inv(T_BC) @ flange_poses(1)[0] @ T_FB
    obj, img, ids = he.detect_target(render_target(T_cb, BOARD, hide_id=12), BOARD)
    assert sorted(ids) == [10, 11, 13] and len(obj) == 12
    pos, _ = _pose_error(he.target_pose(obj, img, INTR)[0], T_cb)
    assert pos < 0.01


def test_board_requires_two_markers():
    board_1 = he.Target("board", cols=1, rows=2, first_id=10)
    T_cb = np.linalg.inv(T_BC) @ flange_poses(1)[0] @ T_FB
    img = render_target(T_cb, BOARD, hide_id=12)                  # 1열 보드(id 10, 11)로 보면 둘 다 보인다
    assert he.detect_target(img, board_1) is not None
    only_one = he.Target("board", cols=1, rows=2, first_id=12)    # id 12(가림)·13 → 하나뿐
    assert he.detect_target(img, only_one) is None


def test_end_to_end_board():
    session = synthetic_session(BOARD)
    assert all(p["n_markers"] == 4 and len(p["obj_pts"]) == 16 for p in session["poses"])
    T_bc, T_fb = _solve_session(session)
    pos, rot = _pose_error(T_bc, T_BC)
    assert pos < 0.002 and rot < 0.5
    assert np.linalg.norm(T_fb[:3, 3] - T_FB[:3, 3]) < 0.003


def test_target_round_trips_and_cli_args():
    assert he.Target.from_dict(BOARD.to_dict()).to_dict() == BOARD.to_dict()
    assert he.Target.from_dict(SINGLE.to_dict()).to_dict() == SINGLE.to_dict()
    assert BOARD.ids == [10, 11, 12, 13] and BOARD.size_m[0] == pytest.approx(0.078)
    with pytest.raises(SystemExit):                               # --board 와 --marker-id 는 같이 못 쓴다
        he.main(["board-pdf", "--board", "2x2", "--marker-id", "0"])


# ---------------------------------------------------------------- 명령

def test_warnings_for_few_poses_and_small_rotation():
    down = _rot([1, 0, 0], 180)
    flanges = [_T(down, [0.3 + 0.02 * i, 0.0, 0.25]) for i in range(5)]   # 손목을 안 기울였다
    warn = he.fit_warnings(len(flanges), he.rotation_spread_deg(flanges), np.zeros(5), T_BC)
    assert any("자세가 5개" in w for w in warn) and any("최대 회전" in w for w in warn)
    assert he.rotation_spread_deg(flanges) == pytest.approx(0, abs=1e-6)


def test_fit_command_saves_workcell(tmp_path):
    session_path = tmp_path / "handeye.json"
    he.save_session(session_path, synthetic_session(BOARD))
    workcell = pristine_copy("workcell.yaml", tmp_path / "workcell.yaml")   # 현장 실측값과 무관하게 보정 전에서
    T_bc, _ = he.main(["fit", "--session", str(session_path), "--save", str(workcell)])
    saved = load_T_base_camera(str(workcell))
    assert np.abs(saved - T_bc).max() < 1e-5
    pos, rot = _pose_error(saved, T_BC)
    assert pos < 0.003 and rot < 1.0
    fit = json.loads(session_path.read_text())["fit"]
    assert fit["warnings"] == [] and len(fit["used"]) == 12


def test_fit_refuses_save_with_warnings(tmp_path):
    session = synthetic_session(SINGLE)
    session["poses"] = session["poses"][:5]
    session_path = tmp_path / "handeye.json"
    he.save_session(session_path, session)
    workcell = pristine_copy("workcell.yaml", tmp_path / "workcell.yaml")   # 현장 실측값과 무관하게 보정 전에서
    with pytest.raises(SystemExit, match="경고"):
        he.main(["fit", "--session", str(session_path), "--save", str(workcell)])
    with pytest.raises(ValueError):
        load_T_base_camera(str(workcell))                         # 저장 안 됨
    he.main(["fit", "--session", str(session_path), "--save", str(workcell), "--force"])
    assert load_T_base_camera(str(workcell)).shape == (4, 4)


def test_board_pdf_is_a4_at_exact_scale(tmp_path):
    out = he.main(["board-pdf", "--out", str(tmp_path / "board.pdf")])
    assert out.exists() and out.read_bytes()[:4] == b"%PDF"
    page = he.board_page(BOARD)
    assert page.size == (2480, 3508)                              # A4 @ 300 dpi
    found = he.detect_target(np.asarray(page), BOARD)
    assert found is not None and sorted(found[2]) == [10, 11, 12, 13]
    side_px = np.linalg.norm(found[1][1] - found[1][0])           # 마커 한 변 35 mm = 413 px
    assert side_px == pytest.approx(35 / 25.4 * 300, abs=2)


def test_auto_capture_new_pose_rule():
    from tools.handeye_aruco import is_new_pose
    A = np.eye(4)
    B = np.eye(4); B[:3, 3] = [0.05, 0, 0]
    C = np.eye(4); C[:3, :3] = np.array([[1, 0, 0], [0, np.cos(0.3), -np.sin(0.3)], [0, np.sin(0.3), np.cos(0.3)]])
    assert is_new_pose(A, [], 0.04, 10)
    assert not is_new_pose(A, [A], 0.04, 10)
    assert is_new_pose(B, [A], 0.04, 10)                                      # 50 mm 이동
    assert is_new_pose(C, [A], 0.04, 10)                                      # 17° 회전


def test_charuco_board_piece_detection_and_duplicate_guard():
    """다른 사람이 만든 ChArUco(7×5칸, 칸 28·마커 21 mm)를 잘라 써도 찾은 마커만으로 좌표가 나온다.
    같은 보드가 두 장 보이면 거부한다 (10/9 실기: 책상에 두 장)."""
    import cv2
    from tools.handeye_aruco import Target, detect_target
    t = Target("charuco", cols=7, rows=5, marker_m=0.021, square_m=0.028)
    img = t.board().generateImage((7 * 60, 5 * 60), marginSize=40)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    obj, pix, ids = detect_target(img, t)
    assert len(ids) == 17 and obj.shape == (68, 3)
    piece = np.full_like(img, 255); piece[:, :220] = img[:, :220]             # 왼쪽 일부만 잘라 붙인 조각
    obj2, _, ids2 = detect_target(piece, t)
    assert 2 <= len(ids2) < 17
    assert np.allclose(obj2, obj[[i for k in ids2 for i in range(4 * ids.index(k), 4 * ids.index(k) + 4)]])
    assert detect_target(np.hstack([img, img]), t) is None                   # 두 장이면 None
