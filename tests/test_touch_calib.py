"""점 터치 간이 캘리브레이션 (aple/hal/calibration.py fit_touch_points, tools/touch_calib.py)."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from aple.config import CONFIG_DIR
from aple.hal.calibration import (base_to_pixel, camera_point_at, fit_touch_points, load_T_base_camera,
                                  rpy_to_matrix, tip_from_flange)
from tests.test_perception import INTR, make_T, render
from tools import touch_calib

# 작업 영역에 흩은 표식 (base_link, m). 마지막 셋은 높이 6 cm 받침대 위.
MARKERS = np.array([[0.15, 0.20, 0.0], [0.42, 0.20, 0.0], [0.15, -0.20, 0.0], [0.42, -0.22, 0.0],
                    [0.30, 0.02, 0.0], [0.22, -0.05, 0.06], [0.36, 0.10, 0.06], [0.28, -0.14, 0.06]])


def scene(T, tmp_path, noise_m=0.0):
    """표식이 놓인 테이블 + 받침대 → 스냅샷 .npz 와 표식 픽셀."""
    tops = [(x - 0.03, x + 0.03, y - 0.03, y + 0.03, z) for x, y, z in MARKERS if z > 0]
    depth, _ = render(T, tops=tops, noise_m=noise_m, seed=5)
    path = tmp_path / "touch.npz"
    np.savez_compressed(path, color=np.full(depth.shape + (3,), 128, np.uint8), depth_m=depth,
                        intrinsics=np.array(INTR), stamp_s=0.0)
    uv, _ = base_to_pixel(MARKERS, INTR, T)
    return path, uv


def run_session(tmp_path, T, base_points, noise_m=0.0, click_err_px=0.0):
    snap_path, uv = scene(T, tmp_path, noise_m)
    uv = uv + np.random.default_rng(2).normal(0, click_err_px, uv.shape) if click_err_px else uv
    session = tmp_path / "touch.json"
    touch_calib.main(["mark", str(snap_path), "--session", str(session),
                      "--pixels", ";".join(f"{u:.2f},{v:.2f}" for u, v in uv)])
    data = json.loads(session.read_text(encoding="utf-8"))
    for p, b in zip(data["points"], base_points):
        p["base_m"] = list(map(float, b))
    session.write_text(json.dumps(data), encoding="utf-8")
    return session


def test_full_session_recovers_camera_pose_and_saves(tmp_path):
    T = make_T(0.06)
    session = run_session(tmp_path, T, MARKERS, noise_m=0.001, click_err_px=0.7)
    workcell = tmp_path / "workcell.yaml"
    shutil.copy(CONFIG_DIR / "workcell.yaml", workcell)
    fit = touch_calib.main(["fit", "--session", str(session), "--save", str(workcell)])
    assert fit.rms_m < 0.003 and fit.warnings == []
    assert np.nanmedian(fit.loo_errors_m) < 0.003
    T_saved = load_T_base_camera(str(workcell))
    probe = np.array([[0.30, 0.0, 0.03], [0.18, 0.18, 0.0]])          # 처음 보는 위치
    cam = (probe - T[:3, 3]) @ T[:3, :3]
    pred = cam @ T_saved[:3, :3].T + T_saved[:3, 3]
    assert np.linalg.norm(pred - probe, axis=1).max() < 0.003


def test_bad_touch_is_flagged_and_can_be_dropped(tmp_path, capsys):
    base = MARKERS.copy()
    base[3] += [0.02, -0.015, 0.0]                                       # P4 를 엉뚱한 곳에 터치
    session = run_session(tmp_path, make_T(), base)
    fit = touch_calib.main(["fit", "--session", str(session)])
    assert any("4번째 점" in w for w in fit.warnings)
    fit2 = touch_calib.main(["fit", "--session", str(session), "--drop", "P4"])
    assert fit2.rms_m < 0.001 and fit2.warnings == []


def test_save_refused_with_warnings_unless_forced(tmp_path):
    base = MARKERS.copy()
    base[3] += [0.02, -0.015, 0.0]
    session = run_session(tmp_path, make_T(), base)
    workcell = tmp_path / "workcell.yaml"
    shutil.copy(CONFIG_DIR / "workcell.yaml", workcell)
    with pytest.raises(SystemExit):
        touch_calib.main(["fit", "--session", str(session), "--save", str(workcell)])
    with pytest.raises(ValueError):
        load_T_base_camera(str(workcell))                                # 아직 null 그대로
    touch_calib.main(["fit", "--session", str(session), "--save", str(workcell), "--force"])
    load_T_base_camera(str(workcell))


def test_holdout_point_reports_independent_error(tmp_path, capsys):
    session = run_session(tmp_path, make_T(), MARKERS)
    touch_calib.main(["fit", "--session", str(session), "--holdout", "P8"])
    out = capsys.readouterr().out
    assert "(검증)" in out and "점 7개" in out


def test_millimetre_entry_mistake_is_caught(tmp_path):
    session = run_session(tmp_path, make_T(), MARKERS * 1000)            # m 대신 mm 를 넣은 실수
    fit = touch_calib.main(["fit", "--session", str(session)])
    assert any("RMS" in w for w in fit.warnings)


def test_few_flat_clustered_points_warn():
    T = make_T()
    base = np.array([[0.30, 0.0, 0.0], [0.33, 0.02, 0.0], [0.31, 0.05, 0.0], [0.28, 0.03, 0.0]])
    cam = (base - T[:3, 3]) @ T[:3, :3]
    w = " ".join(fit_touch_points(cam, base).warnings)
    assert "4개" in w and "좁게" in w and "같은 높이" in w


def test_robot_entry_tip_and_flange_forms():
    assert touch_calib.parse_robot_entry("300, -120.5, 15", [0, 0, 0]) == pytest.approx([0.3, -0.1205, 0.015])
    assert touch_calib.parse_robot_entry("  ", [0, 0, 0]) is None
    # 플랜지가 아래를 향함 (rx = 180°): 플랜지 z 축 = base −z → 팁은 플랜지보다 offset 만큼 아래
    tip = touch_calib.parse_robot_entry("300 0 250 180 0 0", [0.0, 0.0, 0.12])
    assert tip == pytest.approx([0.30, 0.0, 0.13])
    with pytest.raises(ValueError):
        touch_calib.parse_robot_entry("1 2", [0, 0, 0])


def test_robot_command_records_typed_points(tmp_path, monkeypatch):
    session = run_session(tmp_path, make_T(), MARKERS)
    data = json.loads(session.read_text(encoding="utf-8"))
    for p in data["points"]:
        p["base_m"] = None
    session.write_text(json.dumps(data), encoding="utf-8")
    answers = iter(["150 200 0", "oops", "420 200 0", "", "q"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    touch_calib.main(["robot", "--session", str(session)])
    pts = json.loads(session.read_text(encoding="utf-8"))["points"]
    assert pts[0]["base_m"] == pytest.approx([0.15, 0.2, 0.0])
    assert pts[1]["base_m"] == pytest.approx([0.42, 0.2, 0.0])          # 잘못된 입력은 다시 묻는다
    assert pts[2]["base_m"] is None                                      # 빈 줄 = 건너뜀


def test_camera_point_needs_depth():
    depth = np.zeros((480, 640), np.float32)
    with pytest.raises(ValueError, match="유효 깊이"):
        camera_point_at(depth, 100, 100, INTR)
    depth[95:106, 95:106] = 0.6
    p = camera_point_at(depth, 100, 100, INTR)
    assert p[2] == pytest.approx(0.6)


def test_tip_from_flange_matches_rpy():
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(0.0, np.pi / 2, 0.0)                       # 플랜지 z → base +x
    T[:3, 3] = [0.2, 0.0, 0.3]
    assert tip_from_flange(T, [0, 0, 0.1]) == pytest.approx([0.3, 0.0, 0.3])


# ---------------------------------------------------------------- --auto 순기구학 (10/9 설치본)

class _FakeKin:
    """v0.5.12 piper_robot.kinematics 모양 — fk 함수 없이 link_transforms + geometry 만 있다."""

    class _Geom:
        tip, tip_index = "link6", 2

    def geometry(self):
        return self._Geom()

    def link_transforms(self, q, geom):
        q = np.asarray(q)
        out = np.tile(np.eye(4), (q.shape[0], 3, 1, 1))
        out[:, geom.tip_index, :3, 3] = q[:, :3]           # 플랜지 위치 = 앞 세 관절값 (시험용)
        return out


def test_piper_fk_falls_back_to_link_transforms_flange():
    fk = touch_calib._piper_fk(_FakeKin())
    T = fk([0.1, 0.2, 0.3, 0, 0, 0])
    assert T.shape == (4, 4)
    assert T[:3, 3] == pytest.approx([0.1, 0.2, 0.3])


def test_piper_fk_rejects_non_flange_tip():
    kin = _FakeKin()
    kin._Geom = type("G", (), {"tip": "gripper", "tip_index": 2})
    with pytest.raises(SystemExit, match="link6"):
        touch_calib._piper_fk(kin)


def test_piper_fk_with_installed_piper_robot():
    kin = pytest.importorskip("piper_robot.kinematics")
    T = touch_calib._piper_fk(kin)([0.0] * 6)
    assert T.shape == (4, 4) and np.allclose(T[3], [0, 0, 0, 1])
    assert np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-6)


def test_robot_auto_with_iface_reads_joints_without_enabling_config(tmp_path, monkeypatch):
    session_path = tmp_path / "touch.json"
    touch_calib.save_session(session_path, {"snapshot": None, "intrinsics": None, "points": [
        {"name": "P1", "pixel": [10.0, 20.0], "cam_m": [0.0, 0.0, 0.5], "base_m": None}]})
    seen = {}

    class FakeStudio:
        def __init__(self, cfg):
            seen["piper_web"] = cfg["piper_web"]

        def joint_state(self):
            return (0.1, 0.2, 0.3, 0.0, 0.0, 0.0)

    import aple.hal.piper_studio as ps
    monkeypatch.setattr(ps, "PiperStudio", FakeStudio)
    fake_fk = touch_calib._piper_fk(_FakeKin())
    monkeypatch.setattr(touch_calib, "_piper_fk", lambda: fake_fk)
    monkeypatch.setattr("builtins.input", lambda _: "")
    touch_calib.main(["robot", "--session", str(session_path), "--auto", "--iface", "can0"])
    assert seen["piper_web"]["enabled"] is True and seen["piper_web"]["iface"] == "can0"
    base = touch_calib.load_session(session_path)["points"][0]["base_m"]
    assert base[:3] == pytest.approx([0.1, 0.2, 0.3], abs=1e-4)          # tcp_offset_m null → 오프셋 0
    from aple.config import load_config
    assert load_config()["piper_web"]["enabled"] is False                  # 설정 파일은 그대로


# ---------------------------------------------------------------- 피벗 보정 (자로 재지 않는 TCP 오프셋)

def _rot(axis, deg):
    """로드리게스 회전 행렬."""
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    a = np.deg2rad(deg)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def _pivot_poses(t, x, tilts, noise_m=0.0, seed=0):
    """끝점 x 에 고정하고 자세만 바꾼 플랜지 자세들 (툴 아래 방향 기준으로 기울임)."""
    rng = np.random.default_rng(seed)
    down = _rot([1, 0, 0], 180)
    out = []
    for axis, deg in tilts:
        T = np.eye(4)
        T[:3, :3] = _rot(axis, deg) @ down
        T[:3, 3] = x - T[:3, :3] @ t + rng.normal(0, noise_m, 3)
        out.append(T)
    return out


TILTS = [([1, 0, 0], 0), ([1, 0, 0], 25), ([1, 0, 0], -25), ([0, 1, 0], 25), ([0, 1, 0], -25), ([0, 0, 1], 60)]


def test_pivot_recovers_tip_offset():
    t, x = np.array([0.002, -0.001, 0.135]), np.array([0.30, 0.05, 0.02])
    est_t, est_x, res, spread = touch_calib.pivot_fit(_pivot_poses(t, x, TILTS, noise_m=0.0005))
    assert est_t == pytest.approx(t, abs=0.003) and est_x == pytest.approx(x, abs=0.003)
    assert np.median(res) < 0.002 and spread > 40


def test_pivot_needs_four_poses():
    with pytest.raises(ValueError):
        touch_calib.pivot_fit(_pivot_poses(np.zeros(3), np.zeros(3), TILTS[:3]))


def test_pivot_command_saves_offset_and_robot_uses_it(tmp_path, monkeypatch):
    t, x = np.array([0.0, 0.0, 0.14]), np.array([0.30, 0.05, 0.02])
    poses = iter(_pivot_poses(t, x, TILTS))
    session_path = tmp_path / "touch.json"
    touch_calib.save_session(session_path, {"snapshot": None, "intrinsics": None, "points": [
        {"name": "P1", "pixel": [1.0, 2.0], "cam_m": [0.0, 0.0, 0.5], "base_m": None}]})
    monkeypatch.setattr(touch_calib, "_auto_reader", lambda cfg, iface: (lambda q: q, type(
        "S", (), {"joint_state": lambda self: next(poses)})()))
    answers = iter([""] * len(TILTS) + ["q"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))
    workcell = tmp_path / "workcell.yaml"
    shutil.copy(CONFIG_DIR / "workcell.yaml", workcell)
    touch_calib.main(["pivot", "--session", str(session_path), "--iface", "can0", "--save", str(workcell)])
    session = touch_calib.load_session(session_path)
    assert session["tcp_offset_m"] == pytest.approx(t, abs=1e-4)
    import yaml
    saved = yaml.safe_load(workcell.read_text(encoding="utf-8"))["workcell"]["tcp_offset_m"]
    assert saved == pytest.approx(t, abs=1e-4)

    flange = _pivot_poses(t, x, [([1, 0, 0], 0)])[0]                     # robot 단계: 끝점 = 세션 오프셋 반영
    monkeypatch.setattr(touch_calib, "_auto_reader", lambda cfg, iface: (lambda q: flange, type(
        "S", (), {"joint_state": lambda self: None})()))
    monkeypatch.setattr("builtins.input", lambda _: "")
    touch_calib.main(["robot", "--session", str(session_path), "--auto", "--iface", "can0"])
    assert touch_calib.load_session(session_path)["points"][0]["base_m"] == pytest.approx(x, abs=1e-4)


# ---------------------------------------------------------------- 끝점 오프셋을 카메라 보정과 함께 풀기

def _touch_with_flanges(t, T_bc, tilts, seed=0, noise_m=0.0005):
    """끝점이 표식에 닿을 때의 플랜지 자세(점마다 손목이 조금씩 기운다)와 카메라 좌표."""
    rng = np.random.default_rng(seed)
    down = _rot([1, 0, 0], 180)
    pts = [np.array([0.20 + 0.06 * (i % 4), -0.12 + 0.08 * (i // 4), 0.0 if i % 3 else 0.08]) for i in range(len(tilts))]
    cams, flanges = [], []
    for x, (axis, deg) in zip(pts, tilts):
        R = _rot(axis, deg) @ down
        F = np.eye(4); F[:3, :3] = R; F[:3, 3] = x - R @ t
        flanges.append(F)
        cams.append(np.linalg.inv(T_bc)[:3, :3] @ x + np.linalg.inv(T_bc)[:3, 3] + rng.normal(0, noise_m, 3))
    return cams, flanges


def _cam_pose():
    """카메라가 로봇 앞 위에서 약간 기운 채 아래를 본다 (10/9 실기와 비슷)."""
    T = np.eye(4)
    T[:3, :3] = _rot([0, 0, 1], -4) @ _rot([0, 1, 0], 3) @ _rot([1, 0, 0], 180)
    T[:3, 3] = [0.55, -0.15, 0.95]
    return T


SOLVE_TILTS = [([1, 0, 0], 12), ([0, 1, 0], -15), ([1, 1, 0], 20), ([0, 1, 0], 10), ([1, 0, 0], -18),
               ([1, -1, 0], 14), ([0, 1, 0], 22), ([1, 0, 0], 5), ([0, 0, 1], 30)]


def test_fit_with_tcp_recovers_lateral_offset():
    """옆 방향 오프셋은 손목 기울기로 정해진다. 툴 축 길이는 거의 수직으로 찍으면 정해지지 않으므로
    실측값(시작값)을 그대로 쓴다 — 보정과 실행에 같은 값을 쓰면 그 오차는 상쇄된다."""
    t = np.array([0.004, -0.002, 0.143])
    T_bc = _cam_pose()
    cams, flanges = _touch_with_flanges(t, T_bc, SOLVE_TILTS)
    T_est, t_est = touch_calib.fit_with_tcp(cams, flanges, np.array([0, 0, 0.143]))
    assert t_est == pytest.approx(t, abs=0.0015)                             # 표식 0.5 mm 잡음
    assert T_est[:3, 3] == pytest.approx(T_bc[:3, 3], abs=0.002)


def test_fit_session_with_flanges_beats_flange_only():
    t = np.array([0.0, 0.0, 0.143])
    cams, flanges = _touch_with_flanges(t, _cam_pose(), SOLVE_TILTS)
    session = {"points": [{"name": f"P{i + 1}", "pixel": [0, 0], "cam_m": list(c), "flange_T": F.tolist(),
                           "base_m": list(F[:3, 3])} for i, (c, F) in enumerate(zip(cams, flanges))]}
    flange_only, _, _ = touch_calib.fit_session(json.loads(json.dumps(session)))
    assert flange_only.rms_m > 0.02                                          # 10/9 실기와 같은 실패: 손목 기울기 × 14 cm
    fixed, _, _ = touch_calib.fit_session(json.loads(json.dumps(session)), tcp=[0, 0, 0.143])
    assert fixed.rms_m < 0.003                                               # 실측 길이 + 플랜지 자세면 충분
    solved, _, _ = touch_calib.fit_session(session, tcp=[0, 0, 0.143], solve_tcp=True)
    assert solved.rms_m < 0.003 and session["fit_tcp_offset_m"] == pytest.approx(t, abs=0.002)
