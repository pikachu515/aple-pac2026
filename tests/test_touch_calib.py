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
