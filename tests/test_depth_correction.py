"""깊이 거리 보정 (aple/hal/depth_correction.py, tools/depth_check.py) — #29 '직경이 작게 나온다'.

10/7 실물 D435i 는 0.74 m 에서 깊이가 45% 짧았다 (회색 카드 130×100 mm 기준). 같은 성질(1/z 에서 배율 + 오프셋)로 합성 깊이를 틀어
보정 전에는 사과가 작게, 보정 후에는 제 크기로 나오는지 본다.
"""

from __future__ import annotations

import copy
import re

import cv2
import numpy as np
import pytest

from aple.config import load_config
from aple.hal.depth_correction import DepthCorrection, card_distance, fit_correction
from aple.perception import detector, pose_estimator
from aple.perception.sim_scene import INTR, SIZE, camera_looking_down
from aple.perception.sim_scene import render as sim_render
from aple.schema import CaptureMode, SceneSnapshot, Source

GAIN, OFFSET = 1.0379, 1.0524          # 10/7 실측 (카드 3거리)


def distort(z_true, gain=GAIN, offset=OFFSET):
    """보정이 틀어진 카메라가 내는 깊이: 1/z_측정 = gain/z_실제 + offset."""
    z = np.asarray(z_true, np.float64)
    out = np.where(z > 0, 1.0 / (gain / np.where(z > 0, z, 1.0) + offset), 0.0)
    return out.astype(np.float32)


def snap_of(depth, color=None):
    color = np.full(SIZE + (3,), 128, np.uint8) if color is None else color
    return SceneSnapshot(1, Source.TOP, CaptureMode.FULL, color, depth, INTR, 0.0)


def test_identity_leaves_snapshot_untouched():
    snap = snap_of(np.full(SIZE, 0.5, np.float32))
    assert DepthCorrection().snapshot(snap) is snap
    assert DepthCorrection.from_cfg(load_config()).identity        # 기본 설정은 보정 안 함


def test_measured_camera_error_matches_field_data():
    """10/7: 카드로 잰 0.30 / 0.43 / 0.74 m 에서 깊이 0.22 / 0.29 / 0.41 m."""
    assert np.allclose(distort([0.296, 0.431, 0.740]), [0.218, 0.290, 0.407], atol=0.004)


def test_fit_recovers_gain_and_offset_and_inverts():
    zt = np.array([0.30, 0.45, 0.65])
    fit = fit_correction(distort(zt), zt)
    assert fit.gain == pytest.approx(GAIN, rel=1e-4) and fit.offset_per_m == pytest.approx(OFFSET, rel=1e-4)
    depth = distort(np.linspace(0.25, 0.9, 50)).reshape(5, 10)
    depth[0, 0] = 0.0                                               # 결측은 결측으로
    out = fit.apply(depth)
    assert out[0, 0] == 0.0
    assert np.allclose(out.ravel()[1:], np.linspace(0.25, 0.9, 50)[1:], rtol=1e-4)


def test_single_distance_fits_scale_only():
    fit = fit_correction([0.40, 0.41], [0.50, 0.51])
    assert fit.offset_per_m == 0.0 and fit.gain == pytest.approx((0.50 / 0.40 + 0.51 / 0.41) / 2)
    with pytest.raises(ValueError):
        fit_correction([0.4], [0.5, 0.6])


def test_card_distance_from_known_card_size():
    """회색 카드(130×100 mm)를 0.6 m 에 눕혀 두면 픽셀 크기로 거리를 잰다 — 깊이는 틀려도 된다."""
    z = 0.6
    color = np.full(SIZE + (3,), (40, 160, 60), np.uint8)              # 초록 배경 (무채색 아님)
    w, h = 130e-3 * INTR[0] / z, 100e-3 * INTR[1] / z
    cx, cy = 300, 220
    cv2.rectangle(color, (int(cx - w / 2), int(cy - h / 2)), (int(cx + w / 2), int(cy + h / 2)), (128, 128, 128), -1)
    cd = card_distance(snap_of(distort(np.full(SIZE, z)), color), (130, 100))
    assert cd is not None
    assert cd.z_true_m == pytest.approx(z, rel=0.02)
    assert cd.z_depth_m == pytest.approx(float(distort(z)), abs=1e-4) and cd.error == pytest.approx(float(distort(z)) / z - 1, abs=0.02)
    assert card_distance(snap_of(np.full(SIZE, z, np.float32)), (130, 100)) is None   # 카드 없음


def test_m1_diameter_is_fixed_by_correction():
    """85 mm 사과를 0.65 m 아래에서: 틀어진 깊이로는 직경이 작게, depth_correction 을 넣으면 제 크기."""
    T = camera_looking_down((0.30, 0.0), 0.65)
    r = 0.0425
    apples = [((0.30, y, r), r) for y in (-0.10, 0.0, 0.10)]
    depth, _, color = sim_render(T, apples)
    snap = snap_of(distort(depth), color)

    def diameters(gain, offset):
        cfg = copy.deepcopy(load_config())
        cfg["workcell"]["T_base_camera"] = T.tolist()
        cfg["workcell"]["tray"]["center_m"] = None                 # 영상 전체
        cfg["live_camera"]["depth_correction"] = {"gain": gain, "offset_per_m": offset}
        dets = detector.detect(snap, cfg)
        return [s.diameter_m for s in pose_estimator.estimate(snap, dets, T, None, cfg)]

    raw = diameters(1.0, 0.0)
    fixed = diameters(GAIN, OFFSET)
    assert len(fixed) == 3
    assert all(d < 0.065 for d in raw)                             # 10/7 실물처럼 작게 나온다
    assert all(abs(d - 2 * r) < 0.003 for d in fixed)


def test_depth_check_tool_fit(tmp_path, capsys):
    from tools import depth_check

    for i, z in enumerate((0.50, 0.70)):
        color = np.full(SIZE + (3,), (40, 160, 60), np.uint8)
        w, h = 130e-3 * INTR[0] / z, 100e-3 * INTR[1] / z
        cv2.rectangle(color, (int(320 - w / 2), int(240 - h / 2)), (int(320 + w / 2), int(240 + h / 2)),
                      (128, 128, 128), -1)
        np.savez_compressed(tmp_path / f"snap_{i:04d}.npz", color=color, depth_m=distort(np.full(SIZE, z)),
                            intrinsics=np.array(INTR), stamp_s=0.0)
    assert depth_check.main([str(tmp_path), "--card-mm", "130,100"]) == 2   # 보정 없이는 불합격
    assert depth_check.main(["fit", str(tmp_path), "--card-mm", "130,100"]) == 0
    out = capsys.readouterr().out
    gain = float(re.search(r"depth_correction.gain ([0-9.]+)", out).group(1))
    offset = float(re.search(r"depth_correction.offset_per_m ([0-9.]+)", out).group(1))
    assert gain == pytest.approx(GAIN, rel=0.05) and offset == pytest.approx(OFFSET, rel=0.05)   # 카드 픽셀 반올림
