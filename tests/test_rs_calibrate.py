"""D435 재보정 도구의 계산 부분 (tools/rs_calibrate.py) — 카메라 없이.

IMU 기울기 + 수직 높이 → 픽셀 시선의 바닥 깊이, 평면 평탄도. 장치를 여는 부분은 10/8 실물 D435i 로 확인했다.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from aple.perception.sim_scene import INTR, SIZE, camera_looking_down
from aple.perception.sim_scene import render as sim_render
from aple.schema import CaptureMode, SceneSnapshot, Source
from tools import rs_calibrate


def test_floor_depth_matches_rendered_table_for_tilted_camera():
    """카메라가 기울어도 '아래' 방향과 수직 높이로 화면 어디든 바닥 깊이를 맞힌다 (Tare 기준 거리)."""
    tilt = math.radians(22)
    T = camera_looking_down((0.30, 0.0), 0.73, tilt)
    depth, _, _ = sim_render(T, [])
    down_cam = T[:3, :3].T @ np.array([0.0, 0.0, -1.0])          # base −z (아래)를 카메라 좌표로
    for u, v in ((INTR[2], INTR[3]), (100, 400), (600, 80)):
        assert rs_calibrate.floor_depth(down_cam, 0.73, u, v, INTR) == pytest.approx(float(depth[int(v), int(u)]), rel=2e-3)
    assert rs_calibrate.floor_depth(down_cam, 0.73, INTR[2], INTR[3], INTR) == pytest.approx(0.73 / math.cos(tilt), rel=1e-6)


def test_flatness_of_a_flat_tilted_floor_is_small():
    depth, _, _ = sim_render(camera_looking_down((0.30, 0.0), 0.73, math.radians(20)), [])
    snap = SceneSnapshot(0, Source.TOP, CaptureMode.FULL, np.zeros(SIZE + (3,), np.uint8), depth, INTR, 0.0)
    f = rs_calibrate.flatness(snap)
    assert f["rms_mm"] < 0.5 and abs(f["bow_mm"]) < 0.5 and f["tilt_deg"] == pytest.approx(20, abs=0.5)
