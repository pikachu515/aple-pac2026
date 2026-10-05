"""모드 B 자세 계산 검증 — 로봇 자세 → 카메라에서 본 사과 방향·중심 → 다면 누적."""

import numpy as np
import pytest

from aple.inspection.coloration import accumulate_views
from aple.inspection.inhand import (apple_center_cam, apple_in_tcp_from_base, expected_coverage, inv,
                                    make_view, rot_cam_apple)
from aple.schema import CaptureMode, SceneSnapshot, Source
from tests.synthetic import INTR, Apple, full_truth, render, rot_x, rot_z, side_band
from tests.test_inspection import P


def T(R, t):
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, t
    return out


# 카메라: base (0.3, 0, 0.65) 에서 수직 아래를 본다 (카메라 x = base x, y = base −y, z = base −z)
T_BASE_CAM = T(np.diag([1.0, -1.0, -1.0]), [0.3, 0.0, 0.65])
APPLE_IN_TCP = np.array([0.0, 0.0, 0.09])        # TCP 에서 앞쪽 90 mm 에 사과 중심
HOLD = np.array([0.3, 0.0, 0.20])               # 검사 위치의 사과 중심 (카메라에서 0.45 m 아래)


def tcp_pose(R_base_tcp):
    """사과 중심이 HOLD 에 오도록 TCP 위치를 정한다."""
    return T(R_base_tcp, HOLD - R_base_tcp @ APPLE_IN_TCP)


DOWN = np.diag([1.0, -1.0, -1.0])               # 툴이 아래를 향함 (집는 자세)
POSES = [DOWN] + [DOWN @ rot_x(np.pi / 2) @ rot_z(k * np.pi / 2) for k in range(4)]   # 수평으로 눕힌 뒤 J6 4회


def test_pose_math_round_trips():
    Tt = tcp_pose(POSES[2])
    c = apple_center_cam(T_BASE_CAM, Tt, APPLE_IN_TCP)
    assert np.allclose(c, [0.0, 0.0, 0.45], atol=1e-9)                       # 카메라 바로 아래 0.45 m
    assert np.allclose(apple_in_tcp_from_base(Tt, HOLD), APPLE_IN_TCP)
    R = rot_cam_apple(T_BASE_CAM, Tt)
    assert np.allclose(R @ R.T, np.eye(3)) and np.isclose(np.linalg.det(R), 1)
    assert np.allclose(inv(inv(Tt)), Tt)


def test_views_built_from_robot_poses_recover_full_surface():
    pattern = side_band(0.3)
    views = []
    for Rb in POSES:
        Tt = tcp_pose(Rb)
        R = rot_cam_apple(T_BASE_CAM, Tt)
        c = apple_center_cam(T_BASE_CAM, Tt, APPLE_IN_TCP)
        color, depth, masks = render([Apple(tuple(c), 0.04, pattern, R)])          # 그 자세에서 실제로 보이는 모습
        snap = SceneSnapshot(1, Source.IN_HAND, CaptureMode.FULL, color, depth, INTR, 0.0)
        views.append(make_view(snap, masks[0], T_BASE_CAM, Tt, APPLE_IN_TCP, 0.04))
    cr = accumulate_views(views, P)
    truth = full_truth(pattern)
    assert cr.coverage > 0.85
    assert cr.ratio_lo <= truth <= cr.ratio_hi
    assert abs(cr.ratio - truth) <= (1 - cr.coverage) + 0.02


def test_expected_coverage_guides_tilt_choice():
    """기울임 각도별 예상 커버리지 — 검사 동작 설계용 (전략 S5-4 표와 일치)."""
    def plan(tilt):
        return [rot_cam_apple(T_BASE_CAM, tcp_pose(DOWN))] + [
            rot_cam_apple(T_BASE_CAM, tcp_pose(DOWN @ rot_x(tilt) @ rot_z(k * np.pi / 2))) for k in range(4)]
    cov = {deg: expected_coverage(plan(np.radians(deg))) for deg in (45, 60, 90)}
    assert cov[45] == pytest.approx(0.59, abs=0.03)
    assert cov[60] == pytest.approx(0.70, abs=0.03)
    assert cov[90] == pytest.approx(0.91, abs=0.03)
