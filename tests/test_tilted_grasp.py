"""먼 사과는 그리퍼를 바깥으로 기울여 잡고, 놓을 때는 놓는 위치에 맞춰 다시 기울인다 (10/9 실기: 18 cm 그리퍼는
수직으로 수평 ~0.31 m 까지만 닿았다)."""

import dataclasses
import math

import numpy as np
import pytest

from aple.config import load_config
from aple.execution.goal_builder import Pose3D, build_place_targets, compose, inverse
from aple.perception.grasp_planner import plan_grasps, tilt_for, tool_orientation_at
from aple.schema import AppleState

FIELDS = {f.name for f in dataclasses.fields(AppleState)}


def cfg_tilt():
    cfg = load_config()
    cfg["grasp_planner"] = {**cfg["grasp_planner"], "tilt_start_m": 0.30, "tilt_full_m": 0.60, "max_tilt_deg": 32}
    return cfg


def apple(x, y):
    d = {k: None for k in FIELDS}
    d.update(scene_id=1, apple_id=1, center_m=(x, y, 0.03), diameter_m=0.085, occluded=False)
    return AppleState(**d)


def tool_z(q):
    x, y, z, w = q
    return np.array([2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)])


def test_tilt_grows_with_distance_and_points_outward():
    cfg = cfg_tilt()
    p = cfg["grasp_planner"]
    assert tilt_for((0.25, 0.0), p) == 0.0
    assert math.degrees(tilt_for((0.45, 0.0), p)) == pytest.approx(16.0, abs=0.1)
    assert math.degrees(tilt_for((0.70, 0.0), p)) == pytest.approx(32.0)
    g = plan_grasps([apple(0.45, 0.20)], cfg)[0]
    z = tool_z(g.grasp_pose.orientation_xyzw)                     # 그리퍼 축(아래)이 바깥으로 기운다
    radial = np.array([0.45, 0.20]) / np.hypot(0.45, 0.20)
    assert float(z[:2] @ radial) > 0.2 and z[2] < -0.9
    assert np.asarray(g.approach_vector) == pytest.approx(-z, abs=1e-6)  # 축을 따라 위로 접근
    u = np.array([math.cos(g.grasp_yaw_rad), math.sin(g.grasp_yaw_rad)])
    assert abs(float(u @ radial)) < 1e-6                           # 접선 방향으로 닫는다


def test_near_apple_stays_vertical_and_off_by_default():
    g = plan_grasps([apple(0.25, 0.0)], cfg_tilt())[0]
    assert tool_z(g.grasp_pose.orientation_xyzw) == pytest.approx([0, 0, -1], abs=1e-6)
    g0 = plan_grasps([apple(0.55, 0.0)], load_config())[0]       # 시험 기본값: 기울이기 꺼짐
    assert tool_z(g0.grasp_pose.orientation_xyzw) == pytest.approx([0, 0, -1], abs=1e-6)


def test_place_targets_reorient_tcp_but_keep_apple_center():
    cfg = cfg_tilt()
    g = plan_grasps([apple(0.50, 0.0)], cfg)[0]
    pick = Pose3D((0.50, 0.0, 0.03), g.grasp_pose.orientation_xyzw)
    apple_in_tcp = compose(inverse(pick), Pose3D((0.50, 0.0, 0.03), (0, 0, 0, 1)))
    plan = type("P", (), dict(feasible=True, place_center_m=(0.30, -0.40, 0.06), hover_z_m=0.12, release_z_m=0.06))()
    want = tool_orientation_at((0.30, -0.40), cfg)
    t = build_place_targets(apple_in_tcp, plan, want)
    assert np.asarray(t.place_pose.orientation_xyzw) == pytest.approx(np.asarray(want), abs=1e-6) or \
        np.asarray(t.place_pose.orientation_xyzw) == pytest.approx(-np.asarray(want), abs=1e-6)
    center = compose(t.place_pose, apple_in_tcp).position_m       # 사과 중심은 놓는 칸 중심 그대로
    assert center == pytest.approx((0.30, -0.40, 0.06), abs=1e-6)
