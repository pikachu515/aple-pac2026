"""그리퍼 축 둘레 180° 대칭 자세와 탑뷰 가림으로 실행 직전 그리퍼 자세를 고른다 (10/9 실기: 6축 한계로 IK_FAIL,
손목을 돌린 그리퍼가 탑뷰에서 사과를 가림)."""

import math
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from aple.config import load_config
from aple.execution.camera_view import ArmOcclusion
from aple.execution.goal_builder import Pose3D
from aple.execution.motion_executor import MotionExecutor
from aple.execution.pick_place_pipeline import PickPlacePipeline
from aple.execution.testing import RecordingMotionAdapter
from aple.hal.calibration import rpy_to_matrix
from aple.perception.grasp_planner import (flip_about_tool_axis, nearest_tool_flip, quat_matrix, quat_xyzw,
                                           tool_orientation_at)
from aple.schema import AppleState, BoxOccupancy, Grade, GradeResult, InspectMode, RejectReason


def apple():
    return AppleState(7, 1, 1, (0.40, 0.0, 0.03), 0.08, 0.3, 0.03, 0.001, 0.9, 0.9, 0.0)


def grade():
    return GradeResult(7, 1, Grade.UPPER, 0.8, 0.7, 0.9, 0.25, 0.4,
                       RejectReason.NONE, True, InspectMode.TOP_SINGLE, "fuji")


def box():
    return BoxOccupancy(7, Grade.UPPER, 2, 3, 0.09, 0.08, 0.08, (0.2, 0.1, 0.1),
                        [0.10, 0.11, 0.12, 0.10, 0.13, 0.14], [False] * 6, True)


@dataclass
class Candidate:
    scene_id: int
    apple_id: int
    grasp_pose: Pose3D
    grasp_yaw_rad: float
    approach_vector: tuple
    approach_height_m: float
    min_clearance_m: float
    collision_free: bool
    score: float


def down(yaw_deg):
    return quat_xyzw(rpy_to_matrix(0, 0, math.radians(yaw_deg)) @ rpy_to_matrix(math.pi, 0, 0))


def closing_deg(q):
    x = quat_matrix(q)[:, 0]
    return math.degrees(math.atan2(x[1], x[0]))


class WristIK:
    """6축 = 닫힘 방향 각도, ±limit 밖이면 못 푼다 (Piper 6축 한계 흉내)."""

    def __init__(self, limit_deg=115.0):
        self.limit = math.radians(limit_deg)

    def solve(self, pose, seed):
        j6 = math.radians(closing_deg(pose.orientation_xyzw))
        return None if abs(j6) > self.limit else (0.0, 0.0, 0.0, 0.0, 0.0, j6)


@pytest.fixture
def cfg():
    config = load_config()
    config["piper_web"]["gripper_width_range_m"] = [.02, .12]
    config["size_bins"]["force"]["mu"] = 0.5
    config["size_bins"]["force"]["f_bruise_n"] = 30.0
    config["size_bins"]["gripper_calib"]["effort_to_force_table"] = [[0, 0.0], [5000, 50.0]]
    return config


def item_with_yaw(cfg, yaw_deg):
    cand = Candidate(7, 1, Pose3D((0.40, 0.0, 0.03), down(yaw_deg)), math.radians(yaw_deg),
                     (0.0, 0.0, 1.0), 0.08, 0.02, True, 0.5)
    pipeline = PickPlacePipeline(cfg, RecordingMotionAdapter())
    plan = pipeline.plan_scene([apple()], [grade()], [box()], [cand])
    return pipeline, plan.items[0]


def test_flip_keeps_tool_axis_and_reverses_closing_direction():
    q = down(150)
    f = flip_about_tool_axis(q)
    assert quat_matrix(f)[:, 2] == pytest.approx(quat_matrix(q)[:, 2], abs=1e-9)
    assert closing_deg(f) == pytest.approx(-30.0, abs=1e-6)
    assert nearest_tool_flip(q, down(-20)) == pytest.approx(f, abs=1e-9)
    assert nearest_tool_flip(q, down(170)) == pytest.approx(q, abs=1e-9)


def test_unreachable_wrist_angle_uses_flipped_grasp_with_same_finger_line(cfg):
    pipeline, item = item_with_yaw(cfg, 150)               # 6축 150° — 한계 밖, 뒤집으면 −30°
    pipeline._ik_solver = WristIK()
    chosen = pipeline._ik_feasible_candidate(item).grasp_candidate
    assert closing_deg(chosen.grasp_pose.orientation_xyzw) == pytest.approx(-30.0, abs=1e-6)
    assert chosen.grasp_pose.position_m == item.grasp_candidate.grasp_pose.position_m


def test_both_reachable_prefers_less_wrist_turn(cfg):
    pipeline, item = item_with_yaw(cfg, 100)               # 100° 와 −80° 모두 한계 안 — 덜 도는 −80°
    pipeline._ik_solver = WristIK()
    chosen = pipeline._ik_feasible_candidate(item).grasp_candidate
    assert closing_deg(chosen.grasp_pose.orientation_xyzw) == pytest.approx(-80.0, abs=1e-6)


def test_occluding_wrist_angle_is_skipped_for_visible_one(cfg):
    pipeline, item = item_with_yaw(cfg, 0)
    pipeline._ik_solver = WristIK()
    seen = []

    def fraction(joints, center, radius, open_w):
        seen.append(joints[5])
        return 0.9 if abs(joints[5]) < 1e-6 else 0.05     # 반지름 방향(0°)으로 닫으면 먼 핑거가 사과를 가린다
    pipeline._arm_occlusion = SimpleNamespace(fraction=fraction)
    chosen = pipeline._ik_feasible_candidate(item).grasp_candidate
    assert abs(closing_deg(chosen.grasp_pose.orientation_xyzw)) == pytest.approx(15.0, abs=1e-6)


def test_all_occluded_falls_back_to_least_occluded(cfg):
    pipeline, item = item_with_yaw(cfg, 0)
    pipeline._ik_solver = WristIK()
    pipeline._arm_occlusion = SimpleNamespace(
        fraction=lambda joints, *_: 0.5 + 0.1 * abs(math.degrees(joints[5]) - 45) / 45)
    chosen = pipeline._ik_feasible_candidate(item).grasp_candidate
    assert abs(closing_deg(chosen.grasp_pose.orientation_xyzw)) == pytest.approx(45.0, abs=1e-6)


def test_place_orientation_keeps_grasp_side(cfg):
    cfg["grasp_planner"] = {**cfg["grasp_planner"], "tilt_start_m": 0.20, "tilt_full_m": 0.50, "max_tilt_deg": 40}
    executor = MotionExecutor(RecordingMotionAdapter(), cfg)
    plan = SimpleNamespace(place_center_m=(0.30, -0.35, 0.06))
    base = tool_orientation_at((0.30, -0.35), cfg)
    assert executor._place_orientation(plan) == base             # 집기 전에는 기본 규칙 그대로
    executor._grasp_xyzw = flip_about_tool_axis(base)
    q = executor._place_orientation(plan)
    assert quat_matrix(q)[:, 0] @ quat_matrix(executor._grasp_xyzw)[:, 0] > 0.99


class FakeKinematics:
    """링크 하나(원점 근처 점 덩어리) — 관절과 무관하게 고정."""

    def __init__(self, points):
        self._g = SimpleNamespace(names=("base_link",), pts=np.asarray(points, float),
                                  pt_link=np.zeros(len(points), int), tip_index=0)

    def geometry(self):
        return self._g

    def link_transforms(self, q, geom):
        return np.tile(np.eye(4), (len(q), 1, 1, 1))


def test_arm_occlusion_counts_only_bodies_between_camera_and_apple():
    T_base_camera = np.eye(4)
    T_base_camera[:3, :3] = np.diag([1.0, -1.0, -1.0])    # 1 m 위에서 아래를 본다
    T_base_camera[2, 3] = 1.0
    g = np.linspace(-0.06, 0.06, 13)
    slab = [(x, y, 0.5) for x in g for y in g]              # 높이 0.5 m 판 (12 cm 사각)
    occ = ArmOcclusion(T_base_camera, (0.0, 0.0, 0.0), finger_length_m=0.0, finger_thickness_m=0.0,
                       finger_width_m=0.0, kinematics_module=FakeKinematics(slab), stride=1)
    q = (0.0,) * 6
    assert occ.fraction(q, (0.0, 0.0, 0.04), 0.04, 0.09) > 0.95       # 판 바로 아래 사과는 가린다
    assert occ.fraction(q, (0.30, 0.0, 0.04), 0.04, 0.09) == 0.0      # 옆 사과는 보인다
    assert occ.fraction(q, (0.0, 0.0, 0.70), 0.04, 0.09) == 0.0       # 판보다 카메라 쪽 사과는 보인다
