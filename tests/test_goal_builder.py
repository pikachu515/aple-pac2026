"""Coordinate-transform tests for constructing execution TCP targets."""

import math
from types import SimpleNamespace

import pytest

from aple.execution.goal_builder import Pose3D, build_targets, compose, inverse
from aple.schema import AppleState, Grade, PlacementPlan, GraspProfile


def test_target_builder_applies_offsets_and_preserves_apple_center():
    apple = AppleState(
        3, 8, 8, (0.2, -0.1, 0.4), 0.08, 0.4, 0.03, 0.001, 0.9, 0.9
    )
    profile = GraspProfile(3, 8, 1, 0.2, 0.09, 0.02, 3.0, 300, True)
    candidate = SimpleNamespace(
        grasp_pose=Pose3D((0.2, -0.1, 0.35), (0.0, 0.0, 0.0, 1.0)),
        grasp_yaw_rad=math.pi / 2,
        approach_vector=(0.0, 0.0, 2.0),
        approach_height_m=0.1,
    )
    placement = PlacementPlan(
        3, 8, Grade.UPPER, 2, (0.5, 0.3, 0.2), 0.31, 0.28, 0.01, True
    )

    targets = build_targets(apple, profile, candidate, placement)

    assert targets.pick_pose.position_m == pytest.approx((0.2, -0.1, 0.37))
    assert targets.pregrasp_pose.position_m == pytest.approx((0.2, -0.1, 0.47))
    assert targets.apple_in_tcp.position_m == pytest.approx((0.0, 0.0, 0.03))
    assert targets.place_pose.position_m == pytest.approx((0.5, 0.3, 0.17))
    assert targets.hover_pose.position_m == pytest.approx((0.5, 0.3, 0.20))
    assert targets.grasp_yaw_rad == pytest.approx(math.pi / 2)


def test_pose_composition_round_trip_with_rotation():
    half_angle = math.pi / 4
    pose = Pose3D(
        (0.3, -0.2, 0.1),
        (0.0, 0.0, math.sin(half_angle), math.cos(half_angle)),
    )

    identity = compose(pose, inverse(pose))

    assert identity.position_m == pytest.approx((0.0, 0.0, 0.0))
    assert identity.orientation_xyzw == pytest.approx((0.0, 0.0, 0.0, 1.0))
