"""Phase-B soft-landing waypoint generation tests."""

import pytest

from aple.execution.goal_builder import MotionTargets, Pose3D
from aple.execution.soft_landing import (
    GripperAction,
    build_soft_landing_plan,
)


def targets():
    pose = lambda z: Pose3D((0.4, 0.2, z), (0.0, 0.0, 0.0, 1.0))
    return MotionTargets(
        pick_pose=pose(0.2),
        pregrasp_pose=pose(0.3),
        apple_in_tcp=pose(0.0),
        place_pose=pose(0.1),
        hover_pose=pose(0.15),
        grasp_yaw_rad=0.0,
    )


def test_soft_landing_has_slow_final_descent_release_and_retreat():
    plan = build_soft_landing_plan(
        targets(),
        final_descent_m=0.03,
        slow_speed_scale=0.1,
        approach_speed_scale=0.5,
    )

    assert plan.slow_descent_start_z_m == pytest.approx(0.12)
    assert [step.pose.position_m[2] for step in plan.steps] == pytest.approx(
        [0.12, 0.1, 0.15]
    )
    assert [step.gripper_action for step in plan.steps] == [
        GripperAction.HOLD,
        GripperAction.RELEASE,
        GripperAction.HOLD,
    ]
    assert plan.steps[1].speed_scale == pytest.approx(0.1)


def test_soft_landing_uses_slow_motion_when_entire_drop_is_short():
    motion_targets = targets()
    motion_targets = MotionTargets(
        motion_targets.pick_pose,
        motion_targets.pregrasp_pose,
        motion_targets.apple_in_tcp,
        motion_targets.place_pose,
        Pose3D((0.4, 0.2, 0.11), (0.0, 0.0, 0.0, 1.0)),
        0.0,
    )

    plan = build_soft_landing_plan(
        motion_targets,
        final_descent_m=0.03,
        slow_speed_scale=0.1,
        approach_speed_scale=0.5,
    )

    assert plan.slow_descent_start_z_m == pytest.approx(0.1)
    assert plan.steps[0].pose.position_m[2] == pytest.approx(0.1)


def test_soft_landing_rejects_hover_below_release():
    motion_targets = targets()
    motion_targets = MotionTargets(
        motion_targets.pick_pose,
        motion_targets.pregrasp_pose,
        motion_targets.apple_in_tcp,
        motion_targets.place_pose,
        Pose3D((0.4, 0.2, 0.09), (0.0, 0.0, 0.0, 1.0)),
        0.0,
    )

    with pytest.raises(ValueError, match="hover pose"):
        build_soft_landing_plan(
            motion_targets,
            final_descent_m=0.03,
            slow_speed_scale=0.1,
            approach_speed_scale=0.5,
        )
