"""In-hand inspection target generation tests."""

import math

import pytest

from aple.execution.goal_builder import Pose3D, compose
from aple.execution.inspect_skill import build_inspection_plan


def test_inspection_views_keep_apple_center_fixed_and_change_orientation():
    apple_in_tcp = Pose3D(
        (0.03, 0.0, 0.02),
        (0.0, 0.0, 0.0, 1.0),
    )
    center = (0.4, 0.2, 0.3)

    plan = build_inspection_plan(
        apple_in_tcp,
        center,
        tilt_rad=math.pi / 2,
        rotation_angles_rad=(0.0, math.pi / 2, math.pi, 3 * math.pi / 2),
        settle_time_s=0.3,
    )

    assert len(plan.target_tool_poses) == 4
    assert plan.settle_time_s == pytest.approx(0.3)
    recovered_apple_poses = [
        compose(tool_pose, apple_in_tcp) for tool_pose in plan.target_tool_poses
    ]
    assert all(pose.position_m == pytest.approx(center) for pose in recovered_apple_poses)
    assert len({tuple(round(v, 5) for v in pose.orientation_xyzw) for pose in recovered_apple_poses}) == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"tilt_rad": float("nan")},
        {"rotation_angles_rad": ()},
        {"settle_time_s": -0.1},
    ],
)
def test_inspection_plan_rejects_invalid_view_settings(overrides):
    settings = {
        "tilt_rad": 0.5,
        "rotation_angles_rad": (0.0,),
        "settle_time_s": 0.1,
    }
    settings.update(overrides)

    with pytest.raises(ValueError):
        build_inspection_plan(
            Pose3D((0.0, 0.0, 0.05), (0.0, 0.0, 0.0, 1.0)),
            (0.1, 0.2, 0.3),
            **settings,
        )
