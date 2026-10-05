"""Tests for fail-closed Cartesian-to-joint trajectory planning."""

import pytest

from aple.execution.trajectory import (
    JointTrajectoryPlanner,
    ToolPose,
    TrajectoryPlanningError,
)
from aple.schema import FailureCode


LIMITS = [(-1.0, 1.0)] * 6
IDENTITY = (0.0, 0.0, 0.0, 1.0)


class LinearIK:
    def __init__(self, solution="from-pose"):
        self.calls = []
        self.solution = solution

    def solve(self, pose, seed_rad):
        self.calls.append((pose, seed_rad))
        if self.solution == "no-solution":
            return None
        if self.solution != "from-pose":
            return self.solution
        return (pose.position_m[0], 0.0, 0.0, 0.0, 0.0, 0.0)


class SafeSegments:
    def __init__(self, safe=True):
        self.safe = safe
        self.calls = []

    def is_segment_safe(self, start_rad, end_rad):
        self.calls.append((start_rad, end_rad))
        return self.safe


def pose(x):
    return ToolPose((x, 0.0, 0.2), IDENTITY)


def planner(ik=None, validator=None, *, max_step=0.2, limits=LIMITS):
    return JointTrajectoryPlanner(
        ik or LinearIK(),
        validator or SafeSegments(),
        limits,
        max_joint_step_rad=max_step,
    )


def test_planner_interpolates_each_ik_waypoint_with_bounded_steps():
    ik = LinearIK()
    validator = SafeSegments()

    result = planner(ik, validator, max_step=0.25).plan(
        [pose(0.6), pose(-0.2)], (0.0,) * 6
    )

    assert result.waypoint_end_indices == (2, 6)
    assert len(result.points_rad) == len(validator.calls) == 7
    assert result.points_rad[2][0] == pytest.approx(0.6)
    assert result.points_rad[-1][0] == pytest.approx(-0.2)
    assert max(
        abs(end - start)
        for before, after in validator.calls
        for start, end in zip(before, after)
    ) <= 0.25
    assert ik.calls[1][1][0] == pytest.approx(0.6)


def test_planner_fails_closed_when_ik_has_no_solution():
    with pytest.raises(TrajectoryPlanningError) as error:
        planner(LinearIK(solution="no-solution")).plan([pose(0.0)], (0.0,) * 6)

    assert error.value.failure_code == FailureCode.IK_FAIL


def test_planner_rejects_ik_solution_outside_joint_limits():
    with pytest.raises(TrajectoryPlanningError) as error:
        planner(LinearIK(solution=(1.1, 0.0, 0.0, 0.0, 0.0, 0.0))).plan(
            [pose(0.0)], (0.0,) * 6
        )

    assert error.value.failure_code == FailureCode.IK_FAIL


def test_planner_fails_closed_when_any_segment_is_unsafe():
    validator = SafeSegments(safe=False)

    with pytest.raises(TrajectoryPlanningError) as error:
        planner(validator=validator).plan([pose(0.1)], (0.0,) * 6)

    assert error.value.failure_code == FailureCode.COLLISION
    assert len(validator.calls) == 1


@pytest.mark.parametrize(
    "start",
    [
        (0.0,) * 5,
        (0.0, 0.0, 0.0, 0.0, 0.0, float("nan")),
        (2.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    ],
)
def test_planner_rejects_invalid_or_out_of_range_start(start):
    with pytest.raises((ValueError, TrajectoryPlanningError)):
        planner().plan([pose(0.0)], start)


def test_planner_rejects_non_unit_tool_orientation():
    invalid = ToolPose((0.0, 0.0, 0.2), (0.0, 0.0, 0.0, 2.0))

    with pytest.raises(ValueError, match="unit quaternion"):
        planner().plan([invalid], (0.0,) * 6)


def test_planner_requires_collision_validator_and_valid_limits():
    with pytest.raises(ValueError, match="six"):
        JointTrajectoryPlanner(LinearIK(), SafeSegments(), LIMITS[:5],
                              max_joint_step_rad=0.1)
    with pytest.raises(ValueError, match="positive"):
        planner(max_step=0.0)
