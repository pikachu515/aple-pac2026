"""Tests for the injected scripted Piper motion adapter."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from aple.config import load_config
from aple.execution.goal_builder import MotionTargets, PlaceTargets, Pose3D
from aple.execution.inspect_skill import InspectionPlan
from aple.execution.motion_executor import MotionExecutor
from aple.execution.soft_landing import (
    GripperAction,
    SoftLandingPlan,
    SoftLandingStep,
)
from aple.execution.trajectory import ToolPose
from aple.hal.piper_motion import PiperTrajectoryMotionAdapter
from aple.inspection.in_hand_capture import InHandCapture
from aple.schema import (
    AppleState,
    CaptureMode,
    FailureCode,
    GraspProfile,
    Grade,
    Phase,
    PlacementPlan,
    SceneSnapshot,
    Source,
)


@dataclass
class FakeIK:
    def solve(self, pose: ToolPose, seed_rad):
        return (pose.position_m[0], 0.0, 0.0, 0.0, 0.0, 0.0)


class SafeSegments:
    def __init__(self, safe=True):
        self.safe = safe
        self.calls = []

    def is_segment_safe(self, start_rad, end_rad):
        self.calls.append((start_rad, end_rad))
        return self.safe


class FakeTransport:
    def __init__(self):
        self.joints = (0.0,) * 6
        self.calls = []
        self.pose = None
        self.gripper_success = True

    def joint_state(self):
        return self.joints

    def move_joint_goal(self, position_rad, *, speed_scale, timeout_s):
        self.calls.append(("move", position_rad, speed_scale, timeout_s))
        self.joints = position_rad
        return True

    def set_gripper_width(self, width_m, effort, *, timeout_s):
        self.calls.append(("gripper", width_m, effort, timeout_s))
        return self.gripper_success

    def tool_pose(self):
        return self.pose

    def capture_in_hand(self, apple_id, view_index, *, timeout_s):
        self.calls.append(("capture", apple_id, view_index, timeout_s))
        return None


def pose(x, z=0.2):
    return Pose3D((x, 0.0, z), (0.0, 0.0, 0.0, 1.0))


def make_adapter(*, safe=True):
    cfg = {
        "safety": {
            "joint_limits_rad": [(-1.0, 1.0)] * 6,
            "speed_scale_max": 0.5,
        },
        "execution": {"trajectory": {"max_joint_step_rad": 0.1}},
    }
    transport = FakeTransport()
    validator = SafeSegments(safe)
    adapter = PiperTrajectoryMotionAdapter(
        transport, cfg, FakeIK(), validator
    )
    return adapter, transport, validator


def test_cartesian_waypoints_preserve_lift_height_and_check_every_segment():
    adapter, transport, validator = make_adapter()
    transport.pose = pose(0)
    adapter._cartesian_step_m = .02
    requested = []

    class RecordingIK(FakeIK):
        def solve(self, target, seed):
            requested.append(target)
            return super().solve(target, seed)

    adapter._planner._ik_solver = RecordingIK()
    adapter._move_poses((pose(.1),), speed_scale=.2, deadline=adapter._deadline(2))
    assert len(requested) == 5
    assert [p.position_m[0] for p in requested] == pytest.approx([.02, .04, .06, .08, .1])
    assert [p.position_m[2] for p in requested] == pytest.approx([.2] * 5)
    assert len(validator.calls) >= 5


@pytest.mark.parametrize('step', [0, -1, float('nan'), True])
def test_invalid_cartesian_step_is_rejected_before_motion(step):
    cfg = {'safety': {'joint_limits_rad': [(-1., 1.)]*6},
           'execution': {'trajectory': {'max_joint_step_rad': .1, 'max_cartesian_step_m': step}}}
    transport = FakeTransport()
    with pytest.raises(ValueError):
        PiperTrajectoryMotionAdapter(transport, cfg, FakeIK(), SafeSegments())
    assert not transport.calls


def test_adapter_moves_through_checked_ik_steps_and_uses_requested_speed():
    adapter, transport, validator = make_adapter()
    target = MotionTargets(
        pick_pose=pose(0.2),
        pregrasp_pose=pose(0.0, 0.3),
        apple_in_tcp=pose(0.0),
        place_pose=pose(0.4),
        hover_pose=pose(0.4, 0.3),
        grasp_yaw_rad=0.0,
    )
    result = adapter.execute_phase_a(
        AppleState(1, 1, 1, (0.0, 0.0, 0.2), 0.08, 0.2, 0.03, 0.0, 1.0, 1.0),
        GraspProfile(1, 1, 0, 0.2, 0.09, 0.0, 0.0, 1000, True),
        None,
        PlacementPlan(1, 1, Grade.UPPER, 0, (0.3, 0.0, 0.2), 0.3, 0.2, 0.01, True),
        target,
        backend="scripted",
        effort=1000,
        task_text="",
        speed_scale=0.4,
        timeout_s=3.0,
    )

    assert result.success and result.holding and result.hover_reached
    assert validator.calls
    assert all(call[2] == pytest.approx(0.4) for call in transport.calls if call[0] == "move")
    assert [call[1] for call in transport.calls if call[0] == "gripper"] == [0.09, 0.08]


def test_adapter_fails_closed_when_joint_segment_validator_rejects():
    adapter, transport, _ = make_adapter(safe=False)
    target = MotionTargets(
        pose(0.2), pose(0.0, 0.3), pose(0.0), pose(0.4), pose(0.4, 0.3), 0.0
    )

    result = adapter.execute_phase_a(
        AppleState(1, 1, 1, (0.0, 0.0, 0.2), 0.08, 0.2, 0.03, 0.0, 1.0, 1.0),
        GraspProfile(1, 1, 0, 0.2, 0.09, 0.0, 0.0, 1000, True),
        None,
        PlacementPlan(1, 1, Grade.UPPER, 0, (0.3, 0.0, 0.2), 0.3, 0.2, 0.01, True),
        target,
        backend="scripted",
        effort=1000,
        task_text="",
        speed_scale=0.4,
        timeout_s=3.0,
    )

    assert not result.success
    assert result.failure_code == FailureCode.COLLISION
    assert not any(call[0] == "move" for call in transport.calls)


def test_motion_executor_runs_pick_and_place_through_the_trajectory_adapter():
    cfg = load_config()
    cfg["execution"]["gripper"]["control_mode"] = "fixed_effort"
    cfg["safety"]["joint_limits_rad"] = [(-1.0, 1.0)] * 6
    cfg["safety"]["gripper_effort_max"] = 1000
    cfg["execution"]["trajectory"] = {"max_joint_step_rad": 0.1}
    transport = FakeTransport()
    adapter = PiperTrajectoryMotionAdapter(
        transport, cfg, FakeIK(), SafeSegments()
    )
    executor = MotionExecutor(adapter, cfg)
    apple = AppleState(1, 1, 1, (0.0, 0.0, 0.2), 0.08, 0.2, 0.03, 0.0, 1.0, 1.0)
    profile = GraspProfile(1, 1, 0, 0.2, 0.09, 0.0, 0.0, 1000, True)
    candidate = SimpleNamespace(
        scene_id=1,
        apple_id=1,
        grasp_pose=pose(0.0),
        grasp_yaw_rad=0.0,
        approach_vector=(0.0, 0.0, 1.0),
        approach_height_m=0.1,
        min_clearance_m=0.02,
        collision_free=True,
        score=1.0,
    )
    placement = PlacementPlan(
        1, 1, Grade.UPPER, 0, (0.3, 0.0, 0.2), 0.3, 0.2, 0.01, True
    )

    result = executor.pick_place(apple, profile, candidate, placement)

    assert result.success
    assert result.phase_reached == Phase.RETREAT
    assert any(call[0] == "gripper" and call[1] == apple.diameter_m for call in transport.calls)


def test_adapter_requires_measured_joint_limits():
    cfg = {
        "safety": {"joint_limits_rad": None},
        "execution": {"trajectory": {"max_joint_step_rad": 0.1}},
    }

    with pytest.raises(ValueError, match="must be measured"):
        PiperTrajectoryMotionAdapter(FakeTransport(), cfg, FakeIK(), SafeSegments())


def test_phase_b_applies_distinct_approach_and_slow_descent_scales():
    adapter, transport, _ = make_adapter()
    apple = AppleState(1, 1, 1, (0.0, 0.0, 0.2), 0.08, 0.2, 0.03, 0.0, 1.0, 1.0)
    profile = GraspProfile(1, 1, 0, 0.2, 0.09, 0.0, 0.0, 1000, True)
    plan = PlacementPlan(1, 1, Grade.UPPER, 0, (0.2, 0.0, 0.2), 0.3, 0.2, 0.01, True)
    targets = PlaceTargets(pose(0.2), pose(0.2, 0.3), pose(0.0))
    landing = SoftLandingPlan(
        (
            SoftLandingStep(pose(0.2, 0.25), 0.5, GripperAction.HOLD),
            SoftLandingStep(pose(0.2), 0.1, GripperAction.RELEASE),
            SoftLandingStep(pose(0.2, 0.3), 0.5, GripperAction.HOLD),
        ),
        0.25,
    )

    result = adapter.execute_phase_b(
        apple, profile, plan, targets, landing,
        effort=1000, final_descent_m=0.05, slow_speed_scale=0.1, timeout_s=3.0,
    )

    assert result.success and result.phase_reached == Phase.RETREAT
    speeds = [call[2] for call in transport.calls if call[0] == "move"]
    assert 0.1 in speeds and 0.5 in speeds
    assert [call[1] for call in transport.calls if call[0] == "gripper"] == [0.09]


def test_inspection_adapter_returns_each_captured_view_payload():
    adapter, transport, _ = make_adapter()

    def capture_in_hand(apple_id, view_index, *, timeout_s):
        return InHandCapture(
            SceneSnapshot(
                10 + view_index,
                Source.IN_HAND,
                CaptureMode.FULL,
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.ones((2, 2), dtype=np.float32),
                (1.0, 1.0, 0.0, 0.0),
                float(view_index),
                apple_id=apple_id,
                view_index=view_index,
                tool_pose=np.eye(4),
            ),
            np.ones((2, 2), dtype=bool),
            np.eye(3),
        )

    transport.capture_in_hand = capture_in_hand
    result = adapter.inspect_views(
        GraspProfile(7, 3, 0, 0.2, 0.09, 0.0, 0.0, 1000, True),
        InspectionPlan((pose(0.1), pose(0.2)), 0.0),
        effort=1000,
        timeout_s=3.0,
    )

    assert result.success
    assert result.captured_views == 2
    assert [capture.snapshot.scene_id for capture in result.captures] == [10, 11]
    assert [capture.snapshot.apple_id for capture in result.captures] == [3, 3]


def test_builder_wires_speed_aware_piper_transport():
    from aple.hal.piper_motion import build_piper_motion_adapter

    cfg = {
        "safety": {"joint_limits_rad": [(-1.0, 1.0)] * 6, "speed_scale_max": 0.5},
        "execution": {"trajectory": {"max_joint_step_rad": 0.1}},
    }
    studio = SimpleNamespace(joint_state=lambda: (0.0,) * 6)

    adapter = build_piper_motion_adapter(studio, cfg, FakeIK(), SafeSegments())

    assert isinstance(adapter, PiperTrajectoryMotionAdapter)
