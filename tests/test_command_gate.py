"""Fail-closed command-gate tests without any hardware connection."""

import pytest

from aple.execution.command_gate import (
    CommandGate,
    GripperCommand,
)
from aple.schema import ArmCommand


@pytest.fixture
def cfg():
    return {
        "safety": {
            "joint_limits_rad": [[-1.0, 1.0]] * 6,
            "workspace_aabb_m": [[-0.5, 0.5], [-0.5, 0.5], [0.0, 1.0]],
            "speed_scale_max": 0.5,
            "gripper_effort_max": 1000,
        },
        "piper_web": {"gripper_width_range_m": [0.0, 0.1]},
    }


def arm_command(*, effort=1200, width=0.05):
    return ArmCommand(
        position_rad=(0.0,) * 6,
        gripper_width_m=width,
        gripper_effort=effort,
        cmd_source="scripted",
    )


def test_arm_gate_clamps_speed_and_effort_and_records_safe_command(cfg):
    gate = CommandGate(cfg)

    decision = gate.filter_arm_command(
        arm_command(),
        target_tcp_m=(0.1, 0.2, 0.3),
        speed_scale=0.8,
        estop=False,
        arm_ready=True,
    )

    assert decision.accepted
    assert decision.command.gripper_effort == 1000
    assert decision.speed_scale == 0.5
    assert decision.status.effort_clamped
    assert decision.status.velocity_clamped
    assert gate.last_safe_arm_command == decision.command


@pytest.mark.parametrize(
    ("options", "status_name"),
    [
        ({"estop": True}, "estop"),
        ({"arm_ready": False}, "limit_violation"),
        ({"target_tcp_m": (0.1, 0.2, 1.1)}, "workspace_violation"),
        ({"command": arm_command(effort=1000, width=0.11)}, "limit_violation"),
        ({"command": ArmCommand((1.1,) * 6, 0.05, 1000, "scripted")}, "limit_violation"),
    ],
)
def test_arm_gate_rejects_unsafe_targets_without_replacing_last_safe(
    cfg, options, status_name
):
    gate = CommandGate(cfg)
    accepted = gate.filter_arm_command(
        arm_command(effort=1000),
        target_tcp_m=(0.1, 0.2, 0.3),
        speed_scale=0.4,
        estop=False,
        arm_ready=True,
    )
    previous = accepted.command
    inputs = {
        "command": arm_command(effort=1000),
        "target_tcp_m": (0.1, 0.2, 0.3),
        "speed_scale": 0.4,
        "estop": False,
        "arm_ready": True,
    }
    inputs.update(options)

    rejected = gate.filter_arm_command(**inputs)

    assert not rejected.accepted
    assert rejected.command is None
    assert getattr(rejected.status, status_name)
    assert gate.last_safe_arm_command == previous


def test_gate_requires_measured_safety_limits(cfg):
    cfg["safety"]["workspace_aabb_m"] = None

    with pytest.raises(ValueError, match="workspace_aabb_m"):
        CommandGate(cfg).filter_arm_command(
            arm_command(),
            target_tcp_m=(0.1, 0.2, 0.3),
            speed_scale=0.4,
            estop=False,
            arm_ready=True,
        )


def test_gripper_gate_clamps_effort_and_rejects_out_of_range_width(cfg):
    gate = CommandGate(cfg)
    accepted = gate.filter_gripper_command(
        GripperCommand("close_hold", 0.05, 1500),
        estop=False,
        gripper_ready=True,
    )

    assert accepted.accepted
    assert accepted.command.effort == 1000
    assert accepted.status.effort_clamped

    rejected = gate.filter_gripper_command(
        GripperCommand("open", 0.11, 500),
        estop=False,
        gripper_ready=True,
    )
    assert not rejected.accepted
    assert rejected.status.limit_violation
    assert gate.last_safe_gripper_command == accepted.command
