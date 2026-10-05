"""Fail-closed validation and clamping for robot and gripper commands."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from aple.schema import ArmCommand


@dataclass(frozen=True)
class GateStatus:
    effort_clamped: bool = False
    velocity_clamped: bool = False
    limit_violation: bool = False
    workspace_violation: bool = False
    estop: bool = False


@dataclass(frozen=True)
class ArmGateDecision:
    accepted: bool
    command: ArmCommand | None
    speed_scale: float | None
    status: GateStatus


@dataclass(frozen=True)
class GripperCommand:
    command: Literal["open", "close_hold", "release"]
    width_m: float
    effort: int


@dataclass(frozen=True)
class GripperGateDecision:
    accepted: bool
    command: GripperCommand | None
    status: GateStatus


class CommandGate:
    """Gate individual commands before an adapter is allowed to publish them."""

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._last_safe_arm_command: ArmCommand | None = None
        self._last_safe_gripper_command: GripperCommand | None = None

    @property
    def last_safe_arm_command(self) -> ArmCommand | None:
        return self._last_safe_arm_command

    @property
    def last_safe_gripper_command(self) -> GripperCommand | None:
        return self._last_safe_gripper_command

    def filter_arm_command(
        self,
        command: ArmCommand,
        *,
        target_tcp_m: tuple[float, float, float],
        speed_scale: float,
        estop: bool,
        arm_ready: bool,
    ) -> ArmGateDecision:
        if estop:
            return ArmGateDecision(False, None, None, GateStatus(estop=True))
        if not arm_ready:
            return ArmGateDecision(False, None, None, GateStatus(limit_violation=True))

        limits = self._joint_limits()
        if len(command.position_rad) != 6 or not all(
            math.isfinite(float(value)) for value in command.position_rad
        ):
            raise ValueError("ArmCommand.position_rad must contain six finite values")
        for value, (lower, upper) in zip(command.position_rad, limits):
            if not lower <= value <= upper:
                return ArmGateDecision(
                    False, None, None, GateStatus(limit_violation=True)
                )
        if not isinstance(command.cmd_source, str) or command.cmd_source not in {
            "scripted", "act", "soft_landing", "inspect", "move_named",
        }:
            raise ValueError("ArmCommand.cmd_source is not a supported command source")

        if not self._inside_workspace(target_tcp_m):
            return ArmGateDecision(
                False, None, None, GateStatus(workspace_violation=True)
            )
        if not self._inside_gripper_range(command.gripper_width_m):
            return ArmGateDecision(
                False, None, None, GateStatus(limit_violation=True)
            )

        max_speed = self._bounded_setting(
            self._cfg["safety"].get("speed_scale_max"),
            "safety.speed_scale_max",
            0.0,
            1.0,
            lower_open=True,
        )
        requested_speed = float(speed_scale)
        if not math.isfinite(requested_speed) or requested_speed <= 0:
            raise ValueError("speed_scale must be finite and positive")
        applied_speed = min(requested_speed, max_speed)

        max_effort = self._max_effort()
        if isinstance(command.gripper_effort, bool) or not isinstance(
            command.gripper_effort, int
        ) or not 0 <= command.gripper_effort <= 5000:
            raise ValueError("ArmCommand.gripper_effort must be an integer from 0 to 5000")
        applied_effort = min(command.gripper_effort, max_effort)
        safe_command = ArmCommand(
            position_rad=command.position_rad,
            gripper_width_m=command.gripper_width_m,
            gripper_effort=applied_effort,
            cmd_source=command.cmd_source,
        )
        self._last_safe_arm_command = safe_command
        return ArmGateDecision(
            True,
            safe_command,
            applied_speed,
            GateStatus(
                effort_clamped=applied_effort != command.gripper_effort,
                velocity_clamped=applied_speed != requested_speed,
            ),
        )

    def filter_gripper_command(
        self,
        command: GripperCommand,
        *,
        estop: bool,
        gripper_ready: bool,
    ) -> GripperGateDecision:
        if estop:
            return GripperGateDecision(False, None, GateStatus(estop=True))
        if not gripper_ready:
            return GripperGateDecision(
                False, None, GateStatus(limit_violation=True)
            )
        if command.command not in {"open", "close_hold", "release"}:
            raise ValueError("gripper command must be open, close_hold, or release")
        if not self._inside_gripper_range(command.width_m):
            return GripperGateDecision(
                False, None, GateStatus(limit_violation=True)
            )
        if isinstance(command.effort, bool) or not isinstance(command.effort, int):
            raise ValueError("gripper effort must be an integer")
        if not 0 <= command.effort <= 5000:
            raise ValueError("gripper effort must be between 0 and 5000")

        max_effort = self._max_effort()
        applied_effort = min(command.effort, max_effort)
        safe_command = GripperCommand(
            command=command.command,
            width_m=command.width_m,
            effort=applied_effort,
        )
        self._last_safe_gripper_command = safe_command
        return GripperGateDecision(
            True,
            safe_command,
            GateStatus(effort_clamped=applied_effort != command.effort),
        )

    def _joint_limits(self) -> tuple[tuple[float, float], ...]:
        configured = self._cfg["safety"].get("joint_limits_rad")
        if configured is None or len(configured) != 6:
            raise ValueError("safety.joint_limits_rad must contain six measured joint limits")
        limits: list[tuple[float, float]] = []
        for index, pair in enumerate(configured):
            if len(pair) != 2:
                raise ValueError(f"safety.joint_limits_rad[{index}] must be a [min, max] pair")
            lower, upper = float(pair[0]), float(pair[1])
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise ValueError(f"safety.joint_limits_rad[{index}] must be finite and increasing")
            limits.append((lower, upper))
        return tuple(limits)

    def _inside_workspace(self, position_m: tuple[float, float, float]) -> bool:
        if len(position_m) != 3 or not all(
            math.isfinite(float(value)) for value in position_m
        ):
            raise ValueError("target_tcp_m must contain three finite coordinates")
        bounds = self._cfg["safety"].get("workspace_aabb_m")
        if bounds is None or len(bounds) != 3:
            raise ValueError("safety.workspace_aabb_m must contain measured XYZ bounds")
        for axis, (value, interval) in enumerate(zip(position_m, bounds)):
            if len(interval) != 2:
                raise ValueError(
                    f"safety.workspace_aabb_m[{axis}] must be a [min, max] pair"
                )
            lower, upper = (float(bound) for bound in interval)
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower >= upper
            ):
                raise ValueError(
                    f"safety.workspace_aabb_m[{axis}] must be finite and increasing"
                )
            if not lower <= value <= upper:
                return False
        return True

    def _max_effort(self) -> int:
        value = self._cfg["safety"].get("gripper_effort_max")
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 5000:
            raise ValueError("safety.gripper_effort_max must be measured and set from 0 to 5000")
        return value

    def _inside_gripper_range(self, width_m: float) -> bool:
        width_range = self._cfg["piper_web"].get("gripper_width_range_m")
        if width_range is None or len(width_range) != 2:
            raise ValueError("piper_web.gripper_width_range_m must be measured and configured")
        lower, upper = (float(value) for value in width_range)
        if (
            not math.isfinite(lower)
            or not math.isfinite(upper)
            or lower < 0
            or lower >= upper
        ):
            raise ValueError("piper_web.gripper_width_range_m must be finite and increasing")
        if not math.isfinite(width_m):
            raise ValueError("gripper width must be finite")
        return lower <= width_m <= upper

    @staticmethod
    def _bounded_setting(
        value: object,
        name: str,
        lower: float,
        upper: float,
        *,
        lower_open: bool = False,
    ) -> float:
        result = float(value)
        too_low = result <= lower if lower_open else result < lower
        if not math.isfinite(result) or too_low or result > upper:
            raise ValueError(f"{name} must be finite and within its configured range")
        return result
