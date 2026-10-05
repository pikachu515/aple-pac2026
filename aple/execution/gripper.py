"""Read the calibrated gripper effort selected by the grasp profile."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

from aple.schema import GraspProfile


class GraspStateCode(IntEnum):
    OPEN = 0
    CLOSING = 1
    HOLDING = 2
    SLIP = 3
    RELEASED = 4
    FAULT = 5


@dataclass(frozen=True)
class GripperTelemetry:
    width_m: float
    effort_nm: float
    force_est_n: float


@dataclass(frozen=True)
class GraspFeedback:
    state: GraspStateCode
    width_m: float
    effort_nm: float
    force_est_n: float
    slip_flag: bool


def effort_for(profile: GraspProfile) -> int:
    """Return the profile's calibrated actuator effort."""
    if isinstance(profile.effort, bool) or not isinstance(profile.effort, int):
        raise ValueError("profile.effort must be an integer")
    if not 0 <= profile.effort <= 5000:
        raise ValueError("profile.effort must be between 0 and 5000")
    return profile.effort


class GripperForceController:
    """Estimate force from the role-D calibration and detect contact or slip."""

    def __init__(self, cfg: dict) -> None:
        self._cfg = cfg
        self._contact_width_m: float | None = None
        self._slipping = False

    def reset(self) -> None:
        self._contact_width_m = None
        self._slipping = False

    def observe(
        self,
        profile: GraspProfile,
        telemetry: GripperTelemetry,
    ) -> GraspFeedback:
        if not profile.feasible or profile.grasp_force_n <= 0:
            raise ValueError("a feasible profile with a positive calibrated force is required")
        if not all(math.isfinite(value) for value in (
            telemetry.width_m,
            telemetry.effort_nm,
            telemetry.force_est_n,
        )):
            raise ValueError("gripper telemetry values must be finite")
        if telemetry.width_m < 0 or telemetry.effort_nm < 0 or telemetry.force_est_n < 0:
            raise ValueError("gripper telemetry values must be non-negative")

        force_est_n = telemetry.force_est_n
        bruise_limit_n = self._cfg["size_bins"]["force"].get("f_bruise_n")
        if bruise_limit_n is None:
            raise ValueError("size_bins.force.f_bruise_n must be measured for force monitoring")
        bruise_limit_n = float(bruise_limit_n)
        if not math.isfinite(bruise_limit_n) or bruise_limit_n <= 0:
            raise ValueError("size_bins.force.f_bruise_n must be finite and positive")
        if profile.grasp_force_n >= bruise_limit_n:
            raise ValueError("profile grasp force must be below the bruise limit")
        if force_est_n >= bruise_limit_n:
            raise ValueError("measured gripper force reached the configured bruise limit")
        slip_width_m = float(self._cfg["execution"]["gripper"]["slip_width_m"])
        if not math.isfinite(slip_width_m) or slip_width_m <= 0:
            raise ValueError("execution.gripper.slip_width_m must be finite and positive")

        if self._contact_width_m is None:
            if force_est_n >= profile.grasp_force_n:
                self._contact_width_m = telemetry.width_m
                state = GraspStateCode.HOLDING
            else:
                state = GraspStateCode.CLOSING
        elif (
            self._slipping
            or telemetry.width_m - self._contact_width_m > slip_width_m
            or force_est_n < profile.grasp_force_n
        ):
            self._slipping = True
            state = GraspStateCode.SLIP
        else:
            state = GraspStateCode.HOLDING

        return GraspFeedback(
            state=state,
            width_m=telemetry.width_m,
            effort_nm=telemetry.effort_nm,
            force_est_n=force_est_n,
            slip_flag=self._slipping,
        )
