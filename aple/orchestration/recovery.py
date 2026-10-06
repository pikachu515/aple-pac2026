"""Bounded recovery decisions from the team orchestration specification.

This module chooses the next recovery intent only. It never commands the robot,
captures a scene, or retries an operation; those effects stay with the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from aple.schema import CaptureMode, FailureCode


class RecoveryAction(str, Enum):
    CONTINUE = "continue"
    RESCAN = "rescan"
    RESCAN_AND_RETRY = "rescan_and_retry"
    RETRY_GRASP_ONCE = "retry_grasp_once"
    FULL_RESCAN = "full_rescan"
    RECALIBRATE_LIGHT = "recalibrate_light"
    SKIP_AND_ALERT = "skip_and_alert"
    SAFE_STOP_AND_ALERT = "safe_stop_and_alert"
    STOP_AND_IDLE = "stop_and_idle"
    REEVALUATE_DEFERRED = "reevaluate_deferred"


@dataclass(frozen=True)
class RecoveryDecision:
    action: RecoveryAction
    capture_mode: CaptureMode | None = None
    alert: bool = False


class RecoveryPolicy:
    """Map observed conditions to bounded next-step intents."""

    def __init__(self, cfg: dict) -> None:
        orchestration = cfg.get("orchestration")
        if not isinstance(orchestration, dict):
            raise ValueError("orchestration config must be a dict")
        self._max_retry = self._non_negative_count(
            orchestration.get("max_retry"), "orchestration.max_retry"
        )
        self._max_rescan = self._non_negative_count(
            orchestration.get("max_rescan"), "orchestration.max_rescan"
        )

    def decide(
        self,
        *,
        failure_code: FailureCode | None = None,
        retry_count: int = 0,
        rescan_count: int = 0,
        slip_retry_count: int = 0,
        scene_ready: bool = True,
        estop: bool = False,
        lighting_fault: bool = False,
        deferred: bool = False,
        deferred_capture_mode: CaptureMode | None = None,
    ) -> RecoveryDecision:
        retry_count = self._non_negative_count(retry_count, "retry_count")
        rescan_count = self._non_negative_count(rescan_count, "rescan_count")
        slip_retry_count = self._non_negative_count(slip_retry_count, "slip_retry_count")
        for value, name in (
            (scene_ready, "scene_ready"),
            (estop, "estop"),
            (lighting_fault, "lighting_fault"),
            (deferred, "deferred"),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be bool")

        if estop or failure_code == FailureCode.GATE_VIOLATION:
            return RecoveryDecision(RecoveryAction.STOP_AND_IDLE, alert=True)
        if lighting_fault:
            return RecoveryDecision(RecoveryAction.RECALIBRATE_LIGHT, alert=True)
        if not scene_ready:
            if rescan_count < self._max_rescan:
                return RecoveryDecision(RecoveryAction.RESCAN)
            return RecoveryDecision(RecoveryAction.SKIP_AND_ALERT, alert=True)
        if deferred:
            if deferred_capture_mode not in (CaptureMode.FULL, CaptureMode.POSE_ONLY):
                raise ValueError(
                    "deferred_capture_mode must be FULL or POSE_ONLY for deferred work"
                )
            if rescan_count < self._max_rescan:
                return RecoveryDecision(
                    RecoveryAction.REEVALUATE_DEFERRED,
                    capture_mode=deferred_capture_mode,
                )
            return RecoveryDecision(RecoveryAction.SKIP_AND_ALERT, alert=True)

        if failure_code is None or failure_code == FailureCode.NONE:
            return RecoveryDecision(RecoveryAction.CONTINUE)
        if failure_code in (
            FailureCode.NO_CONTACT,
            FailureCode.IK_FAIL,
            FailureCode.COLLISION,
            FailureCode.SCENE_MISMATCH,
        ):
            if retry_count < self._max_retry and rescan_count < self._max_rescan:
                return RecoveryDecision(
                    RecoveryAction.RESCAN_AND_RETRY,
                    capture_mode=CaptureMode.POSE_ONLY,
                )
            return RecoveryDecision(RecoveryAction.SKIP_AND_ALERT, alert=True)
        if failure_code == FailureCode.SLIP:
            if slip_retry_count == 0:
                return RecoveryDecision(RecoveryAction.RETRY_GRASP_ONCE)
            return RecoveryDecision(RecoveryAction.SAFE_STOP_AND_ALERT, alert=True)
        if failure_code == FailureCode.DROP:
            if rescan_count < self._max_rescan:
                return RecoveryDecision(
                    RecoveryAction.FULL_RESCAN,
                    capture_mode=CaptureMode.FULL,
                    alert=True,
                )
            return RecoveryDecision(RecoveryAction.SAFE_STOP_AND_ALERT, alert=True)
        return RecoveryDecision(RecoveryAction.SAFE_STOP_AND_ALERT, alert=True)

    @staticmethod
    def _non_negative_count(value: object, name: str) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        return value
