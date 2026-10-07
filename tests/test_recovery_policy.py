import pytest

from aple.orchestration.recovery import RecoveryAction, RecoveryPolicy, is_lighting_fault
from aple.schema import CaptureMode, FailureCode, LightingStatus, LightState


@pytest.fixture
def policy():
    return RecoveryPolicy(
        {"orchestration": {"max_retry": 2, "max_rescan": 3}}
    )


@pytest.mark.parametrize(
    "failure_code",
    [
        FailureCode.NO_CONTACT,
        FailureCode.IK_FAIL,
        FailureCode.COLLISION,
        FailureCode.SCENE_MISMATCH,
    ],
)
def test_retryable_failures_require_pose_only_rescan(policy, failure_code):
    decision = policy.decide(failure_code=failure_code)

    assert decision.action == RecoveryAction.RESCAN_AND_RETRY
    assert decision.capture_mode == CaptureMode.POSE_ONLY
    assert not decision.alert


def test_retryable_failure_becomes_skip_after_retry_limit(policy):
    decision = policy.decide(failure_code=FailureCode.IK_FAIL, retry_count=2)

    assert decision.action == RecoveryAction.SKIP_AND_ALERT
    assert decision.alert


def test_gate_violation_and_estop_stop_immediately(policy):
    for decision in (
        policy.decide(failure_code=FailureCode.GATE_VIOLATION),
        policy.decide(estop=True),
    ):
        assert decision.action == RecoveryAction.STOP_AND_IDLE
        assert decision.alert


def test_slip_allows_exactly_one_retry(policy):
    assert policy.decide(failure_code=FailureCode.SLIP).action == (
        RecoveryAction.RETRY_GRASP_ONCE
    )
    second = policy.decide(failure_code=FailureCode.SLIP, slip_retry_count=1)
    assert second.action == RecoveryAction.PLACE_REJECT_AND_ALERT
    assert second.alert


def test_slip_with_lost_gripper_safe_stops_without_retry_or_reject_placement(policy):
    for retries in (0, 1):
        decision = policy.decide(
            failure_code=FailureCode.SLIP, slip_retry_count=retries, gripper_lost=True
        )
        assert decision.action == RecoveryAction.SAFE_STOP_AND_ALERT
        assert decision.alert


def test_gripper_lost_must_be_bool(policy):
    with pytest.raises(TypeError, match="gripper_lost"):
        policy.decide(failure_code=FailureCode.SLIP, gripper_lost=1)


def test_drop_requests_full_rescan_and_stops_when_rescan_budget_is_exhausted(policy):
    first = policy.decide(failure_code=FailureCode.DROP)
    exhausted = policy.decide(failure_code=FailureCode.DROP, rescan_count=3)

    assert first.action == RecoveryAction.FULL_RESCAN
    assert first.capture_mode == CaptureMode.FULL
    assert exhausted.action == RecoveryAction.SAFE_STOP_AND_ALERT


def test_unready_scene_rescans_only_within_budget(policy):
    assert policy.decide(scene_ready=False).action == RecoveryAction.RESCAN
    assert policy.decide(scene_ready=False, rescan_count=3).action == (
        RecoveryAction.SKIP_AND_ALERT
    )


def test_lighting_fault_requests_recalibration(policy):
    decision = policy.decide(lighting_fault=True)

    assert decision.action == RecoveryAction.RECALIBRATE_LIGHT
    assert decision.alert


@pytest.mark.parametrize(
    "status, fault",
    [
        (LightingStatus(LightState.OK, 1.0, True), False),
        (LightingStatus(LightState.CORRECTING, 4.0, True), False),
        (LightingStatus(LightState.FAULT, 9.0, True), True),
        (LightingStatus(LightState.FAULT, float("nan"), False), True),
        (LightingStatus(LightState.OK, 1.0, False), True),
    ],
)
def test_lighting_status_maps_to_fault_flag(policy, status, fault):
    assert is_lighting_fault(status) is fault
    expected = RecoveryAction.RECALIBRATE_LIGHT if fault else RecoveryAction.CONTINUE
    assert policy.decide(lighting_fault=is_lighting_fault(status)).action == expected


def test_lighting_fault_helper_rejects_non_status():
    with pytest.raises(TypeError):
        is_lighting_fault(None)


def test_deferred_item_requires_explicit_recheck_capture_mode(policy):
    with pytest.raises(ValueError, match="deferred_capture_mode"):
        policy.decide(deferred=True)

    decision = policy.decide(
        deferred=True, deferred_capture_mode=CaptureMode.FULL
    )
    assert decision.action == RecoveryAction.REEVALUATE_DEFERRED
    assert decision.capture_mode == CaptureMode.FULL


def test_unknown_failure_defaults_to_safe_stop_and_alert(policy):
    decision = policy.decide(failure_code=FailureCode.TIMEOUT)

    assert decision.action == RecoveryAction.SAFE_STOP_AND_ALERT
    assert decision.alert


@pytest.mark.parametrize("count", [-1, True, 1.2])
def test_invalid_counters_are_rejected(policy, count):
    with pytest.raises(ValueError):
        policy.decide(retry_count=count)
