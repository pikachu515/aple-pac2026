"""모드 B 미끄러짐 검사 — 자세로 예측한 사과 중심 vs 깊이로 본 중심 (마할라노비스).

핑거 가림 + 깊이 잡음 + Piper Studio 8bit 깊이 양자화(약 4 mm) 조건에서
미끄러지지 않으면 통과하고, 미끄러지면 잡는지 본다.
"""

from dataclasses import replace

import numpy as np
import pytest

from aple.config import load_config
from aple.inspection.inhand import InHandSlip, apple_center_cam, check_slip, rot_cam_apple
from tests.synthetic import INTR, Apple, render, side_band
from tests.test_inhand import APPLE_IN_TCP, POSES, T_BASE_CAM, tcp_pose
from tests.test_inspection import P
from tests.test_pick_place_pipeline import cfg  # noqa: F401
from tests.test_regrade_pose_sphere import RADIUS_M, finger_mask, held_with_captures

SLIP = load_config()["inspection"]["slip_check"]
STEP_M = (1.2 - 0.15) / 254                     # Piper Studio 깊이 인코딩 기본값 한 칸


def piper_depth(depth: np.ndarray) -> np.ndarray:
    """깊이 1 mm 잡음 후 Piper Studio 8bit 인코딩처럼 칸 단위로 양자화."""
    noisy = depth + np.random.default_rng(1).normal(0, 0.001, depth.shape)
    return (np.floor((noisy - 0.15) / STEP_M) + 0.5) * STEP_M + 0.15


def observe(true_center, R=np.eye(3), **render_kw):
    color, depth, masks = render([Apple(tuple(true_center), RADIUS_M, side_band(0.3), R)], **render_kw)
    depth = np.where(depth > 0, piper_depth(depth), 0.0)
    return depth, finger_mask(masks[0], np.asarray(true_center))


def test_no_slip_passes_every_pose_under_finger_occlusion():
    for R_base_tcp in POSES:
        T_base_tcp = tcp_pose(R_base_tcp)
        predicted = apple_center_cam(T_BASE_CAM, T_base_tcp, APPLE_IN_TCP)
        depth, mask = observe(predicted, rot_cam_apple(T_BASE_CAM, T_base_tcp))
        check = check_slip(depth, mask, INTR, predicted, RADIUS_M, **SLIP)
        assert check.ok, check.mahalanobis
        assert np.linalg.norm(check.observed_center - predicted) < 0.002   # 고정 반지름 피팅은 가림에 강하다
        assert check.mahalanobis < 1.0


def test_slip_is_detected():
    predicted = np.array([0.0, 0.0, 0.45])
    depth, mask = observe(predicted + [0.0, 0.02, 0.0])                     # 핑거 사이로 20 mm 밀림
    check = check_slip(depth, mask, INTR, predicted, RADIUS_M, **SLIP)
    assert not check.ok
    assert check.mahalanobis > SLIP["max_mahalanobis"]


def test_threshold_depends_on_direction():
    """같은 12.5 mm 라도 깊이 방향(원래 불확실)은 허용, 옆 방향(원래 정확)은 미끄러짐."""
    predicted = np.array([0.0, 0.0, 0.45])
    lateral = check_slip(*observe(predicted + [0.0125, 0.0, 0.0]), INTR, predicted, RADIUS_M, **SLIP)
    axial = check_slip(*observe(predicted + [0.0, 0.0, 0.0125]), INTR, predicted, RADIUS_M, **SLIP)
    assert not lateral.ok
    assert axial.ok
    assert lateral.mahalanobis > axial.mahalanobis


def test_missing_depth_cannot_verify():
    predicted = np.array([0.0, 0.0, 0.45])
    depth, mask = observe(predicted, depth_dropout=0.99)
    check = check_slip(depth, mask, INTR, predicted, RADIUS_M, **SLIP)
    assert not check.ok and check.observed_center is None


def test_regrade_stops_at_the_slipped_view(cfg):  # noqa: F811
    cfg["workcell"]["T_base_camera"] = T_BASE_CAM.tolist()
    pipeline, held = held_with_captures(cfg)
    captures = list(held.inspection.captures)
    shifted = captures[2].snapshot.tool_pose.copy()
    shifted[:3, 3] += [0.0, 0.02, 0.0]          # 자세는 20 mm 옮겨졌다고 하는데 사과는 그대로 = 손 안에서 밀림
    captures[2] = replace(captures[2], snapshot=replace(captures[2].snapshot, tool_pose=shifted))
    held = replace(held, inspection=replace(held.inspection, captures=tuple(captures)))

    with pytest.raises(InHandSlip, match="view 2") as info:
        pipeline.regrade_held_apple(held, P)
    assert info.value.view_index == 2
    assert isinstance(info.value, ValueError)


def _slipped_held(cfg):  # noqa: F811
    cfg["workcell"]["T_base_camera"] = T_BASE_CAM.tolist()
    pipeline, held = held_with_captures(cfg)
    captures = list(held.inspection.captures)
    shifted = captures[2].snapshot.tool_pose.copy()
    shifted[:3, 3] += [0.0, 0.02, 0.0]
    captures[2] = replace(captures[2], snapshot=replace(captures[2].snapshot, tool_pose=shifted))
    return pipeline, replace(held, inspection=replace(held.inspection, captures=tuple(captures)))


def test_slip_decision_first_slip_retries_grasp_once(cfg):  # noqa: F811
    from aple.orchestration.recovery import RecoveryAction, RecoveryPolicy
    from aple.schema import FailureCode

    pipeline, held = _slipped_held(cfg)
    out = pipeline.regrade_with_slip_decision(held, P, RecoveryPolicy(cfg))
    assert out.failure_code == FailureCode.SLIP
    assert out.view_index == 2
    assert out.slip is not None and not out.slip.ok
    assert out.decision.action == RecoveryAction.RETRY_GRASP_ONCE
    assert out.held.regraded_grade is None


def test_slip_decision_second_slip_places_reject_and_alerts(cfg):  # noqa: F811
    from aple.orchestration.recovery import RecoveryAction, RecoveryPolicy

    pipeline, held = _slipped_held(cfg)
    out = pipeline.regrade_with_slip_decision(held, P, RecoveryPolicy(cfg), slip_retry_count=1)
    assert out.decision.action == RecoveryAction.PLACE_REJECT_AND_ALERT
    assert out.decision.alert


def test_slip_decision_no_slip_returns_grade(cfg):  # noqa: F811
    from aple.orchestration.recovery import RecoveryPolicy
    from aple.schema import FailureCode

    cfg["workcell"]["T_base_camera"] = T_BASE_CAM.tolist()
    pipeline, held = held_with_captures(cfg)
    out = pipeline.regrade_with_slip_decision(held, P, RecoveryPolicy(cfg))
    assert out.failure_code == FailureCode.NONE
    assert out.decision is None
    assert out.held.regraded_grade is not None


def test_slip_decision_does_not_swallow_other_value_errors(cfg):  # noqa: F811
    from aple.orchestration.recovery import RecoveryPolicy

    pipeline, held = held_with_captures(cfg)
    held = replace(held, inspection=None)
    with pytest.raises(ValueError, match="successful in-hand inspection"):
        pipeline.regrade_with_slip_decision(held, P, RecoveryPolicy(cfg))
