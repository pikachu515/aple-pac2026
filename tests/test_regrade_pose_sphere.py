"""모드 B 재판정에서 구 중심을 로봇 자세로 정하는지 검증 (핑거 가림 대응).

핑거가 사과 옆을 가리면 실루엣으로 맞춘 구는 반지름이 작아지고 커버리지가 부풀려진다.
workcell.T_base_camera 가 있으면 regrade_held_apple 은 tool_pose·apple_in_tcp·직경으로 구를 정한다.
"""

from dataclasses import replace

import numpy as np
import pytest

from aple.execution import pick_place_pipeline as pipeline_module
from aple.execution.goal_builder import Pose3D
from aple.execution.motion_executor import InspectionExecutionResult
from aple.execution.pick_place_pipeline import PickPlacePipeline
from aple.execution.testing import RecordingMotionAdapter
from aple.inspection.in_hand_capture import InHandCapture
from aple.inspection.inhand import apple_center_cam, rot_cam_apple
from aple.schema import CaptureMode, FailureCode, Phase, SceneSnapshot, Source
from tests.synthetic import INTR, Apple, full_truth, render, side_band
from tests.test_inhand import APPLE_IN_TCP, POSES, T_BASE_CAM, tcp_pose
from tests.test_inspection import P
from tests.test_pick_place_pipeline import apple, box, cfg, grade, grasp_candidate  # noqa: F401

RADIUS_M = apple().diameter_m / 2.0
PATTERN = side_band(0.3)
REAL_ACCUMULATE = pipeline_module.accumulate_views


def finger_mask(mask: np.ndarray, center_cam: np.ndarray) -> np.ndarray:
    """화면에서 사과 중심 기준 |x| > 0.55 r 인 양쪽 띠를 핑거가 가린 것으로 본다."""
    cu = INTR[0] * center_cam[0] / center_cam[2] + INTR[2]
    r_px = INTR[0] * RADIUS_M / center_cam[2]
    xs = np.arange(mask.shape[1])[None, :]
    return mask & (np.abs(xs - cu) <= 0.55 * r_px)


def held_with_captures(config):
    pipeline = PickPlacePipeline(config, RecordingMotionAdapter())
    item = pipeline.plan_scene(
        [apple()], [grade(final=False)], [box()], [grasp_candidate()]
    ).items[0]
    held = pipeline.inspect_held_apple(pipeline.pick_for_inspection(item))
    held = replace(
        held, pick=replace(held.pick, apple_in_tcp=Pose3D(tuple(APPLE_IN_TCP), (0.0, 0.0, 0.0, 1.0)))
    )
    captures = []
    for index, R_base_tcp in enumerate(POSES):
        T_base_tcp = tcp_pose(R_base_tcp)
        R = rot_cam_apple(T_BASE_CAM, T_base_tcp)
        center = apple_center_cam(T_BASE_CAM, T_base_tcp, APPLE_IN_TCP)
        color, depth, masks = render([Apple(tuple(center), RADIUS_M, PATTERN, R)])
        snapshot = SceneSnapshot(
            8 + index, Source.IN_HAND, CaptureMode.FULL, color, depth, INTR, float(index),
            apple_id=1, view_index=index, tool_pose=T_base_tcp,
        )
        captures.append(InHandCapture(snapshot, finger_mask(masks[0], center), R))
    inspection = InspectionExecutionResult(
        True, FailureCode.NONE, Phase.INSPECT, len(captures), tuple(captures)
    )
    return pipeline, replace(held, inspection=inspection)


def regrade_recording(config, monkeypatch):
    """실제 accumulate_views 를 돌리면서 넘어간 뷰와 결과를 기록한다."""
    seen = {}

    def recording(views, params, **kwargs):
        seen["views"] = views
        seen["ratio"] = REAL_ACCUMULATE(views, params, **kwargs)
        return seen["ratio"]

    monkeypatch.setattr(pipeline_module, "accumulate_views", recording)
    pipeline, held = held_with_captures(config)
    held = pipeline.regrade_held_apple(held, P)
    return held, seen


def test_regrade_uses_pose_sphere_when_calibrated(cfg, monkeypatch):  # noqa: F811
    cfg["workcell"]["T_base_camera"] = T_BASE_CAM.tolist()
    held, seen = regrade_recording(cfg, monkeypatch)

    for view, R_base_tcp in zip(seen["views"], POSES):
        expected = apple_center_cam(T_BASE_CAM, tcp_pose(R_base_tcp), APPLE_IN_TCP)
        assert view.sphere is not None
        assert np.allclose(view.sphere.center, expected)
        assert view.sphere.radius == pytest.approx(RADIUS_M)
    ratio = seen["ratio"]
    assert ratio.ratio_lo <= full_truth(PATTERN) <= ratio.ratio_hi
    assert held.regraded_grade is not None and held.regraded_grade.scene_id == 8 + len(POSES) - 1


def test_silhouette_fallback_inflates_coverage_under_finger_occlusion(cfg, monkeypatch):  # noqa: F811
    """보정값이 없으면 기존처럼 실루엣으로 구를 맞춘다 — 핑거 가림에서 커버리지가 부풀려진다."""
    assert cfg["workcell"]["T_base_camera"] is None
    _, silhouette = regrade_recording(cfg, monkeypatch)
    assert all(view.sphere is None for view in silhouette["views"])

    cfg["workcell"]["T_base_camera"] = T_BASE_CAM.tolist()
    _, pose = regrade_recording(cfg, monkeypatch)

    assert silhouette["ratio"].coverage > pose["ratio"].coverage + 0.05


def test_regrade_rejects_malformed_calibration(cfg, monkeypatch):  # noqa: F811
    cfg["workcell"]["T_base_camera"] = np.eye(3).tolist()
    with pytest.raises(ValueError, match="T_base_camera must be a finite 4x4"):
        regrade_recording(cfg, monkeypatch)


def test_regrade_rejects_missing_apple_in_tcp(cfg):  # noqa: F811
    cfg["workcell"]["T_base_camera"] = T_BASE_CAM.tolist()
    pipeline, held = held_with_captures(cfg)
    held = replace(held, pick=replace(held.pick, apple_in_tcp=None))
    with pytest.raises(ValueError, match="did not produce apple_in_tcp"):
        pipeline.regrade_held_apple(held, P)


def test_silhouette_fallback_logs_warning(cfg, monkeypatch):  # noqa: F811
    # caplog 대신 로거를 직접 가로챈다 — ROS 등 pytest 플러그인이 로그 수집을 바꿔도 같은 결과
    warnings = []
    monkeypatch.setattr(pipeline_module._LOG, "warning", lambda msg, *a, **k: warnings.append(msg % a if a else msg))
    regrade_recording(cfg, monkeypatch)
    assert any("T_base_camera is not set" in w for w in warnings)
