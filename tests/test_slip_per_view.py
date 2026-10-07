"""모드 B 뷰마다 즉시 미끄러짐 검사 — 촬영 직후 검사, 확인 불가는 1회 재촬영, 미끄러지면 남은 회전 중단."""

from dataclasses import replace

import numpy as np

from aple.execution.motion_executor import (
    InspectionPhaseResult,
    ViewVerdict,
)
from aple.execution.inspect_skill import InspectionPlan
from aple.execution.pick_place_pipeline import PickPlacePipeline
from aple.execution.testing import RecordingMotionAdapter
from aple.orchestration.recovery import RecoveryAction, RecoveryPolicy
from aple.schema import FailureCode, GraspProfile
from tests.test_inhand import T_BASE_CAM
from tests.test_inspection import P
from tests.test_pick_place_pipeline import apple, box, cfg, grade, grasp_candidate  # noqa: F401
from tests.test_piper_motion import make_adapter, pose
from tests.test_regrade_pose_sphere import held_with_captures


def _planned_item(config):
    pipeline = PickPlacePipeline(config, RecordingMotionAdapter())
    return pipeline.plan_scene(
        [apple()], [grade(final=False)], [box()], [grasp_candidate()]
    ).items[0]


class ScriptedInspectionAdapter(RecordingMotionAdapter):
    """뷰마다 view_check 를 부르고 SLIP 이면 거기서 멈춘다 (실제 HAL 루프와 같은 규칙)."""

    def __init__(self, captures):
        super().__init__()
        self.scripted = tuple(captures)
        self.verdicts = []
        self.view_check_given = None

    def inspect_views(self, profile, inspection_plan, *, effort, timeout_s, view_check=None):
        self.view_check_given = view_check is not None
        done = []
        views = self.scripted[: len(inspection_plan.target_tool_poses)]
        for index, capture in enumerate(views):
            done.append(capture)
            if view_check is None:
                continue
            verdict = view_check(index, capture)
            self.verdicts.append(verdict)
            if verdict != ViewVerdict.OK:
                return InspectionPhaseResult(False, len(done), tuple(done), slip_view_index=index)
        return InspectionPhaseResult(True, len(done), tuple(done))


def _pipeline_with(config, captures):
    adapter = ScriptedInspectionAdapter(captures)
    pipeline = PickPlacePipeline(config, adapter)
    held = pipeline.pick_for_inspection(_planned_item(config))
    held = replace(
        held, pick=replace(held.pick, apple_in_tcp=_apple_in_tcp())
    )
    return adapter, pipeline, held


def _apple_in_tcp():
    from aple.execution.goal_builder import Pose3D
    from tests.test_inhand import APPLE_IN_TCP

    return Pose3D(tuple(APPLE_IN_TCP), (0.0, 0.0, 0.0, 1.0))


def _captures(config):
    config["workcell"]["T_base_camera"] = T_BASE_CAM.tolist()
    _, held = held_with_captures(config)
    # The pose-sphere fixture has five images; the inspection configuration has four views.
    count = len(config["inspection"]["views"]["rotation_angles_rad"])
    return list(held.inspection.captures[:count])


def _shift(capture, offset=(0.0, 0.02, 0.0)):
    shifted = capture.snapshot.tool_pose.copy()
    shifted[:3, 3] += offset
    return replace(capture, snapshot=replace(capture.snapshot, tool_pose=shifted))


def test_inspection_stops_at_the_first_slipped_view(cfg):  # noqa: F811
    captures = _captures(cfg)
    captures[2] = _shift(captures[2])
    adapter, pipeline, held = _pipeline_with(cfg, captures)

    held = pipeline.inspect_held_apple(held)

    assert adapter.verdicts == [ViewVerdict.OK, ViewVerdict.OK, ViewVerdict.SLIP]
    assert held.inspection.slip_view_index == 2
    assert held.inspection.failure_code == FailureCode.SLIP
    assert not held.inspection.success
    assert held.inspection.captured_views == 3          # 남은 회전은 하지 않았다
    assert held.slip is not None and not held.slip.ok and held.slip.observed_center is not None


def test_slip_outcome_comes_from_inspection_without_regrading(cfg):  # noqa: F811
    captures = _captures(cfg)
    captures[1] = _shift(captures[1])
    _, pipeline, held = _pipeline_with(cfg, captures)
    held = pipeline.inspect_held_apple(held)

    first = pipeline.regrade_with_slip_decision(held, P, RecoveryPolicy(cfg))
    assert first.failure_code == FailureCode.SLIP
    assert first.view_index == 1
    assert first.decision.action == RecoveryAction.RETRY_GRASP_ONCE
    assert first.held.regraded_grade is None

    second = pipeline.regrade_with_slip_decision(held, P, RecoveryPolicy(cfg), slip_retry_count=1)
    assert second.decision.action == RecoveryAction.PLACE_REJECT_AND_ALERT


def test_no_slip_inspection_is_unchanged(cfg):  # noqa: F811
    captures = _captures(cfg)
    adapter, pipeline, held = _pipeline_with(cfg, captures)

    held = pipeline.inspect_held_apple(held)

    assert adapter.verdicts == [ViewVerdict.OK] * len(captures)
    assert held.inspection.success and held.inspection.slip_view_index is None
    assert held.slip is None


def test_missing_depth_is_unverified_not_slip(cfg):  # noqa: F811
    captures = _captures(cfg)
    empty = replace(captures[1].snapshot, depth_m=np.zeros_like(captures[1].snapshot.depth_m))
    captures[1] = replace(captures[1], snapshot=empty)
    adapter, pipeline, held = _pipeline_with(cfg, captures)

    pipeline.inspect_held_apple(held)

    assert adapter.verdicts[1] == ViewVerdict.UNVERIFIED


def test_no_view_check_without_calibration(cfg):  # noqa: F811
    assert cfg["workcell"].get("T_base_camera") is None
    adapter, pipeline, held = _pipeline_with(cfg, [])

    pipeline.inspect_held_apple(held)

    assert adapter.view_check_given is False


def _hal_capture_factory(transport, template, log):
    def capture_in_hand(apple_id, view_index, *, timeout_s):
        log.append(view_index)
        return template(view_index)

    transport.capture_in_hand = capture_in_hand


def _run_hal(verdicts):
    """verdicts: 뷰 호출 순서대로 돌려줄 ViewVerdict 목록 (재촬영 호출 포함)."""
    adapter, transport, _ = make_adapter()
    captured = []
    _hal_capture_factory(transport, lambda i: ("capture", i, len(captured)), captured)
    script = iter(verdicts)
    seen = []

    def view_check(index, capture):
        seen.append((index, capture))
        return next(script)

    result = adapter.inspect_views(
        GraspProfile(7, 3, 0, 0.2, 0.09, 0.0, 0.0, 1000, True),
        InspectionPlan((pose(0.1), pose(0.2), pose(0.3)), 0.0),
        effort=1000,
        timeout_s=3.0,
        view_check=view_check,
    )
    return result, captured, seen, transport


def test_hal_slip_stops_remaining_rotations():
    result, captured, seen, transport = _run_hal([ViewVerdict.OK, ViewVerdict.SLIP])

    assert not result.success
    assert result.slip_view_index == 1
    assert result.captured_views == 2
    assert captured == [0, 1]                            # 3번째 뷰는 촬영도 이동도 하지 않는다
    _, _, _, full_transport = _run_hal([ViewVerdict.OK] * 3)
    moves = lambda t: len([c for c in t.calls if c[0] == "move"])  # noqa: E731
    assert moves(transport) < moves(full_transport)


def test_hal_retakes_an_unverified_view_once_then_continues():
    result, captured, seen, _ = _run_hal(
        [ViewVerdict.OK, ViewVerdict.UNVERIFIED, ViewVerdict.OK, ViewVerdict.OK]
    )

    assert result.success and result.slip_view_index is None
    assert result.captured_views == 3
    assert captured == [0, 1, 1, 2]                      # 뷰 1만 한 번 더 찍었다


def test_hal_unverified_twice_is_treated_as_slip():
    result, captured, _, _ = _run_hal(
        [ViewVerdict.OK, ViewVerdict.UNVERIFIED, ViewVerdict.UNVERIFIED]
    )

    assert not result.success
    assert result.slip_view_index == 1
    assert captured == [0, 1, 1]                         # 재촬영은 1회뿐


def test_hal_without_view_check_behaves_as_before():
    adapter, transport, _ = make_adapter()
    transport.capture_in_hand = lambda apple_id, view_index, *, timeout_s: None
    result = adapter.inspect_views(
        GraspProfile(7, 3, 0, 0.2, 0.09, 0.0, 0.0, 1000, True),
        InspectionPlan((pose(0.1),), 0.0),
        effort=1000,
        timeout_s=3.0,
    )
    assert not result.success and result.slip_view_index is None


def _run_width_watch(widths, *, enabled=True):
    """widths: 검사 시작 기준 폭 + 뷰마다 촬영 직후 읽는 폭."""
    adapter, transport, _ = make_adapter()
    adapter._cfg["execution"].update(
        {
            "recovery": {"width_watch": enabled, "lost_grip_width_m": 0.01},
            "gripper": {"slip_width_m": 0.002},
        }
    )
    readings = iter(widths)
    transport.gripper_width_m = lambda: next(readings)
    transport.capture_in_hand = lambda apple_id, view_index, *, timeout_s: ("capture", view_index)
    checked = []
    result = adapter.inspect_views(
        GraspProfile(7, 3, 0, 0.2, 0.09, 0.0, 0.0, 1000, True),
        InspectionPlan((pose(0.1), pose(0.2), pose(0.3)), 0.0),
        effort=1000,
        timeout_s=3.0,
        view_check=lambda index, capture: checked.append(index) or ViewVerdict.OK,
    )
    return result, checked


def test_width_growth_beyond_slip_width_stops_before_the_image_check():
    result, checked = _run_width_watch([0.070, 0.0705, 0.0745])

    assert not result.success
    assert result.slip_view_index == 1
    assert checked == [0]                                 # 뷰 1은 폭 변화에서 먼저 걸렸다


def test_closed_gripper_during_inspection_is_slip():
    result, _ = _run_width_watch([0.070, 0.070, 0.005])

    assert result.slip_view_index == 1


def test_stable_width_lets_the_inspection_finish():
    result, checked = _run_width_watch([0.070, 0.0701, 0.0699, 0.0701])

    assert result.success and result.slip_view_index is None
    assert checked == [0, 1, 2]


def test_width_watch_is_off_unless_configured():
    result, checked = _run_width_watch([0.070, 0.090, 0.001, 0.001], enabled=False)

    assert result.success and checked == [0, 1, 2]
