"""미끄러짐 복구 실제 동작 (PR #15 합의): 전부 저속, 공중에서 그리퍼를 열지 않는다.

1회차는 원래 트레이 자리에 내려놓고 재촬영·재파지, 2회차는 리젝트 상자에 적재 후 계속,
그리퍼가 이미 완전히 닫혔으면(사과를 놓침) 움직이지 않고 정지한다.
"""

from dataclasses import replace

import pytest

from aple.execution.motion_executor import PhaseBResult
from aple.execution.pick_place_pipeline import PickPlacePipeline
from aple.execution.testing import RecordingMotionAdapter
from aple.orchestration.recovery import RecoveryAction, RecoveryDecision, RecoveryPolicy
from aple.schema import FailureCode, Grade, Phase
from aple.execution.goal_builder import Pose3D
from tests.test_inhand import APPLE_IN_TCP
from tests.test_inspection import P
from tests.test_pick_place_pipeline import apple, box, cfg, grade, grasp_candidate  # noqa: F401
from tests.test_slip_per_view import ScriptedInspectionAdapter, _captures, _shift

RETRY = RecoveryDecision(RecoveryAction.RETRY_GRASP_ONCE)
REJECT = RecoveryDecision(RecoveryAction.PLACE_REJECT_AND_ALERT, alert=True)
STOP = RecoveryDecision(RecoveryAction.SAFE_STOP_AND_ALERT, alert=True)


def reject_box():
    return replace(box(), box_id=Grade.REJECT)


def held_apple(config, adapter=None):
    adapter = adapter or RecordingMotionAdapter()
    pipeline = PickPlacePipeline(config, adapter)
    item = pipeline.plan_scene(
        [apple()], [grade(final=False)], [box(), reject_box()], [grasp_candidate()]
    ).items[0]
    return adapter, pipeline, pipeline.pick_for_inspection(item)


def motion_calls(adapter):
    return [name for name, _ in adapter.calls if name in ("place_phase_a", "phase_b")]


def test_retry_sets_the_apple_back_on_its_original_spot_slowly(cfg):  # noqa: F811
    adapter, pipeline, held = held_apple(cfg)
    held = replace(held, slip=object())

    result = pipeline.recover_from_slip(held, RETRY, [box(), reject_box()])

    assert result.success and result.rescan and not result.stop and not result.alert
    assert result.failure_code == FailureCode.NONE
    plan = result.placement
    assert plan.cell_index == -1                                   # 상자 칸이 아니라 트레이 원래 자리
    assert plan.place_center_m[:2] == pytest.approx(held.item.apple.center_m[:2])
    assert plan.planned_drop_m <= cfg["workcell"]["placement"]["max_drop_m"]
    assert plan.hover_z_m > plan.release_z_m >= held.item.apple.center_m[2]
    place_a = next(data for name, data in adapter.calls if name == "place_phase_a")
    assert place_a["speed_scale"] == cfg["execution"]["recovery"]["speed_scale"]
    assert place_a["speed_scale"] < cfg["safety"]["speed_scale_max"]
    assert motion_calls(adapter) == ["place_phase_a", "phase_b"]
    assert result.held.inspection is None and result.held.regraded_grade is None
    assert result.held.slip is None                                 # 모드 B 결과는 버린다


def test_second_slip_places_in_the_reject_box_and_continues(cfg):  # noqa: F811
    adapter, pipeline, held = held_apple(cfg)

    result = pipeline.recover_from_slip(held, REJECT, [box(), reject_box()])

    assert result.success and result.alert
    assert not result.stop and not result.rescan                    # 알림만 남기고 사이클 계속
    assert result.placement.box_id == Grade.REJECT
    assert result.placement.cell_index >= 0
    place_a = next(data for name, data in adapter.calls if name == "place_phase_a")
    assert place_a["speed_scale"] == cfg["execution"]["recovery"]["speed_scale"]


def test_safe_stop_moves_nothing_and_never_opens_the_gripper(cfg):  # noqa: F811
    adapter, pipeline, held = held_apple(cfg)
    before = len(adapter.calls)

    result = pipeline.recover_from_slip(held, STOP, [box(), reject_box()])

    assert result.stop and result.alert and not result.success
    assert len(adapter.calls) == before


@pytest.mark.parametrize("decision", [RETRY, REJECT])
def test_closed_gripper_means_the_apple_is_lost_so_stop_without_moving(cfg, decision):  # noqa: F811
    adapter = RecordingMotionAdapter(gripper_width=0.005)           # lost_grip_width_m (0.01) 이하
    adapter, pipeline, held = held_apple(cfg, adapter)
    before = len(adapter.calls)

    result = pipeline.recover_from_slip(held, decision, [box(), reject_box()])

    assert result.stop and result.alert
    assert result.failure_code == FailureCode.DROP
    assert len(adapter.calls) == before


def test_gripper_still_holding_proceeds(cfg):  # noqa: F811
    adapter = RecordingMotionAdapter(gripper_width=0.07)
    adapter, pipeline, held = held_apple(cfg, adapter)

    assert pipeline.recover_from_slip(held, RETRY, [box()]).success


def test_failed_slow_set_down_stops_the_cycle(cfg):  # noqa: F811
    adapter = RecordingMotionAdapter(phase_b_result=PhaseBResult(False, Phase.RELEASE))
    adapter, pipeline, held = held_apple(cfg, adapter)

    result = pipeline.recover_from_slip(held, RETRY, [box()])

    assert result.stop and result.alert and not result.success


def test_no_free_reject_cell_stops_instead_of_dropping(cfg):  # noqa: F811
    adapter, pipeline, held = held_apple(cfg)
    full = replace(reject_box(), occupied=[True] * 6)
    before = len(adapter.calls)

    result = pipeline.recover_from_slip(held, REJECT, [box(), full])

    assert result.stop and not result.success
    assert len(adapter.calls) == before


def test_recovery_speed_may_not_exceed_the_safety_limit(cfg):  # noqa: F811
    cfg["safety"]["speed_scale_max"] = 0.5
    cfg["execution"]["recovery"]["speed_scale"] = cfg["safety"]["speed_scale_max"] + 0.1
    _, pipeline, held = held_apple(cfg)

    with pytest.raises(ValueError, match="speed_scale_max"):
        pipeline.recover_from_slip(held, RETRY, [box()])


def test_only_slip_actions_are_accepted(cfg):  # noqa: F811
    _, pipeline, held = held_apple(cfg)

    with pytest.raises(ValueError, match="not a slip recovery action"):
        pipeline.recover_from_slip(held, RecoveryDecision(RecoveryAction.CONTINUE), [box()])


class AlignedPickAdapter(ScriptedInspectionAdapter):
    """잡은 순간의 TCP 자세를 합성 영상이 가정한 apple_in_tcp 와 맞춘다."""

    def execute_pick(self, apple, profile, candidate, targets, **kwargs):
        base = super().execute_pick(apple, profile, candidate, targets, **kwargs)
        grasp = Pose3D(
            tuple(float(c - a) for c, a in zip(apple.center_m, APPLE_IN_TCP)),
            (0.0, 0.0, 0.0, 1.0),
        )
        return replace(base, tool_pose_at_grasp_base=grasp)


def _mode_b(config, *, lost=False):
    captures = _captures(config)
    captures[2] = _shift(captures[2])
    adapter = AlignedPickAdapter(captures)
    adapter.gripper_width = 0.005 if lost else None
    pipeline = PickPlacePipeline(config, adapter)
    item = pipeline.plan_scene(
        [apple()], [grade(final=False)], [box(), reject_box()], [grasp_candidate()]
    ).items[0]
    return adapter, pipeline, item


def test_mode_b_first_slip_inspection_stops_then_sets_the_apple_back(cfg):  # noqa: F811
    adapter, pipeline, item = _mode_b(cfg)

    out = pipeline.run_mode_b_item(item, P, RecoveryPolicy(cfg), [box(), reject_box()])

    assert out.failure_code == FailureCode.SLIP
    assert out.slip.view_index == 2
    assert out.recovery.action == RecoveryAction.RETRY_GRASP_ONCE
    assert out.recovery.success and out.recovery.rescan
    assert out.item is None                                          # 적재 완료 항목 없음
    assert motion_calls(adapter) == ["place_phase_a", "phase_b"]


def test_mode_b_second_slip_goes_to_the_reject_box(cfg):  # noqa: F811
    adapter, pipeline, item = _mode_b(cfg)

    out = pipeline.run_mode_b_item(
        item, P, RecoveryPolicy(cfg), [box(), reject_box()], slip_retry_count=1
    )

    assert out.recovery.action == RecoveryAction.PLACE_REJECT_AND_ALERT
    assert out.recovery.success and out.recovery.alert and not out.recovery.stop
    assert out.recovery.placement.box_id == Grade.REJECT


def test_mode_b_slip_with_closed_gripper_stops(cfg):  # noqa: F811
    adapter, pipeline, item = _mode_b(cfg, lost=True)

    out = pipeline.run_mode_b_item(item, P, RecoveryPolicy(cfg), [box(), reject_box()])

    assert out.slip.decision.action == RecoveryAction.SAFE_STOP_AND_ALERT
    assert out.recovery.stop
    assert motion_calls(adapter) == []


def test_mode_b_pick_failure_is_reported_without_inspection(cfg):  # noqa: F811
    adapter = RecordingMotionAdapter()
    adapter.phase_a_result = replace(adapter.phase_a_result, success=False, failure_code=FailureCode.IK_FAIL)
    pipeline = PickPlacePipeline(cfg, adapter)
    item = pipeline.plan_scene(
        [apple()], [grade(final=False)], [box()], [grasp_candidate()]
    ).items[0]

    out = pipeline.run_mode_b_item(item, P, RecoveryPolicy(cfg), [box()])

    assert out.failure_code == FailureCode.IK_FAIL
    assert out.recovery is None
    assert all(name != "inspect_views" for name, _ in adapter.calls)


def test_scene_regrasps_once_with_a_fresh_plan_then_rejects(cfg):
    from aple.execution.pick_place_pipeline import SceneWorkPlan
    from aple.schema import Grade

    adapter, pipeline, item = _mode_b(cfg)
    rescans = []

    def rescan(original):
        rescans.append(original.apple.apple_id)
        fresh = pipeline.plan_scene(
            [replace(original.apple, scene_id=8)],
            [grade(scene_id=8, final=False)],
            [replace(box(), scene_id=8), replace(reject_box(), scene_id=8)],
            [grasp_candidate(scene_id=8)],
        )
        return fresh.items[0], [replace(box(), scene_id=8), replace(reject_box(), scene_id=8)]

    out = pipeline.execute_scene(
        SceneWorkPlan(7, (item,)), inspect_params=P, boxes=[box(), reject_box()],
        rescan_for_retry=rescan,
    )

    assert rescans == [item.apple.apple_id]
    assert len([name for name, _ in adapter.calls if name == "pick"]) == 2
    assert out.items[0].execution.success
    assert out.items[0].placement.box_id == Grade.REJECT
    assert out.items[0].final_grade.grade == Grade.REJECT


def test_scene_does_not_regrasp_without_a_rescan_provider(cfg):
    from aple.execution.pick_place_pipeline import SceneWorkPlan

    adapter, pipeline, item = _mode_b(cfg)
    out = pipeline.execute_scene(
        SceneWorkPlan(7, (item,)), inspect_params=P, boxes=[box(), reject_box()],
    )

    assert len([name for name, _ in adapter.calls if name == "pick"]) == 1
    assert not out.items[0].execution.success
    assert out.items[0].execution.failure_code == FailureCode.SLIP


def test_recovery_descent_does_not_exceed_recovery_speed(cfg):
    cfg["execution"]["recovery"]["speed_scale"] = 0.05
    adapter, pipeline, held = held_apple(cfg)

    result = pipeline.recover_from_slip(held, RETRY, [box()])

    assert result.success
    landing = next(data for name, data in adapter.calls if name == "phase_b")
    assert landing["slow_speed_scale"] <= 0.05
