"""Tests for joining perception signals with role-B plans."""

from dataclasses import dataclass

import pytest

from aple.config import load_config
from aple.execution.goal_builder import Pose3D
from aple.execution.motion_executor import GraspCandidate, PhaseAResult
from aple.execution.pick_place_pipeline import PickPlacePipeline
from aple.execution.testing import RecordingMotionAdapter
from aple.schema import (
    AppleState,
    BoxOccupancy,
    FailureCode,
    Grade,
    GradeResult,
    InspectMode,
    Phase,
    RejectReason,
    TaskAction,
)


def apple(apple_id=1, *, scene_id=7, clearance_m=0.03):
    return AppleState(
        scene_id, apple_id, apple_id, (0.1, 0.2, 0.3), 0.078, 0.3,
        clearance_m, 0.001, 0.9, 0.9, 0.0,
    )


def grade(apple_id=1, *, scene_id=7, final=True, reject_reason=RejectReason.NONE):
    return GradeResult(
        scene_id, apple_id, Grade.UPPER, 0.8, 0.7, 0.9, 0.25, 0.4,
        reject_reason, final, InspectMode.TOP_SINGLE, "fuji",
    )


def box(*, scene_id=7, occupied=None):
    return BoxOccupancy(
        scene_id, Grade.UPPER, 2, 3, 0.09, 0.08, 0.08, (0.2, 0.1, 0.1),
        [0.10, 0.11, 0.12, 0.10, 0.13, 0.14],
        occupied if occupied is not None else [False] * 6,
        True,
    )


@dataclass
class GraspCandidateFixture:
    scene_id: int
    apple_id: int
    grasp_pose: Pose3D
    grasp_yaw_rad: float
    approach_vector: tuple[float, float, float]
    approach_height_m: float
    min_clearance_m: float
    collision_free: bool
    score: float


def grasp_candidate(apple_id=1, *, scene_id=7, collision_free=True):
    candidate: GraspCandidate = GraspCandidateFixture(
        scene_id=scene_id,
        apple_id=apple_id,
        grasp_pose=Pose3D((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0)),
        grasp_yaw_rad=0.0,
        approach_vector=(0.0, 0.0, 1.0),
        approach_height_m=0.08,
        min_clearance_m=0.02,
        collision_free=collision_free,
        score=0.5,
    )
    return candidate


@pytest.fixture
def cfg():
    config = load_config()
    config["safety"]["gripper_effort_max"] = 1000
    config["size_bins"]["force"]["mu"] = 0.5
    config["size_bins"]["force"]["f_bruise_n"] = 30.0
    config["size_bins"]["gripper_calib"]["effort_to_force_table"] = [
        [0, 0.0],
        [5000, 50.0],
    ]
    return config


def test_plan_scene_joins_contract_inputs_and_reserves_distinct_cells(cfg):
    apples = [apple(1, clearance_m=0.03), apple(2, clearance_m=0.08)]
    grades = [grade(1), grade(2)]

    result = PickPlacePipeline(cfg).plan_scene(
        apples, grades, [box()], [grasp_candidate(1), grasp_candidate(2)]
    )

    assert result.scene_id == 7
    assert [item.task.apple_id for item in result.items] == [2, 1]
    assert [item.apple for item in result.items] == [apples[1], apples[0]]
    assert all(item.profile.scene_id == 7 for item in result.items)
    placements = [item.placement for item in result.items]
    assert all(placement is not None and placement.feasible for placement in placements)
    assert len({
        placement.cell_index for placement in placements if placement is not None
    }) == 2
    assert all(item.execution is None for item in result.items)


def test_plan_scene_defers_in_hand_inspection_until_final_grade(cfg):
    result = PickPlacePipeline(cfg).plan_scene(
        [apple()], [grade(final=False)], [box()], [grasp_candidate()]
    )

    item = result.items[0]
    assert item.task.action == TaskAction.PICK_INSPECT_PLACE
    assert item.placement is None


def test_plan_scene_preserves_skip_and_defer_actions_without_placement(cfg):
    result = PickPlacePipeline(cfg).plan_scene(
        [apple(1), apple(2)],
        [grade(1, reject_reason=RejectReason.OVERSIZE), grade(2, final=False)],
        [box()],
        [grasp_candidate(1), grasp_candidate(2)],
    )

    by_id = {item.task.apple_id: item for item in result.items}
    assert by_id[1].task.action == TaskAction.SKIP
    assert by_id[1].placement is None
    assert by_id[2].task.action == TaskAction.PICK_INSPECT_PLACE
    assert by_id[2].placement is None


def test_plan_scene_rejects_mixed_scene_signals(cfg):
    with pytest.raises(ValueError, match="newer than the apple scene"):
        PickPlacePipeline(cfg).plan_scene(
            [apple()], [grade(scene_id=8)], [box()], [grasp_candidate()]
        )


def test_plan_scene_reuses_grade_only_within_configured_displacement(cfg):
    moved = apple(scene_id=8)
    moved.displacement_m = 0.004
    plan = PickPlacePipeline(cfg).plan_scene(
        [moved], [grade(scene_id=7)], [box(scene_id=8)], [grasp_candidate(scene_id=8)]
    )

    item = plan.items[0]
    assert item.task.action == TaskAction.PICK_PLACE
    assert item.task.grade_scene_id == 7
    assert item.placement is not None
    assert item.placement.scene_id == 8


def test_plan_scene_defers_reused_grade_when_displacement_exceeds_limit(cfg):
    moved = apple(scene_id=8)
    moved.displacement_m = 0.006
    plan = PickPlacePipeline(cfg).plan_scene(
        [moved], [grade(scene_id=7)], [box(scene_id=8)], [grasp_candidate(scene_id=8)]
    )

    assert plan.items[0].task.action == TaskAction.DEFER
    assert plan.items[0].placement is None


def test_execute_scene_uses_adapter_and_stops_after_first_failure(cfg):
    adapter = RecordingMotionAdapter(
        phase_a_result=PhaseAResult(
            False, FailureCode.IK_FAIL, Phase.APPROACH, False, False
        )
    )
    pipeline = PickPlacePipeline(cfg, adapter)
    plan = pipeline.plan_scene(
        [apple(1, clearance_m=0.03), apple(2, clearance_m=0.08)],
        [grade(1), grade(2)],
        [box()],
        [grasp_candidate(1), grasp_candidate(2)],
    )

    result = pipeline.execute_scene(plan)

    assert len([call for call in adapter.calls if call[0] == "phase_a"]) == 1
    assert adapter.calls[0][1]["apple_id"] == 2
    assert result.items[0].execution is not None
    assert result.items[0].execution.failure_code == FailureCode.IK_FAIL
    assert result.items[1].execution is None


def test_execute_scene_refuses_to_run_without_injected_adapter(cfg):
    plan = PickPlacePipeline(cfg).plan_scene(
        [apple()], [grade()], [box()], [grasp_candidate()]
    )

    with pytest.raises(ValueError, match="motion_adapter"):
        PickPlacePipeline(cfg).execute_scene(plan)


def test_mode_b_runs_pick_inspect_regrade_and_place_stages(cfg):
    adapter = RecordingMotionAdapter()
    pipeline = PickPlacePipeline(cfg, adapter)
    plan = pipeline.plan_scene(
        [apple()],
        [grade(final=False)],
        [box()],
        [grasp_candidate()],
    )
    item = plan.items[0]
    held = pipeline.pick_for_inspection(item)
    assert held.pick.success
    assert held.pick.apple_in_tcp is not None
    assert held.pick.apple_center_in_base == pytest.approx((0.1, 0.2, 0.38))

    held = pipeline.inspect_held_apple(held)
    assert held.inspection is not None and held.inspection.success
    assert held.inspection.captured_views == len(
        cfg["inspection"]["views"]["rotation_angles_rad"]
    )
    inspect_call = next(data for name, data in adapter.calls if name == "inspect_views")
    assert inspect_call["inspection_plan"].target_tool_poses[0].position_m == pytest.approx(
        held.pick.apple_center_in_base
    )

    placed = pipeline.place_regraded_apple(held, grade(final=True), [box()])

    assert placed.task.action == TaskAction.PICK_PLACE
    assert placed.task.grade_scene_id == 7
    assert placed.placement is not None and placed.placement.feasible
    assert placed.execution is not None and placed.execution.success
    assert [name for name, _ in adapter.calls] == [
        "pick", "inspect_views", "place_phase_a", "phase_b"
    ]
    place_call = next(data for name, data in adapter.calls if name == "place_phase_a")
    assert place_call["targets"].apple_in_tcp == held.pick.apple_in_tcp


def test_mode_b_does_not_place_when_not_all_views_were_captured(cfg):
    adapter = RecordingMotionAdapter(captured_views=1)
    pipeline = PickPlacePipeline(cfg, adapter)
    item = pipeline.plan_scene(
        [apple()],
        [grade(final=False)],
        [box()],
        [grasp_candidate()],
    ).items[0]
    held = pipeline.inspect_held_apple(pipeline.pick_for_inspection(item))

    assert held.inspection is not None and not held.inspection.success
    with pytest.raises(ValueError, match="successful in-hand inspection"):
        pipeline.place_regraded_apple(held, grade(final=True), [box()])
    assert not any(name == "place_phase_a" for name, _ in adapter.calls)


def test_plan_scene_defers_missing_or_colliding_candidates(cfg):
    result = PickPlacePipeline(cfg).plan_scene(
        [apple(1), apple(2)],
        [grade(1), grade(2)],
        [box()],
        [grasp_candidate(2, collision_free=False)],
    )

    by_id = {item.task.apple_id: item for item in result.items}
    assert by_id[1].task.action == TaskAction.DEFER
    assert by_id[1].grasp_candidate is None
    assert by_id[2].task.action == TaskAction.DEFER
    assert by_id[2].placement is None


def test_plan_scene_selects_best_collision_free_candidate_per_apple(cfg):
    unsafe = grasp_candidate(collision_free=False)
    safe_low_clearance = grasp_candidate()
    safe_low_clearance.min_clearance_m = 0.01
    safe_best = grasp_candidate()
    safe_best.min_clearance_m = 0.04
    safe_best.score = 0.9

    result = PickPlacePipeline(cfg).plan_scene(
        [apple()], [grade()], [box()],
        [unsafe, safe_low_clearance, safe_best],
    )

    item = result.items[0]
    assert item.task.action == TaskAction.PICK_PLACE
    assert item.grasp_candidate is safe_best


def test_plan_scene_defers_occluded_apple(cfg):
    occluded = apple()
    occluded.occluded = True

    result = PickPlacePipeline(cfg).plan_scene(
        [occluded], [grade()], [box()], [grasp_candidate()]
    )

    assert result.items[0].task.action == TaskAction.DEFER
    assert result.items[0].placement is None


def test_execute_scene_passes_candidate_contract_to_motion_adapter(cfg):
    adapter = RecordingMotionAdapter()
    candidate = grasp_candidate()
    plan = PickPlacePipeline(cfg, adapter).plan_scene(
        [apple()], [grade()], [box()], [candidate]
    )

    result = PickPlacePipeline(cfg, adapter).execute_scene(plan)

    assert result.items[0].execution.success
    phase_a = next(data for name, data in adapter.calls if name == "phase_a")
    assert phase_a["candidate_apple_id"] == candidate.apple_id
    assert phase_a["grasp_pose"] == candidate.grasp_pose
    assert phase_a["grasp_yaw_rad"] == candidate.grasp_yaw_rad
    assert phase_a["approach_height_m"] == candidate.approach_height_m
    assert phase_a["motion_targets"].pick_pose.position_m == pytest.approx(
        candidate.grasp_pose.position_m
    )


def test_pipeline_keeps_cell_reservations_until_explicit_release(cfg):
    pipeline = PickPlacePipeline(cfg)
    first_plan = pipeline.plan_scene(
        [apple()], [grade()], [box()], [grasp_candidate()]
    )
    second_plan = pipeline.plan_scene(
        [apple(2)], [grade(2)], [box()], [grasp_candidate(2)]
    )

    first_cell = first_plan.items[0].placement.cell_index
    second_cell = second_plan.items[0].placement.cell_index
    assert first_cell != second_cell

    pipeline.release_placement_reservation(int(Grade.UPPER), first_cell)
    third_plan = pipeline.plan_scene(
        [apple(3)], [grade(3)], [box()], [grasp_candidate(3)]
    )
    assert third_plan.items[0].placement.cell_index == first_cell
