from dataclasses import replace
"""Role B planning and fixed-effort regression tests."""

from dataclasses import dataclass

import pytest

from aple.config import load_config
from aple.execution.gripper import effort_for
from aple.execution.goal_builder import Pose3D
from aple.execution.motion_executor import (
    GraspCandidate,
    MotionExecutor,
    PhaseAResult,
    PhaseBResult,
)
from aple.execution.testing import RecordingMotionAdapter
from aple.planning.grasp_profile import select
from aple.planning.placement_planner import PlacementPlanner
from aple.planning.task_planner import plan_queue
from aple.schema import (AppleState, BoxOccupancy, FailureCode, Grade, GradeResult, GraspProfile,
                         InspectMode, Phase, RejectReason, TaskAction, PlacementPlan)


def apple(apple_id=1, *, scene_id=1, diameter_m=0.078, clearance_m=0.03):
    return AppleState(scene_id, apple_id, apple_id, (0.1, 0.2, 0.3), diameter_m, 0.3, clearance_m, 0.001,
                      0.9, 0.9, 0.0)


def grade(apple_id=1, *, scene_id=1, result=Grade.UPPER, reason=RejectReason.NONE, final=True):
    return GradeResult(scene_id, apple_id, result, 0.8, 0.7, 0.9, 0.25, 0.4, reason, final,
                       InspectMode.TOP_SINGLE, "fuji")


def profile(apple_id=1, *, scene_id=1, feasible=True, effort=1000):
    return GraspProfile(scene_id, apple_id, 2, 0.2, 0.083, 0.0, 10.0, effort, feasible)


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


def box(*, heights=None, occupied=None, valid=True, scene_id=1):
    heights = heights or [0.10, 0.11, 0.12, 0.10, 0.13, 0.14]
    occupied = occupied or [False] * 6
    return BoxOccupancy(scene_id, Grade.UPPER, 2, 3, 0.09, 0.08, 0.08, (0.2, 0.1, 0.1),
                        heights, occupied, valid)


def grasp_candidate(apple_id=1, *, scene_id=1, collision_free=True):
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
    config['piper_web']['gripper_width_range_m'] = [.02, .12]
    config["execution"]["gripper"]["control_mode"] = "calibrated_force"
    config["safety"]["gripper_effort_max"] = 1000
    config["size_bins"]["force"]["mu"] = 0.5
    config["size_bins"]["force"]["f_bruise_n"] = 30.0
    # 일반 놓기 규칙(칸 바닥 위 release_gap, 낙하 ≤ 3 cm)을 검사한다 — 현장 설정(10/10 release_above_wall,
    # max_drop_m 0.10)과 무관하게 고정. 벽 위에서 놓기는 아래 test_release_above_wall_* 가 따로 본다
    config["workcell"]["placement"].update(max_drop_m=0.03, release_above_wall=False)
    config["size_bins"]["gripper_calib"]["effort_to_force_table"] = [
        [0, 0.0],
        [5000, 50.0],
    ]
    return config


def test_grasp_profile_uses_size_bin_and_calibrated_force(cfg):
    result = select(apple(), cfg)

    assert result.size_bin == 2
    assert result.open_width_m == pytest.approx(0.083)
    assert result.grasp_height_offset_m == 0.0
    assert result.est_mass_kg == pytest.approx(0.196, rel=0.03)
    assert result.grasp_force_n > 0
    assert result.feasible
    assert effort_for(result) == result.effort
    assert result.effort == pytest.approx(result.grasp_force_n * 100, abs=1)


def test_fixed_effort_profile_does_not_require_force_sensor_calibration():
    config = load_config()
    config['piper_web']['gripper_width_range_m'] = [.02, .12]
    result = select(apple(), config)

    assert result.grasp_force_n == 0
    assert result.effort == 1000
    assert result.feasible is True


def test_calibrated_force_mode_fails_closed_without_calibration():
    config = load_config()
    config["execution"]["gripper"]["control_mode"] = "calibrated_force"
    result = select(apple(), config)

    assert result.grasp_force_n == 0
    assert result.feasible is False


def test_fixed_effort_motion_does_not_require_force_calibration(cfg):
    cfg["execution"]["gripper"]["control_mode"] = "fixed_effort"
    cfg["size_bins"]["force"]["mu"] = None
    cfg["size_bins"]["force"]["f_bruise_n"] = None
    cfg["size_bins"]["gripper_calib"]["effort_to_force_table"] = None
    adapter = RecordingMotionAdapter()
    placement = PlacementPlan(
        1, 1, Grade.UPPER, 0, (0.3, 0.2, 0.2), 0.3, 0.2, 0.01, True
    )

    result = MotionExecutor(adapter, cfg).pick_place(
        apple(), profile(), grasp_candidate(), placement
    )

    assert result.success
    assert adapter.calls[0][1]["effort"] == cfg["execution"]["gripper"]["fixed_effort"]


def test_fixed_effort_motion_rejects_mismatched_profile_effort(cfg):
    cfg["execution"]["gripper"]["control_mode"] = "fixed_effort"
    adapter = RecordingMotionAdapter()
    placement = PlacementPlan(
        1, 1, Grade.UPPER, 0, (0.3, 0.2, 0.2), 0.3, 0.2, 0.01, True
    )

    with pytest.raises(ValueError, match="does not match"):
        MotionExecutor(adapter, cfg).pick_place(
            apple(), profile(effort=900), grasp_candidate(), placement
        )

    assert adapter.calls == []


def test_required_force_at_or_above_bruise_limit_is_infeasible(cfg):
    cfg["size_bins"]["force"]["f_bruise_n"] = 5.0

    result = select(apple(), cfg)

    assert result.feasible is False


def test_effort_is_derived_from_force_calibration_not_fixed_default(cfg):
    cfg["execution"]["gripper"]["fixed_effort"] = 1234
    result = select(apple(), cfg)

    assert result.feasible
    assert result.effort != 1234
    assert effort_for(result) == result.effort


@pytest.mark.parametrize("diameter_m", [0.029, 0.101])
def test_out_of_range_apple_is_not_graspable(cfg, diameter_m):
    result = select(apple(diameter_m=diameter_m), cfg)

    assert result.feasible is False
    assert result.size_bin == -1


def test_invalid_fixed_effort_is_rejected(cfg):
    cfg["execution"]["gripper"]["fixed_effort"] = 5001

    with pytest.raises(ValueError, match="0 to 5000"):
        select(apple(), cfg)


def test_task_queue_applies_actions_and_clearance_order():
    apples = [apple(1, clearance_m=0.03), apple(2, clearance_m=0.08), apple(3, clearance_m=0.05)]
    grades = [
        grade(1, result=Grade.REJECT, reason=RejectReason.BORDERLINE, final=False),
        grade(2, result=Grade.MIDDLE),
        grade(3, reason=RejectReason.LOW_CONF),
    ]
    profiles = [profile(1), profile(2), profile(3, feasible=False)]

    result = plan_queue(
        apples,
        grades,
        profiles,
        candidate_clearance={(1, 1): True, (1, 2): True, (1, 3): True},
    )

    assert [(item.apple_id, item.action) for item in result] == [
        (2, TaskAction.PICK_PLACE),
        (3, TaskAction.SKIP),
        (1, TaskAction.PICK_INSPECT_PLACE),
    ]
    assert result[-1].grade_scene_id == 1


def test_missing_grade_is_deferred_and_scene_mismatch_is_not_joined():
    result = plan_queue([apple()], [], [profile()], candidate_clearance={})

    assert result[0].action == TaskAction.DEFER
    assert result[0].grade is None


def test_task_queue_reuses_final_grade_from_recent_unoccluded_scene():
    current = apple(scene_id=3)
    current.displacement_m = 0.004

    result = plan_queue(
        [current],
        [grade(scene_id=2)],
        [profile(scene_id=3)],
        candidate_clearance={(3, 1): True},
        max_grade_reuse_displacement_m=0.005,
    )

    assert result[0].action == TaskAction.PICK_PLACE
    assert result[0].grade_scene_id == 2


@pytest.mark.parametrize(
    ("displacement_m", "occluded"),
    [(0.006, False), (0.004, True)],
)
def test_task_queue_does_not_reuse_stale_or_occluded_grade(displacement_m, occluded):
    current = apple(scene_id=3)
    current.displacement_m = displacement_m
    current.occluded = occluded

    result = plan_queue(
        [current],
        [grade(scene_id=2)],
        [profile(scene_id=3)],
        candidate_clearance={(3, 1): True},
        max_grade_reuse_displacement_m=0.005,
    )

    assert result[0].action == TaskAction.DEFER
    assert result[0].grade is None


def test_placement_uses_farthest_wall_cell_and_reserves_it(cfg):
    planner = PlacementPlanner(cfg)
    occupancy = box()
    result = planner.plan(apple(), grade(), [occupancy])

    assert result.feasible
    assert result.cell_index == 5
    assert result.place_center_m == pytest.approx((0.425, 0.235, 0.189))
    assert result.release_z_m == pytest.approx(0.189)
    assert result.planned_drop_m == pytest.approx(0.01)
    assert result.hover_z_m == pytest.approx(0.199)
    assert result.hover_z_m >= result.release_z_m + cfg["workcell"]["placement"]["margin_m"]
    assert (int(Grade.UPPER), 5) in planner.reserved
    planner.release(int(Grade.UPPER), 5)
    assert (int(Grade.UPPER), 5) not in planner.reserved


def test_placement_uses_last_valid_box_when_latest_measurement_is_invalid(cfg):
    planner = PlacementPlanner(cfg)
    valid_box = box()
    first = planner.plan(apple(), grade(), [valid_box])
    invalid_measurement = box(scene_id=2, valid=False, heights=[0.9] * 6)

    second = planner.plan(
        apple(2, scene_id=2),
        grade(2, scene_id=2),
        [invalid_measurement],
    )

    assert first.feasible
    assert second.feasible
    assert second.place_center_m[2] < 0.3
    assert second.cell_index != first.cell_index


def test_placement_does_not_use_invalid_box_without_valid_history(cfg):
    result = PlacementPlanner(cfg).plan(
        apple(), grade(), [box(valid=False)]
    )

    assert result.feasible is False


def test_placement_returns_infeasible_when_no_cell_is_available(cfg):
    planner = PlacementPlanner(cfg)
    result = planner.plan(apple(), grade(), [box(occupied=[True] * 6)])

    assert result.feasible is False
    assert result.cell_index == -1


def test_placement_accepts_reused_grade_from_older_scene(cfg):
    reused_grade = grade(scene_id=1)

    result = PlacementPlanner(cfg).plan(
        apple(scene_id=2), reused_grade, [box(scene_id=2)]
    )

    assert result.feasible
    assert result.scene_id == 2


def test_placement_rejects_grade_from_future_scene(cfg):
    with pytest.raises(ValueError, match="newer scene"):
        PlacementPlanner(cfg).plan(
            apple(scene_id=1), grade(scene_id=2), [box(scene_id=1)]
        )


def test_invalid_placement_gap_cannot_exceed_drop_limit(cfg):
    cfg["workcell"]["placement"]["release_gap_m"] = 0.031
    result = PlacementPlanner(cfg).plan(apple(), grade(), [box()])

    assert result.feasible is False


def test_motion_executor_runs_phase_b_only_after_holding_at_hover(cfg):
    adapter = RecordingMotionAdapter()
    result = MotionExecutor(adapter, cfg).pick_place(apple(), profile(), grasp_candidate(), PlacementPlan(
        1, 1, Grade.UPPER, 0, (0.3, 0.2, 0.2), 0.3, 0.2, 0.01, True
    ))

    assert result.success
    assert result.failure_code.value == 0
    assert result.phase_reached.value == 9
    assert [name for name, _ in adapter.calls] == ["phase_a", "phase_b"]
    assert adapter.calls[0][1]["effort"] == 1000
    assert adapter.calls[1][1]["slow_speed_scale"] == pytest.approx(0.1)
    assert adapter.calls[1][1]["landing_plan"].steps[1].gripper_action.value == "release"


@pytest.mark.parametrize(
    "phase_a, expected_code",
    [
        (PhaseAResult(False, FailureCode.IK_FAIL, Phase.APPROACH, False, False), FailureCode.IK_FAIL),
        (PhaseAResult(True, FailureCode.NONE, Phase.GRASP, False, False), FailureCode.NOT_HOLDING),
        (PhaseAResult(True, FailureCode.NONE, Phase.TRANSIT, True, False), FailureCode.HOVER_NOT_REACHED),
        (PhaseAResult(True, FailureCode.NONE, Phase.TRANSIT, True, True), FailureCode.HOVER_NOT_REACHED),
    ],
)
def test_motion_executor_fails_closed_before_phase_b(cfg, phase_a, expected_code):
    adapter = RecordingMotionAdapter(phase_a_result=phase_a)
    result = MotionExecutor(adapter, cfg).pick_place(apple(), profile(), grasp_candidate(), PlacementPlan(
        1, 1, Grade.UPPER, 0, (0.3, 0.2, 0.2), 0.3, 0.2, 0.01, True
    ))

    assert not result.success
    assert result.failure_code == expected_code
    assert [name for name, _ in adapter.calls] == ["phase_a"]


def test_motion_executor_passes_act_policy_and_profile_effort(cfg):
    adapter = RecordingMotionAdapter()
    result = MotionExecutor(adapter, cfg).pick_place(
        apple(), profile(), grasp_candidate(),
        PlacementPlan(1, 1, Grade.UPPER, 0, (0.3, 0.2, 0.2), 0.3, 0.2, 0.01, True),
        backend="act",
    )

    assert result.success
    assert adapter.calls[0][1]["backend"] == "act"
    assert adapter.calls[0][1]["task_text"] == cfg["execution"]["act"]["task_text"]
    assert adapter.calls[0][1]["effort"] == 1000


def test_act_timeout_is_applied_without_exceeding_phase_a_limit(cfg):
    adapter = RecordingMotionAdapter()
    plan = PlacementPlan(1, 1, Grade.UPPER, 0, (0.3, 0.2, 0.2), 0.3, 0.2, 0.01, True)
    cfg["execution"]["act"]["timeout_s"] = 12.0

    MotionExecutor(adapter, cfg).pick_place(apple(), profile(), grasp_candidate(), plan, backend="act")

    assert adapter.calls[0][1]["timeout_s"] == 12.0

    cfg["execution"]["act"]["timeout_s"] = cfg["execution"]["timeouts"]["phase_a_s"] + 30.0   # phase A 한계보다 길게
    adapter.calls.clear()
    MotionExecutor(adapter, cfg).pick_place(apple(), profile(), grasp_candidate(), plan, backend="act")

    assert adapter.calls[0][1]["timeout_s"] == cfg["execution"]["timeouts"]["phase_a_s"]


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_act_timeout_must_be_finite_and_positive(cfg, timeout):
    adapter = RecordingMotionAdapter()
    cfg["execution"]["act"]["timeout_s"] = timeout

    with pytest.raises(ValueError, match="execution.act.timeout_s"):
        MotionExecutor(adapter, cfg).pick_place(
            apple(), profile(), grasp_candidate(),
            PlacementPlan(1, 1, Grade.UPPER, 0, (0.3, 0.2, 0.2), 0.3, 0.2, 0.01, True),
            backend="act",
        )

    assert adapter.calls == []


def test_motion_executor_uses_configured_default_backend(cfg):
    cfg["execution"]["default_backend"] = "act"
    adapter = RecordingMotionAdapter()

    result = MotionExecutor(adapter, cfg).pick_place(
        apple(), profile(), grasp_candidate(),
        PlacementPlan(1, 1, Grade.UPPER, 0, (0.3, 0.2, 0.2), 0.3, 0.2, 0.01, True)
    )

    assert result.success
    assert adapter.calls[0][1]["backend"] == "act"
    assert adapter.calls[0][1]["task_text"] == cfg["execution"]["act"]["task_text"]


def test_explicit_motion_backend_overrides_configured_default(cfg):
    cfg["execution"]["default_backend"] = "act"
    adapter = RecordingMotionAdapter()

    MotionExecutor(adapter, cfg).pick_place(
        apple(),
        profile(),
        grasp_candidate(),
        PlacementPlan(1, 1, Grade.UPPER, 0, (0.3, 0.2, 0.2), 0.3, 0.2, 0.01, True),
        backend="scripted",
    )

    assert adapter.calls[0][1]["backend"] == "scripted"


def test_placement_hover_clears_highest_adjacent_cell(cfg):
    occupancy = box(heights=[0.10, 0.10, 0.16, 0.10, 0.10, 0.10])

    result = PlacementPlanner(cfg).plan(apple(), grade(), [occupancy])

    assert result.feasible
    assert result.hover_z_m >= 0.16 + apple().diameter_m / 2 + cfg["workcell"]["placement"]["margin_m"]
    assert result.hover_z_m >= result.release_z_m + cfg["workcell"]["placement"]["margin_m"]


def test_motion_executor_rejects_effort_over_safety_limit(cfg):
    cfg["safety"]["gripper_effort_max"] = 999
    adapter = RecordingMotionAdapter()

    with pytest.raises(ValueError, match="exceeds"):
        MotionExecutor(adapter, cfg).pick_place(
            apple(), profile(), grasp_candidate(),
            PlacementPlan(1, 1, Grade.UPPER, 0, (0.3, 0.2, 0.2), 0.3, 0.2, 0.01, True)
        )
    assert adapter.calls == []


def test_motion_executor_rejects_drop_above_configured_limit(cfg):
    adapter = RecordingMotionAdapter()

    with pytest.raises(ValueError, match="max_drop_m"):
        MotionExecutor(adapter, cfg).pick_place(
            apple(),
            profile(),
            grasp_candidate(),
            PlacementPlan(
                1, 1, Grade.UPPER, 0, (0.3, 0.2, 0.2),
                0.3, 0.2, 0.031, True,
            ),
        )

    assert adapter.calls == []


def test_move_named_refuses_unmeasured_joint_pose(cfg):
    adapter = RecordingMotionAdapter()

    result = MotionExecutor(adapter, cfg).move_named("home")

    assert not result.success
    assert result.failure_code == FailureCode.ABORTED
    assert adapter.calls == []


def test_move_named_sends_configured_six_joint_pose(cfg):
    cfg["workcell"]["poses"]["scan_joint_rad"] = [0.0] * 6
    adapter = RecordingMotionAdapter()

    result = MotionExecutor(adapter, cfg).move_named("scan")

    assert result.success
    assert adapter.calls[0][0] == "move_named"
    assert adapter.calls[0][1]["joint_target_rad"] == (0.0,) * 6


def test_motion_executor_refuses_scene_or_apple_mismatch(cfg):
    adapter = RecordingMotionAdapter()
    plan = PlacementPlan(2, 1, Grade.UPPER, 0, (0.3, 0.2, 0.2), 0.3, 0.2, 0.01, True)

    with pytest.raises(ValueError, match="scene_id"):
        MotionExecutor(adapter, cfg).pick_place(apple(), profile(), grasp_candidate(), plan)
    assert adapter.calls == []


@pytest.mark.parametrize(
    "phase_b",
    [PhaseBResult(False, Phase.SOFT_LAND), PhaseBResult(True, Phase.SOFT_LAND)],
)
def test_phase_b_failure_or_missing_retreat_reports_failure(cfg, phase_b):
    adapter = RecordingMotionAdapter(phase_b_result=phase_b)

    result = MotionExecutor(adapter, cfg).pick_place(
        apple(), profile(), grasp_candidate(),
        PlacementPlan(1, 1, Grade.UPPER, 0, (0.3, 0.2, 0.2), 0.3, 0.2, 0.01, True)
    )

    assert not result.success
    assert result.failure_code == FailureCode.ABORTED
    assert result.phase_reached == Phase.SOFT_LAND


def test_inspect_uses_profile_effort_adapter(cfg):
    adapter = RecordingMotionAdapter()

    result = MotionExecutor(adapter, cfg).inspect(
        profile(),
        Pose3D((0.0, 0.0, 0.04), (0.0, 0.0, 0.0, 1.0)),
        (0.1, 0.2, 0.3),
        {
            "tilt_rad": 1.0,
            "rotation_angles_rad": [0.0, 1.0],
            "settle_time_s": 0.3,
        },
    )

    assert result.success
    call = adapter.calls[0]
    assert call[0] == "inspect"
    assert call[1]["apple_id"] == 1
    assert call[1]["effort"] == 1000
    assert call[1]["timeout_s"] == cfg["execution"]["timeouts"]["inspect_s"]
    assert len(call[1]["inspection_plan"].target_tool_poses) == 2


def _box(heights, valid=True):
    return BoxOccupancy(1, Grade.UPPER, 1, len(heights), 0.1, 0.1, 0.1, (0.0, 0.0, 0.0),
                        list(heights), [False] * len(heights), valid)


def _planned(cfg):
    planner = PlacementPlanner(cfg)
    plan = planner.plan(apple(), grade(), [_box([0.0, 0.0, 0.0])])
    assert plan.feasible
    return planner, plan


def test_verify_placement_success_releases_reservation(cfg):
    planner, plan = _planned(cfg)
    heights = [0.0] * 3
    heights[plan.cell_index] = 0.078
    result = planner.verify(plan, apple(), _box([0.0] * 3), replace(_box(heights), scene_id=2, occupied=[h > 0 for h in heights]))
    assert result.success and not result.rolled_out and result.verified
    assert not planner.reserved


def test_verify_placement_detects_roll_out(cfg):
    planner, plan = _planned(cfg)
    heights = [0.0] * 3
    heights[(plan.cell_index + 1) % 3] = 0.078
    result = planner.verify(plan, apple(), _box([0.0] * 3), replace(_box(heights), scene_id=2, occupied=[h > 0 for h in heights]))
    assert result.rolled_out and not result.success
    assert not planner.reserved


def test_verify_placement_invalid_depth_keeps_reservation(cfg):
    planner, plan = _planned(cfg)
    result = planner.verify(plan, apple(), _box([0.0] * 3), _box([0.0] * 3, valid=False))
    assert not result.verified
    assert planner.reserved


def test_verify_placement_rejects_occupancy_from_another_box(cfg):
    planner, plan = _planned(cfg)
    other_box = BoxOccupancy(
        1, Grade.MIDDLE, 1, 3, 0.1, 0.1, 0.1, (0.0, 0.0, 0.0),
        [0.0] * 3, [False] * 3, True,
    )

    with pytest.raises(ValueError, match="planned box_id"):
        planner.verify(plan, apple(), other_box, other_box)

    assert (int(plan.box_id), plan.cell_index) in planner.reserved


def test_release_above_wall_lifts_release_to_wall_top(cfg):
    """10/10 현장: 칸(9 cm)이 핑거 벌림보다 좁아 벽 위에서 놓는다 — 낙하 ≈ 벽 높이."""
    cfg["workcell"]["placement"].update(max_drop_m=0.12, release_above_wall=True)    # 테스트 상자 벽 0.10 m
    planner = PlacementPlanner(cfg)
    plan = planner.plan(apple(), grade(), [_box([0.0, 0.0, 0.0])])         # 벽 0.1 m, 칸 바닥 0
    r, gap = 0.078 / 2, cfg["workcell"]["placement"]["release_gap_m"]
    assert plan.feasible
    assert plan.release_z_m == pytest.approx(0.1 + r + gap)
    assert plan.planned_drop_m == pytest.approx(0.1 + gap)
    assert plan.hover_z_m >= plan.release_z_m


def test_release_above_wall_still_respects_max_drop(cfg):
    cfg["workcell"]["placement"].update(max_drop_m=0.05, release_above_wall=True)
    plan = PlacementPlanner(cfg).plan(apple(), grade(), [_box([0.0, 0.0, 0.0])])
    assert plan.feasible is False
