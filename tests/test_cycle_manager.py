from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from aple.orchestration.cycle_manager import CycleManager
from aple.orchestration.metrics_logger import MetricsLogger
from aple.schema import (
    CaptureMode,
    CycleRecord,
    FailStage,
    FailureCode,
    Grade,
    InspectionPolicy,
    RejectReason,
    SceneSnapshot,
    Source,
    TaskAction,
)


def snapshot(scene_id: int = 4) -> SceneSnapshot:
    return SceneSnapshot(
        scene_id=scene_id,
        source=Source.TOP,
        capture_mode=CaptureMode.FULL,
        color=np.zeros((2, 2, 3), dtype=np.uint8),
        depth_m=np.ones((2, 2), dtype=np.float32),
        intrinsics=(1.0, 1.0, 0.0, 0.0),
        stamp_s=float(scene_id),
    )


def record(scene_id: int, apple_id: int = 1) -> CycleRecord:
    return CycleRecord(
        cycle_index=0,
        scene_id=scene_id,
        apple_id=apple_id,
        variety="fuji",
        inspection_policy=InspectionPolicy.NEVER,
        task_action=TaskAction.PICK_PLACE,
        backend="scripted",
        diameter_m=0.078,
        ratio=0.52,
        ratio_lo=0.45,
        ratio_hi=0.60,
        coverage=0.25,
        grade=Grade.UPPER,
        reject_reason=RejectReason.NONE,
        target_box=Grade.UPPER,
        target_cell=0,
        grasp_force_set_n=3.5,
        planned_drop_m=0.01,
        settle_error_m=0.0,
        t_scan_s=1.0,
        t_plan_s=0.1,
        t_pick_s=1.0,
        t_inspect_s=0.0,
        t_place_s=1.0,
        t_total_s=3.1,
        success=True,
        failure_code=FailureCode.NONE,
        fail_stage=FailStage.NONE,
    )


def test_run_executes_stages_assigns_indices_logs_and_resets(tmp_path):
    calls = []
    logger = MetricsLogger(tmp_path / "cycles.csv")

    manager = CycleManager(
        {},
        capture=lambda: (calls.append("capture") or snapshot(10 + calls.count("capture"))),
        plan_scene=lambda scene: (calls.append(("plan", scene.scene_id)) or scene.scene_id),
        execute_scene=lambda plan: (calls.append(("execute", plan)) or "done"),
        make_records=lambda scene, plan, result: (
            calls.append(("record", scene.scene_id, plan, result))
            or [record(scene.scene_id, 1), record(scene.scene_id, 2)]
        ),
        reset=lambda: calls.append("reset"),
        logger=logger,
    )

    result = manager.run(cycles=2)

    assert [row.cycle_index for row in result] == [1, 1, 2, 2]
    assert [row.scene_id for row in result] == [11, 11, 12, 12]
    assert calls.count("reset") == 2
    assert next(index for index, call in enumerate(calls) if call[0] == "execute") < next(
        index for index, call in enumerate(calls) if call == "reset"
    )
    lines = (tmp_path / "cycles.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 5
    assert lines[0].startswith("cycle_index,scene_id,apple_id")


def test_dry_run_uses_replay_and_never_executes_or_resets():
    calls = []
    scene = snapshot()
    manager = CycleManager(
        {},
        capture=lambda: pytest.fail("dry-run must not use live capture"),
        dry_run_capture=lambda: (calls.append("replay") or scene),
        plan_scene=lambda captured: captured.scene_id,
        execute_scene=lambda plan: pytest.fail("dry-run must not execute robot actions"),
        make_records=lambda captured, plan, execution: [record(captured.scene_id)],
        reset=lambda: pytest.fail("dry-run must not reset hardware"),
        dry_run=True,
    )

    assert manager.run() == [replace(record(scene.scene_id), cycle_index=1)]
    assert calls == ["replay"]


def test_dry_run_requires_a_non_live_capture_provider():
    with pytest.raises(ValueError, match="dry_run_capture"):
        CycleManager(
            {},
            capture=snapshot,
            plan_scene=lambda scene: scene,
            make_records=lambda scene, plan, execution: [record(scene.scene_id)],
            dry_run=True,
        )


@pytest.mark.parametrize("cycles", [0, -1, True, 1.5])
def test_run_rejects_invalid_cycle_counts(cycles):
    manager = CycleManager(
        {},
        capture=snapshot,
        plan_scene=lambda scene: scene,
        execute_scene=lambda plan: None,
        make_records=lambda scene, plan, execution: [record(scene.scene_id)],
    )

    with pytest.raises(ValueError, match="positive integer"):
        manager.run(cycles=cycles)


def test_record_scene_must_match_snapshot():
    manager = CycleManager(
        {},
        capture=snapshot,
        plan_scene=lambda scene: scene,
        execute_scene=lambda plan: None,
        make_records=lambda scene, plan, execution: [record(scene.scene_id + 1)],
    )

    with pytest.raises(ValueError, match="scene_id"):
        manager.run()


def test_logger_defaults_to_configured_csv_directory(tmp_path):
    manager = CycleManager(
        {"orchestration": {"logging": {"csv_dir": str(tmp_path)}}},
        capture=snapshot,
        plan_scene=lambda scene: scene,
        execute_scene=lambda plan: None,
        make_records=lambda scene, plan, execution: [record(scene.scene_id)],
    )

    manager.run()

    assert (tmp_path / "cycles.csv").is_file()


def test_failed_cycle_does_not_reset_when_record_creation_raises():
    calls = []
    manager = CycleManager(
        {},
        capture=snapshot,
        plan_scene=lambda scene: scene,
        execute_scene=lambda plan: "failed",
        make_records=lambda scene, plan, execution: (_ for _ in ()).throw(
            RuntimeError("evaluation failed")
        ),
        reset=lambda: calls.append("reset"),
    )

    with pytest.raises(RuntimeError, match="evaluation failed"):
        manager.run()
    assert calls == []
