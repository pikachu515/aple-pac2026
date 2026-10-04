import json
from pathlib import Path

import yaml

from aple.schema import (CycleRecord, FailStage, FailureCode, Grade, InspectionPolicy,
                         RejectReason, TaskAction)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_grade_and_box_share_numbers():
    assert [g.value for g in Grade] == [0, 1, 2]


def test_cycle_record_serializes_with_enum_names():
    rec = CycleRecord(
        cycle_index=1, scene_id=3, apple_id=2, variety="fuji",
        inspection_policy=InspectionPolicy.NEVER, task_action=TaskAction.PICK_PLACE, backend="scripted",
        diameter_m=0.078, ratio=0.52, ratio_lo=0.45, ratio_hi=0.60, coverage=0.25,
        grade=Grade.UPPER, reject_reason=RejectReason.NONE, target_box=Grade.UPPER, target_cell=4,
        grasp_force_set_n=3.5, planned_drop_m=0.01, settle_error_m=0.004,
        t_scan_s=1.2, t_plan_s=0.1, t_pick_s=6.0, t_inspect_s=0.0, t_place_s=5.0, t_total_s=12.3,
        success=False, failure_code=FailureCode.SLIP, fail_stage=FailStage.F2,
    )
    d = json.loads(json.dumps(rec.to_dict()))
    assert d["failure_code"] == "SLIP"
    assert d["fail_stage"] == "F2"
    assert d["grade"] == "UPPER"


def test_all_configs_parse():
    for p in CONFIGS.glob("*.yaml"):
        assert isinstance(yaml.safe_load(p.read_text(encoding="utf-8")), dict), p.name
