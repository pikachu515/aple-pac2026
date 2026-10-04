import json

from aple.schema import CycleLog, FailCode, GraspParams, PlacePlan


def test_cycle_log_serializes_to_json():
    plan = PlacePlan(
        apple_id=0, box_id="high", cell=2, descend_to_mm=120.0,
        grasp=GraspParams(size_bin="M", open_width_mm=85.0, grasp_height_mm=0.0, effort=350),
        task_text="Pick the apple and place it in the left box",
    )
    log = CycleLog(cycle=1, plan=plan, success=False, fail_code=FailCode.F2, cycle_time_s=18.4)
    d = json.loads(json.dumps(log.to_dict()))
    assert d["fail_code"] == "F2"
    assert d["plan"]["grasp"]["effort"] == 350
