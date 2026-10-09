from dataclasses import replace
import copy
import math
import pytest
from aple.orchestration.speed_trial import speed_trial_advice
from aple.schema import FailureCode, TaskAction
from tests.test_cycle_manager import record


def config():
    return {'workcell': {'expected_apple_count': 2}, 'safety': {'speed_scale_max': .5},
        'execution': {'motion': {'speed_trial_steps': [.5,.6,.7,.8]}}}


def rows():
    return [replace(record(1,i),cycle_index=1) for i in (1,2)]


def test_advice_follows_complete_live_run_and_does_not_change_settings():
    cfg = config(); before = copy.deepcopy(cfg)
    assert '0.5 → 0.6' in speed_trial_advice(cfg, rows(), live_execution=True, cycles=1)
    assert cfg == before
    assert speed_trial_advice(cfg, rows(), live_execution=False, cycles=1) is None


@pytest.mark.parametrize('changes', [dict(success=False), dict(failure_code=FailureCode.SLIP),
    dict(task_action=TaskAction.SKIP), dict(apple_id=0), dict(target_cell=-1),
    dict(settle_error_m=math.nan),dict(t_pick_s=math.nan),
    dict(task_action=TaskAction.PICK_INSPECT_PLACE,t_inspect_s=math.nan)])
def test_partial_failed_or_unverified_runs_do_not_prompt_increase(changes):
    data=rows();data[0]=replace(data[0],**changes)
    assert speed_trial_advice(config(),data,live_execution=True,cycles=1) is None


def test_missing_duplicate_retry_and_incomplete_cycles_do_not_prompt():
    data=rows()
    for records,cycles in [(data[:1],1),[data+[data[0]],1],[[data[0],data[0]],1],[data,2]]:
        assert speed_trial_advice(config(),records,live_execution=True,cycles=cycles) is None


def test_all_requested_cycles_must_complete():
    data=rows()+[replace(r,cycle_index=2) for r in rows()]
    assert speed_trial_advice(config(),data,live_execution=True,cycles=2)


def test_upper_trial_limit_has_no_further_increase():
    cfg=config();cfg['safety']['speed_scale_max']=.8
    assert speed_trial_advice(cfg,rows(),live_execution=True,cycles=1) is None
