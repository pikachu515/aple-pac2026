"""Speed selection must never accept partial or failed physical trials."""
from copy import deepcopy

import pytest

from tools.search_simulation_speed import complete_success
from aple.hal.simulation_optimization import SimulationIKRetry
from aple.execution.trajectory import ToolPose


@pytest.mark.parametrize('change', [
    {'full_scene_complete': False},
    {'apples': [{'captured_views': 4, 'execution_success': True}]},
    {'slips': [{'displacement_m': .013}]},
    {'errors': ['IK failed']},
    {'apples': [{'captured_views': 3, 'execution_success': True}]*3},
])
def test_speed_gate_rejects_incomplete_or_failed_trials(change):
    report = dict(success=True, full_scene_complete=True, slips=[], errors=[],
                  apples=[dict(captured_views=4, execution_success=True)]*3)
    assert complete_success(report, 0)
    assert not complete_success(report, 1)
    invalid = deepcopy(report)
    invalid.update(change)
    assert not complete_success(invalid, 0)


def test_ik_retry_preserves_actual_seed_and_rejects_out_of_bounds_solution():
    cfg = dict(piper_web=dict(enabled=False), simulation=dict(in_hand_scene={}),
               safety=dict(joint_limits_rad=[[-1, 1]]*6))
    seed = (0.,)*6
    calls = []
    class Solver:
        def solve(self, pose, guess):
            calls.append(guess)
            # First fallback reports success outside limits; it must be rejected.
            return None if len(calls) == 1 else (2.,)*6 if len(calls) == 2 else (.1,)*6
    retry = SimulationIKRetry(Solver(), cfg, .15)
    assert retry.solve(ToolPose((.2, 0., .2), (0., 0., 0., 1.)), seed) == (.1,)*6
    assert seed == (0.,)*6
    assert calls == [seed, (-.15, 0., 0., 0., 0., 0.), (.15, 0., 0., 0., 0., 0.)]
    cfg['piper_web']['enabled'] = True
    with pytest.raises(ValueError, match='disabled-hardware'):
        SimulationIKRetry(Solver(), cfg, .15)


def test_speed_scene_has_reject_capacity_without_changing_original_config():
    from tests.test_in_hand_simulation import config
    from aple.hal.in_hand_simulation import scene_config
    cfg, _ = config()
    before = deepcopy(cfg)
    result = scene_config(cfg, 'configs/simulation/in_hand_speed_search_scene.yaml', seed=42)
    assert cfg == before
    assert result['workcell']['boxes']['REJECT']['grid'] == [2, 2]
    assert len(result['simulation']['apples']) == 3
    assert not result['piper_web']['enabled']
