"""Benchmark accounting must not double count nested work or lose failures."""
import pytest

from aple.hal.simulation_timing import SimulationTiming


def test_nested_accounting_and_apple_identity():
    clock = [0., 0.]
    timer = SimulationTiming(lambda: clock[1], wall_clock=lambda: clock[0])
    timer.apple = 2
    with timer.measure('inspection'):
        clock[:] = [1., 2.]
        with timer.measure('capture'):
            clock[:] = [4., 2.]
        clock[:] = [5., 6.]
    totals = timer.summary()['stages']
    assert totals['inspection']['wall_time_s'] == 5.
    assert totals['inspection']['exclusive_wall_time_s'] == 2.
    assert totals['inspection']['simulated_time_s'] == 6.
    assert totals['capture']['simulated_time_s'] == 0.
    assert all(r['apple'] == 2 for r in timer.records)
    assert sum(t['exclusive_wall_time_s'] for t in totals.values()) == 5.


def test_failure_keeps_timings_and_restores_stack():
    clock = [0.]
    timer = SimulationTiming(lambda: clock[0], wall_clock=lambda: clock[0])
    with pytest.raises(RuntimeError):
        with timer.measure('pick'):
            with timer.measure('joint_motion'):
                clock[0] = 3.
                raise RuntimeError('motion failed')
    assert not timer.stack
    assert all(r['failed'] for r in timer.records)
    assert timer.summary()['stages']['pick']['wall_time_s'] == 3.


@pytest.mark.parametrize('args', [
    ['--quick', '--video'], ['--quick', '--save-captures'],
    ['--quick', '--apple-count', '3'], ['--apple-count', '0'],
])
def test_invalid_benchmark_arguments_fail_before_physics(args):
    from tools.run_in_hand_simulation import main
    with pytest.raises(SystemExit) as error:
        main(args)
    assert error.value.code == 2


def test_partial_benchmark_cannot_be_replayed_as_full_scene(tmp_path):
    import json
    from tools.replay_in_hand_studio import main
    (tmp_path/'report.json').write_text(json.dumps(dict(success=True, full_scene_complete=False)))
    with pytest.raises(ValueError, match='partial apple benchmark'):
        main([str(tmp_path)])
