"""Only acknowledged empty transit may use all measured joint limits."""
import copy
import math
import numpy as np
import pytest
from aple.hal.piper_motion import PiperTrajectoryMotionAdapter
from aple.execution.goal_builder import MotionTargets, PlaceTargets
from aple.execution.soft_landing import SoftLandingPlan, SoftLandingStep, GripperAction
from aple.execution.timed_motion import MotionLimits, time_parameterize
from aple.schema import AppleState, GraspProfile, PlacementPlan, Grade, Phase
from tests.test_piper_motion import make_adapter, FakeIK, pose


def adapter():
    old, transport, guard = make_adapter()
    cfg = copy.deepcopy(old._cfg)
    cfg['execution']['motion'] = {'empty_speed_scale': 1.0}
    return PiperTrajectoryMotionAdapter(transport, cfg, FakeIK(), guard), transport, guard


def inputs():
    apple = AppleState(1, 1, 1, (0., 0., .2), .08, .2, .03, 0., 1., 1.)
    profile = GraspProfile(1, 1, 0, .2, .09, 0., 0., 1000, True)
    plan = PlacementPlan(1, 1, Grade.UPPER, 0, (.4, 0., .2), .3, .2, .01, True)
    targets = MotionTargets(pose(.3), pose(.1, .3), pose(0.), pose(.4), pose(.4, .3), 0.)
    return apple, profile, plan, targets


def test_open_acknowledgment_boosts_only_pregrasp_and_retains_checked_path():
    a, transport, guard = adapter()
    apple, profile, plan, targets = inputs()
    result = a.execute_phase_a(apple, profile, None, plan, targets,
        backend='scripted', effort=1000, task_text='', speed_scale=.5, timeout_s=3.)
    assert result.success and a._load_state == 'held'
    moves = [c for c in transport.calls if c[0] == 'move']
    assert moves[0][2] == 1.
    assert all(c[2] == .5 for c in moves if c[1][0] > .1)
    assert guard.calls
    transport.calls.clear()
    assert a.move_named('scan', (.2, 0., 0., 0., 0., 0.), speed_scale=.5, timeout_s=3.)
    assert all(c[2] == .5 for c in transport.calls if c[0] == 'move')


def release(a, speed=.5):
    apple, profile, plan, targets = inputs()
    landing = SoftLandingPlan((
        SoftLandingStep(pose(.2, .25), speed, GripperAction.HOLD),
        SoftLandingStep(pose(.3), .1, GripperAction.RELEASE),
        SoftLandingStep(pose(.4, .3), speed, GripperAction.HOLD)), .25)
    return a.execute_phase_b(apple, profile, plan, targets, landing,
        effort=1000, final_descent_m=.03, slow_speed_scale=.1, timeout_s=3.)


def test_release_boosts_retreat_and_next_scan_after_success():
    a, transport, _ = adapter()
    assert release(a).success
    moves = [c for c in transport.calls if c[0] == 'move']
    assert {c[2] for c in moves if c[1][0] <= .3} == {.5, .1}
    assert all(c[2] == 1. for c in moves if c[1][0] > .3)
    transport.calls.clear()
    assert a.move_named('scan', (.5, 0., 0., 0., 0., 0.), speed_scale=.5, timeout_s=3.)
    assert all(c[2] == 1. for c in transport.calls if c[0] == 'move')


def test_recovery_retreat_remains_slow_even_when_empty():
    a, transport, _ = adapter()
    assert release(a, speed=.2).success
    assert all(c[2] == .2 for c in transport.calls if c[0] == 'move' and c[1][0] > .3)


def test_failed_release_cannot_grant_empty_speed():
    a, transport, _ = adapter()
    transport.gripper_success = False
    assert not release(a).success and a._load_state == 'unknown'
    transport.calls.clear()
    assert a.move_named('scan', (.5, 0., 0., 0., 0., 0.), speed_scale=.5, timeout_s=3.)
    assert all(c[2] == .5 for c in transport.calls if c[0] == 'move')


def test_startup_and_failed_motion_never_assume_empty():
    a, transport, guard = adapter()
    assert a.move_named('scan', (.1, 0., 0., 0., 0., 0.), speed_scale=.5, timeout_s=3.)
    assert all(c[2] == .5 for c in transport.calls if c[0] == 'move')
    assert release(a).success
    guard.safe = False
    assert not a.move_named('scan', (.5, 0., 0., 0., 0., 0.), speed_scale=.5, timeout_s=3.)
    assert a._load_state == 'unknown'
    guard.safe = True
    transport.calls.clear()
    assert a.move_named('scan', (.5, 0., 0., 0., 0., 0.), speed_scale=.5, timeout_s=3.)
    assert all(c[2] == .5 for c in transport.calls if c[0] == 'move')


def test_unconfirmed_lift_blocks_boost_after_failure():
    a, transport, _ = adapter()
    transport.holding_confirmed = False
    apple, profile, plan, targets = inputs()
    result = a.execute_phase_a(apple, profile, None, plan, targets,
        backend='scripted', effort=1000, task_text='', speed_scale=.5, timeout_s=3.)
    assert not result.success and a._load_state == 'unknown'
    transport.calls.clear()
    assert a.move_named('scan', (.5, 0., 0., 0., 0., 0.), speed_scale=.5, timeout_s=3.)
    assert all(c[2] == .5 for c in transport.calls if c[0] == 'move')


def test_full_scale_still_obeys_individual_velocity_and_acceleration_limits():
    limits = MotionLimits((.6, .9, 1.2, 1.5, 1.8, 2.), (1., 1.2, 1.5, 2., 2., 2.), .01, .01, .02)
    start = (0.,)*6
    end = (1., .6, -.8, 1.2, -.4, .2)
    goals = time_parameterize(start, (end,), limits, speed_scale=1.)
    q = np.array([start]+[g.position_rad for g in goals])
    velocity = np.diff(q, axis=0)/limits.period_s
    acceleration = np.diff(velocity, axis=0)/limits.period_s
    assert (np.max(np.abs(velocity),axis=0) <= np.array(limits.velocity_rad_s)+1e-8).all()
    assert (np.max(np.abs(acceleration),axis=0) <= np.array(limits.acceleration_rad_s2)+1e-8).all()
    slow = time_parameterize(start, (end,), limits, speed_scale=.5)
    assert goals[-1].time_s < slow[-1].time_s


@pytest.mark.parametrize('scale', [True, math.nan, 1.1, .4])
def test_invalid_empty_speed_is_rejected(scale):
    a, transport, guard = make_adapter()
    cfg = copy.deepcopy(a._cfg)
    cfg['execution']['motion'] = {'empty_speed_scale': scale}
    with pytest.raises(ValueError, match='empty_speed_scale'):
        PiperTrajectoryMotionAdapter(transport, cfg, FakeIK(), guard)


def test_inspection_pick_uses_empty_approach_then_normal_lift():
    from aple.execution.goal_builder import PickTargets
    a, transport, guard = adapter()
    apple, profile, _, full = inputs()
    transport.pose = full.pick_pose
    targets = PickTargets(full.pick_pose, full.pregrasp_pose, full.apple_in_tcp, 0.)
    result = a.execute_pick(apple, profile, None, targets,
        backend='scripted', effort=1000, task_text='', speed_scale=.5, timeout_s=3.)
    assert result.success and a._load_state == 'held'
    moves = [c for c in transport.calls if c[0] == 'move']
    assert moves[0][2] == 1.
    close = max(i for i,c in enumerate(transport.calls) if c[0] == 'gripper')
    assert all(c[2] == .5 for c in transport.calls[close+1:] if c[0] == 'move')
    assert guard.calls
