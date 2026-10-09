"""시간 궤적 추종·속도 감시 (10/10 실기: 정상 이동이 추종 오차·속도 초과 오판으로 멈춤)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import aple.hal.piper_studio as studio_module
from aple.execution.timed_motion import MotionLimits, TimedJointGoal
from aple.hal.piper_studio import PiperWebError
from tests.test_piper_studio import piper_web  # noqa: F401  (pytest fixture)


def _drive(monkeypatch, studio, read_state):
    """가짜 시계와 HTTP — read_state(clock) 가 로봇 관절값을 돌려준다."""
    clock = [0.]

    def request(method, path, body=None):
        clock[0] += .002
        if path.endswith('/start'):
            return {'status': 'started', 'running': True, 'iface': studio._iface}
        return {'status': 'stopped' if path.endswith('/stop') else 'ok'}

    monkeypatch.setattr(studio_module, 'time', SimpleNamespace(
        monotonic=lambda: clock[0], sleep=lambda d: clock.__setitem__(0, clock[0] + d)))
    monkeypatch.setattr(studio, '_ensure_authenticated', lambda **kw: None)
    monkeypatch.setattr(studio, 'verify_gripper_torque', lambda: 1.)
    monkeypatch.setattr(studio, '_start_estop_keepalive', lambda: SimpleNamespace(check=lambda: None, stop=lambda: None))
    monkeypatch.setattr(studio, '_read_raw_state', lambda: read_state(clock[0]))
    monkeypatch.setattr(studio, '_request', request)
    return clock


def _ramp(start, steps, step_rad, period_s):
    return tuple(TimedJointGoal((start[0] + (i + 1) * step_rad, *start[1:]), (i + 1) * period_s) for i in range(steps))


def test_lagging_but_following_robot_is_not_a_tracking_error(piper_web, monkeypatch):
    _, studio = piper_web
    start = studio.joint_state()
    goals = _ramp(start, 20, .03, .05)                     # 0.6 rad/s, 50 ms 주기

    def lagging(t):                                        # 명령보다 0.2 s 뒤에 같은 길을 따라온다
        done = max(0., min(t - .2, 1.)) / .05 * .03
        return (start[0] + min(done, .6), *start[1:]), 0.

    _drive(monkeypatch, studio, lagging)
    limits = MotionLimits((1.,)*6, (2.,)*6, .05, .1, .6)
    studio.send_timed_joint_trajectory(goals, limits, speed_scale=1., timeout_s=5.)


def test_robot_that_does_not_follow_is_still_a_tracking_error(piper_web, monkeypatch):
    _, studio = piper_web
    start = studio.joint_state()
    goals = _ramp(start, 20, .03, .05)
    _drive(monkeypatch, studio, lambda t: (start, 0.))     # 멈춰 있음
    limits = MotionLimits((1.,)*6, (2.,)*6, .05, .1, .6)
    with pytest.raises(PiperWebError, match='tracking error'):
        studio.send_timed_joint_trajectory(goals, limits, speed_scale=1., timeout_s=5.)


def test_back_to_back_reads_do_not_inflate_measured_velocity(piper_web, monkeypatch):
    _, studio = piper_web
    start = studio.joint_state()
    goals = _ramp(start, 10, .03, .05)

    def chunky(t):                                         # 0.6 rad/s 를 33 ms 마다 몰아서 갱신
        frame = int(t / .033)
        return (start[0] + min(frame * .033 * .6, .3), *start[1:]), 0.

    _drive(monkeypatch, studio, chunky)
    limits = MotionLimits((.6,)*6, (2.,)*6, .05, .1, .3)
    studio.send_timed_joint_trajectory(goals, limits, speed_scale=1., timeout_s=5.)


def test_arrival_timeout_reports_where_the_arm_stopped(piper_web, monkeypatch):
    _, studio = piper_web
    start = studio.joint_state()
    target = (start[0] + .1, *start[1:])
    _drive(monkeypatch, studio, lambda t: (start, 0.))     # 목표가 줄여졌거나 못 따라옴
    limits = MotionLimits((1.,)*6, (2.,)*6, .05, 1., 1.)
    with pytest.raises(PiperWebError, match=r'worst joint1 5\.7°'):
        studio.send_timed_joint_trajectory((TimedJointGoal(target, .2),), limits, speed_scale=1., timeout_s=1.)


def test_goal_is_repeated_while_waiting_so_robotd_deadman_does_not_freeze(piper_web, monkeypatch):
    """10/10 실기: 목표를 한 번만 보내고 기다리자 robotd 데드맨(500 ms)이 그리퍼 닫기 도중에 세웠다."""
    _, studio = piper_web
    start = studio.joint_state()
    target = (start[0] + .05, *start[1:])
    goals = []
    clock = _drive(monkeypatch, studio, lambda t: ((target if t > 1.0 else start), 0.))   # 1 s 뒤에야 도착
    plain = studio._request

    def counting(method, path, body=None):
        if path.endswith('/goal'):
            goals.append(clock[0])
        return plain(method, path, body)

    monkeypatch.setattr(studio, '_request', counting)
    limits = MotionLimits((1.,)*6, (2.,)*6, .05, 1., 1.)
    studio.send_timed_joint_trajectory((TimedJointGoal(target, .1),), limits, speed_scale=1., timeout_s=3.)
    gaps = [b - a for a, b in zip(goals, goals[1:])]
    assert len(goals) >= 4 and max(gaps) < .5                         # 데드맨보다 자주 다시 보냈다
