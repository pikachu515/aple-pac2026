"""스크립트 + ACT 흐름의 판단 — 집었나, hold 에서 멈췄나, 놓았나, 어느 사과였나."""

from aple.orchestration.act_flow import PickWatcher, PlaceWatcher, Sample, picked_grade
from aple.schema import AppleState, Grade, GradeResult, InspectMode, RejectReason

HOLD = (0.0, 1.4, -1.9, 0.0, 0.6, 0.0)
FAR = (0.3, 0.8, -0.9, 0.2, 0.3, 0.5)


def feed(watcher, rows, dt=0.1):
    out = []
    for i, (joints, width) in enumerate(rows):
        out.append(watcher.update(Sample(i * dt, joints, width)))
    return out


def test_pick_ends_only_after_grasp_and_still_at_hold():
    w = PickWatcher(HOLD, closed_min_m=0.061)
    rows = [(FAR, 0.11)] * 10 + [(FAR, 0.08)] * 10 + [(HOLD, 0.08)] * 15
    res = feed(w, rows)
    assert w.grasp_sample is not None and w.grasp_sample.joints == FAR   # 사과 위치에서 닫혔다
    assert res.index(True) >= 28 and res[-1]                     # hold 도착(20번째) 뒤 약 1초 정지해야 끝
    assert abs(w.held_width_m - 0.08) < 1e-9


def test_pick_at_hold_without_grasp_or_with_empty_gripper_never_ends():
    assert not any(feed(PickWatcher(HOLD, closed_min_m=0.061), [(HOLD, 0.11)] * 40))
    w = PickWatcher(HOLD, closed_min_m=0.061)
    assert not any(feed(w, [(FAR, 0.11)] * 10 + [(HOLD, 0.058)] * 30))      # 완전히 닫힘 = 빈손


def test_pick_waits_while_arm_still_moving_near_hold():
    w = PickWatcher(HOLD, closed_min_m=0.061)
    moving = [(tuple(q + 0.05 * (i % 2) for q in HOLD), 0.08) for i in range(20)]
    assert not any(feed(w, [(FAR, 0.11)] * 10 + [(FAR, 0.08)] * 10 + moving))


def test_place_ends_after_opening_and_retreat_time():
    w = PlaceWatcher(held_width_m=0.08)
    res = feed(w, [(HOLD, 0.08)] * 5 + [(HOLD, 0.11)] * 20)
    assert not any(res[:19]) and res[-1]


def apple(i, x, y):
    return AppleState(1, i, i, (x, y, 0.03), 0.08, 0.3, 0.03, 0.001, 0.9, 0.9, 0.0)


def grade(i, g):
    return GradeResult(1, i, g, 0.8, 0.7, 0.9, 0.25, 0.4, RejectReason.NONE, True, InspectMode.TOP_SINGLE, "fuji")


def test_picked_grade_is_nearest_apple_and_unknown_is_reject():
    apples = [apple(1, 0.40, 0.0), apple(2, 0.45, 0.10)]
    grades = [grade(1, Grade.UPPER), grade(2, Grade.MIDDLE)]
    assert picked_grade((0.44, 0.09), apples, grades) == (2, Grade.MIDDLE)
    assert picked_grade((0.60, -0.20), apples, grades) == (None, Grade.REJECT)
    assert picked_grade((0.40, 0.0), apples, grades[1:]) == (None, Grade.REJECT)


def test_pick_without_empty_check_accepts_fully_closed_grip():
    w = PickWatcher(HOLD)                                         # 수제 핑거: 쥐어도 완전히 닫힘으로 읽힌다
    res = feed(w, [(FAR, 0.11)] * 10 + [(FAR, 0.059)] * 10 + [(HOLD, 0.059)] * 15)
    assert w.grasp_sample is not None and res[-1]
