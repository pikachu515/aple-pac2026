"""APPROACH/DESCEND 실패 뒤 scan 으로 물러나기 전 충돌 장면 (10/10 실기: 복구가 출발 자세에서 막혀 정지)."""

from __future__ import annotations

from types import SimpleNamespace

from aple.hal.live_runtime import recovery_scene_ready

JOINTS = (0.1, 1.2, -1.0, 0.0, 0.5, 0.0)


class FakeGuard:
    """blocking 에 든 사과 번호가 장애물로 들어 있으면 현재 자세를 충돌로 본다."""

    def __init__(self, blocking):
        self.blocking = set(blocking)
        self.obstacles = set()
        self.calls = []

    def clear_payload(self):
        self.calls.append("clear_payload")

    def update_scene(self, apples, boxes, *, target_apple_id):
        self.obstacles = {a.apple_id for a in apples if a.apple_id != target_apple_id}
        self.calls.append(("update_scene", target_apple_id))

    def is_segment_safe(self, start, end):
        assert start == end == JOINTS
        return not (self.obstacles & self.blocking)


APPLES = [SimpleNamespace(apple_id=i) for i in (1, 2, 3)]


def test_all_apples_stay_obstacles_when_the_pose_is_clear():
    guard = FakeGuard(blocking=())
    assert recovery_scene_ready(guard, JOINTS, APPLES, [], target_apple_id=2) is True
    assert guard.obstacles == {1, 2, 3}
    assert guard.calls == ["clear_payload", ("update_scene", None)]


def test_target_apple_is_dropped_only_when_the_arm_already_touches_it():
    guard = FakeGuard(blocking={2})                        # 팔이 집으려던 사과 2 옆에 멈춤
    assert recovery_scene_ready(guard, JOINTS, APPLES, [], target_apple_id=2) is True
    assert guard.obstacles == {1, 3}                       # 다른 사과는 그대로 장애물


def test_other_obstacles_still_block_recovery():
    guard = FakeGuard(blocking={3})                        # 대상이 아닌 사과와 겹침 → 운영자 확인
    assert recovery_scene_ready(guard, JOINTS, APPLES, [], target_apple_id=2) is False
