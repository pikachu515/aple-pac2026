"""작업 순서·행동 결정 (설계서 task_planner). 필수 (단순형).

SKIP   : OVERSIZE / UNDERSIZE / 파지 불가
DEFER  : 가려짐(OCCLUDED) / 등급 없음 → 재촬영
PICK_INSPECT_PLACE : final=False (적응형 모드)
PICK_PLACE : 그 외
순서: 간격(clearance_m) 큰 사과 우선 → 도달성
"""

from __future__ import annotations

import math
from collections.abc import Mapping

from aple.schema import (
    AppleState,
    GradeResult,
    GraspProfile,
    RejectReason,
    TaskAction,
    TaskItem,
)


def plan_queue(
    apples: list[AppleState],
    grades: list[GradeResult],
    profiles: list[GraspProfile],
    candidate_clearance: Mapping[tuple[int, int], bool],
    *,
    max_grade_reuse_displacement_m: float | None = None,
) -> list[TaskItem]:
    """Build a deterministic pick queue, reusing only eligible prior grades."""
    apple_by_key = {(apple.scene_id, apple.apple_id): apple for apple in apples}
    if len(apple_by_key) != len(apples):
        raise ValueError("duplicate (scene_id, apple_id) in apples")

    grade_by_key = {(grade.scene_id, grade.apple_id): grade for grade in grades}
    if len(grade_by_key) != len(grades):
        raise ValueError("duplicate (scene_id, apple_id) in grades")
    if max_grade_reuse_displacement_m is not None and (
        not math.isfinite(max_grade_reuse_displacement_m)
        or max_grade_reuse_displacement_m < 0
    ):
        raise ValueError("max_grade_reuse_displacement_m must be finite and non-negative")

    profile_by_key = {(profile.scene_id, profile.apple_id): profile for profile in profiles}
    if len(profile_by_key) != len(profiles):
        raise ValueError("duplicate (scene_id, apple_id) in profiles")

    queue: list[tuple[AppleState, TaskItem]] = []
    for key, apple in apple_by_key.items():
        grade = grade_by_key.get(key)
        if grade is None and max_grade_reuse_displacement_m is not None:
            reusable = [
                candidate
                for candidate in grades
                if candidate.apple_id == apple.apple_id
                and candidate.scene_id < apple.scene_id
                and candidate.final
                and not apple.occluded
                and math.isfinite(apple.displacement_m)
                and apple.displacement_m >= 0
                and apple.displacement_m <= max_grade_reuse_displacement_m
            ]
            if reusable:
                grade = max(reusable, key=lambda candidate: candidate.scene_id)
        profile = profile_by_key.get(key)

        if profile is None or not profile.feasible:
            action = TaskAction.SKIP
            grade_value = None if grade is None else grade.grade
            grade_scene_id = 0 if grade is None else grade.scene_id
        elif grade is not None and grade.reject_reason in (
            RejectReason.OVERSIZE,
            RejectReason.UNDERSIZE,
            RejectReason.UNSAFE_GRASP,
        ):
            action = TaskAction.SKIP
            grade_value = grade.grade
            grade_scene_id = grade.scene_id
        elif grade is None or apple.occluded:
            action = TaskAction.DEFER
            grade_value = None if grade is None else grade.grade
            grade_scene_id = 0 if grade is None else grade.scene_id
        elif (
            grade.reject_reason == RejectReason.OCCLUDED
            or candidate_clearance.get(key) is not True
        ):
            action = TaskAction.DEFER
            grade_value = grade.grade
            grade_scene_id = grade.scene_id
        elif not grade.final:
            action = TaskAction.PICK_INSPECT_PLACE
            grade_value = grade.grade
            grade_scene_id = grade.scene_id
        else:
            action = TaskAction.PICK_PLACE
            grade_value = grade.grade
            grade_scene_id = grade.scene_id

        queue.append(
            (
                apple,
                TaskItem(
                    scene_id=apple.scene_id,
                    apple_id=apple.apple_id,
                    action=action,
                    grade=grade_value,
                    grade_scene_id=grade_scene_id,
                ),
            )
        )

    for apple, _ in queue:
        if not math.isfinite(apple.clearance_m):
            raise ValueError(f"apple {apple.apple_id} has non-finite clearance_m")
    queue.sort(key=lambda pair: (-pair[0].clearance_m, pair[0].apple_id))
    return [item for _, item in queue]
