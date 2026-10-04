"""작업 순서·행동 결정 (설계서 task_planner). 필수 (단순형).

SKIP   : OVERSIZE / UNDERSIZE / 파지 불가
DEFER  : 가려짐(OCCLUDED) / 등급 없음 → 재촬영
PICK_INSPECT_PLACE : final=False (적응형 모드)
PICK_PLACE : 그 외
순서: 간격(clearance_m) 큰 사과 우선 → 도달성
"""

from __future__ import annotations

from aple.schema import AppleState, GradeResult, GraspProfile, TaskItem


def plan_queue(apples: list[AppleState], grades: list[GradeResult],
               profiles: list[GraspProfile]) -> list[TaskItem]:
    raise NotImplementedError
