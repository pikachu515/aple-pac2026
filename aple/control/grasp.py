"""크기 구간 → 파지 파라미터, 배치 계획, 태스크 문장 (S7-4~S7-6, S8-5, S8-6)."""

from __future__ import annotations

from aple.schema import AppleGeometry, BoxState, GradeResult, GraspParams, PlacePlan

# 학습 시 사용한 문형과 토큰 단위로 같아야 한다 (II장 2-3 ⑧)
TASK_TEMPLATE = "Pick the apple and place it in the {box} box"


def grasp_params_for(size_bin: str, table: dict) -> GraspParams:
    raise NotImplementedError


def plan_place(geom: AppleGeometry, grade: GradeResult, boxes: list[BoxState], table: dict) -> PlacePlan:
    """등급 → 목표 상자, far-to-near 순서의 빈 셀, 안전 하강 고도."""
    raise NotImplementedError
