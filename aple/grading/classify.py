"""등급 분류 및 리젝트 (S6).

표준규격(1011) 임계 + 경계 구간 리젝트(Chow) + 뷰 간 편차 리젝트 + 크기 리젝트.
임계값은 configs/grading.yaml 에서 읽는다.
"""

from __future__ import annotations

from aple.schema import GradeResult


def classify(apple_id: int, ratio_by_view: dict[str, float], diameter_mm: float, cfg: dict) -> GradeResult:
    raise NotImplementedError
