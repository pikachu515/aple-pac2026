"""등급·리젝트 판정 (설계서 grade_classifier). 필수.

T = 품종 '상' 기준 (후지계 0.40, 홍로계 0.50)
상    : ratio_lo ≥ T + delta_up     ('중→상' 오분류 비용이 크므로 엄격)
중    : ratio_hi ≤ T + delta_mid    ('상→중'은 관대)
그 외 : REJECT (BORDERLINE)

리젝트 사유 우선순위: OVERSIZE/UNDERSIZE → OCCLUDED(재촬영) → LOW_CONF(조명 FAULT 포함)
→ LOW_COVERAGE → VIEW_DISAGREE → BORDERLINE
final: 적응형 모드에서 BORDERLINE·LOW_COVERAGE 는 False (인핸드 검사 대상)
"""

from __future__ import annotations

from aple.schema import AppleState, ColorRatio, GradeResult, InspectionPolicy, LightingStatus


def classify(cr: ColorRatio, apple: AppleState, light: LightingStatus,
             policy: InspectionPolicy, cfg: dict) -> GradeResult:
    raise NotImplementedError
