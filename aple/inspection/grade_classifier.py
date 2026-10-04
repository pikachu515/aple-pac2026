"""등급·리젝트 판정 (설계서 grade_classifier, 전략 S6-2).

T = 품종 '상' 기준 (후지계 0.40, 홍로계 0.50)
상    : ratio_lo ≥ T + delta_up     ('중→상' 오분류 비용이 크므로 엄격)
중    : ratio_hi ≤ T + delta_mid    ('상→중'은 관대, delta_mid ≥ 0)
그 외 : REJECT (BORDERLINE)

리젝트 사유는 위에서부터 먼저 맞는 것을 쓴다.
  1 OVERSIZE / UNDERSIZE   직경이 그리퍼 범위 밖        → 이동하지 않음
  2 OCCLUDED               가려짐 · 측정 실패           → 리젝트 아님, 재촬영
  3 LOW_CONF               검출 신뢰도 부족 · 조명 FAULT
  4 LOW_COVERAGE           관측 커버리지 부족
  5 VIEW_DISAGREE          뷰 간 편차 과다 (모드 B)
  6 BORDERLINE             판정 구간이 기준선에 걸침

final (설계서 M3.5)
  모드 B 결과                         → True
  모드 A + 정책 NEVER                 → True
  모드 A + 정책 ALWAYS                → False (단 사유 1·2는 True)
  모드 A + 정책 ADAPTIVE              → 사유 BORDERLINE·LOW_COVERAGE 면 False, 그 외 True
"""

from __future__ import annotations

from aple.inspection.coloration import InspectParams
from aple.schema import (AppleState, ColorRatio, Grade, GradeResult, InspectionPolicy, InspectMode,
                         LightingStatus, LightState, RejectReason)

_ALWAYS_FINAL = (RejectReason.OVERSIZE, RejectReason.UNDERSIZE, RejectReason.OCCLUDED)
_ADAPTIVE_RECHECK = (RejectReason.BORDERLINE, RejectReason.LOW_COVERAGE)


def decide(cr: ColorRatio, p: InspectParams) -> tuple[Grade, RejectReason]:
    """색 기준만으로 본 등급 (사유 4~6)."""
    if cr.coverage < p.min_coverage:
        return Grade.REJECT, RejectReason.LOW_COVERAGE
    if cr.inspect_mode == InspectMode.IN_HAND_MULTI and cr.view_disagreement > p.max_view_disagreement:
        return Grade.REJECT, RejectReason.VIEW_DISAGREE
    T = p.coloring_threshold
    if cr.ratio_lo >= T + p.delta_up:
        return Grade.UPPER, RejectReason.NONE
    if cr.ratio_hi <= T + p.delta_mid:
        return Grade.MIDDLE, RejectReason.NONE
    return Grade.REJECT, RejectReason.BORDERLINE


def classify(cr: ColorRatio | None, p: InspectParams, *, apple: AppleState | None = None,
             light: LightingStatus | None = None, gripper_range_mm: tuple[float, float] = (30.0, 100.0),
             policy: InspectionPolicy = InspectionPolicy.NEVER,
             scene_id: int = 0, apple_id: int = 0) -> GradeResult:
    reason = RejectReason.NONE
    grade = Grade.REJECT

    if apple is not None and apple.diameter_m * 1000 > gripper_range_mm[1]:
        reason = RejectReason.OVERSIZE
    elif apple is not None and apple.diameter_m * 1000 < gripper_range_mm[0]:
        reason = RejectReason.UNDERSIZE
    elif cr is None or (apple is not None and apple.occluded):
        reason = RejectReason.OCCLUDED
    elif (apple is not None and apple.confidence < p.min_confidence) or \
            (light is not None and light.light_state == LightState.FAULT):
        reason = RejectReason.LOW_CONF
    else:
        grade, reason = decide(cr, p)

    if cr is not None and cr.inspect_mode == InspectMode.IN_HAND_MULTI:
        final = True
    elif reason in _ALWAYS_FINAL or policy == InspectionPolicy.NEVER:
        final = True
    elif policy == InspectionPolicy.ALWAYS:
        final = False
    else:
        final = reason not in _ADAPTIVE_RECHECK

    T = p.coloring_threshold
    return GradeResult(
        scene_id=cr.scene_id if cr else scene_id,
        apple_id=cr.apple_id if cr else apple_id,
        grade=grade,
        ratio=cr.ratio if cr else float("nan"),
        ratio_lo=cr.ratio_lo if cr else float("nan"),
        ratio_hi=cr.ratio_hi if cr else float("nan"),
        coverage=cr.coverage if cr else 0.0,
        margin=(cr.ratio - T) if cr else float("nan"),
        reject_reason=reason,
        final=final,
        inspect_mode=cr.inspect_mode if cr else InspectMode.TOP_SINGLE,
        variety=p.variety,
    )
