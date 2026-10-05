"""M2 진입점 — 스냅샷 한 장의 사과 전부를 판정한다 (공정 관리 M4 가 부른다).

입력은 M1 의 결과(검출 + 3D 상태)와 조명 상태, 출력은 사과별 GradeResult.
같은 scene_id 끼리만 결합한다 (설계서 1.4) — 다르면 판정하지 않고 오류를 낸다.
"""

from __future__ import annotations

from aple.inspection.coloration import InspectParams, measure_view, ratio_from_view
from aple.inspection.grade_classifier import classify
from aple.schema import (AppleDetection2D, AppleState, GradeResult, InspectionPolicy, LightingStatus,
                         SceneSnapshot)


class SceneMismatch(ValueError):
    """검출·3D 상태가 다른 스냅샷에서 왔다 (FailureCode.SCENE_MISMATCH 에 해당)."""


def inspect_scene(snap: SceneSnapshot, dets: list[AppleDetection2D], apples: list[AppleState] | None,
                  p: InspectParams, *, light: LightingStatus | None = None,
                  policy: InspectionPolicy = InspectionPolicy.NEVER,
                  gripper_range_mm: tuple[float, float] = (30.0, 100.0)) -> list[GradeResult]:
    """apples 가 None 이면 (역할 A 3D 상태 없이 단독 실험) 검출 번호를 사과 번호로 쓴다."""
    for d in dets:
        if d.scene_id != snap.scene_id:
            raise SceneMismatch(f"검출 scene {d.scene_id} ≠ 스냅샷 scene {snap.scene_id}")
    by_det = {d.det_id: d for d in dets}

    def ratio(det, apple_id):
        """그림자가 많이 진 사과는 착색비율을 내지 않는다 → OCCLUDED(재촬영). 비율을 고치면 치우친다."""
        vm = measure_view(snap.color, snap.depth_m, det.mask, snap.intrinsics, p)
        if vm is None or vm.shadow_fraction > p.max_shadow_fraction:
            return None
        return ratio_from_view(vm, p, snap.scene_id, apple_id)

    if apples is None:
        return [classify(ratio(d, d.det_id), p, light=light, policy=policy,
                         gripper_range_mm=gripper_range_mm, scene_id=snap.scene_id, apple_id=d.det_id)
                for d in dets]

    results = []
    for a in apples:
        if a.scene_id != snap.scene_id:
            raise SceneMismatch(f"사과 {a.apple_id} scene {a.scene_id} ≠ 스냅샷 scene {snap.scene_id}")
        det = by_det.get(a.det_id)
        cr = None if (det is None or a.occluded) else ratio(det, a.apple_id)
        results.append(classify(cr, p, apple=a, light=light, policy=policy, gripper_range_mm=gripper_range_mm,
                                scene_id=snap.scene_id, apple_id=a.apple_id))
    return results
