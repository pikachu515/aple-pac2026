"""착색비율 (설계서 apple_crop_extractor, color_segmentation_node, surface_area_corrector). 필수.

판정 카메라는 D435 하나 (전략 S5-4). 웹캠은 판정에 쓰지 않는다.

픽셀 판정   h = atan2(b*, a*) ∈ hue_range  AND  C* ≥ min_chroma  → 착색
구 법선     n_i = (p_i − c) / r
경사        cosθ_i = −n_i · v_i                     (v_i: 카메라→픽셀 시선)
면적 요소   dA_i = (z_i / f)² / cosθ_i,  cosθ_i ≥ cos_theta_min 만 포함
ratio_visible = Σ dA_i · colored_i / Σ dA_i
coverage      = Σ dA_i / (4π r²)                    (탑뷰 단일 ≈ 0.25)
"""

from __future__ import annotations

from aple.schema import AppleDetection2D, AppleState, ColorRatio, SceneSnapshot


def color_ratio(snap: SceneSnapshot, det: AppleDetection2D, apple: AppleState, cfg: dict) -> ColorRatio:
    """모드 A: 탑뷰 단일 뷰. ratio_lo/hi 는 탑뷰–전표면 관계의 실측 오차로 정한다."""
    raise NotImplementedError


def accumulate_views(views: list[tuple[SceneSnapshot, AppleDetection2D]], cfg: dict) -> ColorRatio:
    """모드 B (확장): 파지 후 기울임 + J6 회전 다면 뷰를 구면 지도에 누적 (surface_map_accumulator)."""
    raise NotImplementedError
