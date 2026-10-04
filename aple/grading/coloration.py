"""착색비율 산출 (S5).

픽셀 판정: 색상각 h = atan2(b*, a*) <= hue_max  그리고  채도 C >= chroma_min  → 착색
면적 가중: 가장자리 픽셀의 면적 축소를 법선으로 보정 (S5-3)
다뷰 합산: top + front (S5-4)
"""

from __future__ import annotations

import numpy as np

from aple.schema import AppleObservation


def colored_ratio(
    color_bgr: np.ndarray,
    obs: AppleObservation,
    hue_max_deg: float,
    chroma_min: float,
) -> float:
    """한 시점에서의 착색비율 (0~100)."""
    raise NotImplementedError


def combine_views(ratio_by_view: dict[str, float], weight_by_view: dict[str, float]) -> float:
    """뷰별 착색비율 합산."""
    raise NotImplementedError
