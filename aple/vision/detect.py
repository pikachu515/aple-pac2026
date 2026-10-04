"""사과 검출 (S4-2, S4-6). Piper Studio YOLO 루프로 학습한 가중치를 사용한다."""

from __future__ import annotations

import numpy as np

from aple.schema import AppleObservation


def detect_apples(color_bgr: np.ndarray, depth_mm: np.ndarray, weights: str, conf: float) -> list[AppleObservation]:
    """경계상자 검출 → 상자 내부를 깊이(트레이 면보다 솟은 영역)로 분리해 마스크 생성.

    트레이 면과 높이 차가 없는 검출은 오검출로 제외한다 (S4-6).
    """
    raise NotImplementedError
