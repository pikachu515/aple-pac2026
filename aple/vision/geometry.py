"""깊이 기반 기하 측정 (S4-3, S4-4)."""

from __future__ import annotations

import numpy as np

from aple.schema import AppleGeometry, AppleObservation, BoxState


def measure_apples(
    obs: list[AppleObservation],
    depth_mm: np.ndarray,
    intrinsics: dict,
    T_robot_cam: np.ndarray,
) -> list[AppleGeometry]:
    """마스크 영역 깊이 → 사과 중심 위치(로봇 좌표, mm)와 직경.

    - 깊이는 픽셀 단위로 믿지 않고 마스크 영역 통계(중앙값)로 쓴다
    - 결측이 많으면 구 피팅으로 보완, 유효 비율은 obs.depth_valid_ratio 로 기록
    """
    raise NotImplementedError


def measure_boxes(depth_mm: np.ndarray, box_rois: dict, grid: dict) -> list[BoxState]:
    """상자 ROI 깊이 분포 → 벽 높이, 셀별 적재 높이, 빈 셀 (S7-2, S7-3)."""
    raise NotImplementedError
