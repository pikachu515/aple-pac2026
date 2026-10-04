"""상자 칸별 적재 높이·점유 격자 (설계서 box_occupancy_estimator), 적재 검증 (placement_verifier).

box_occupancy: 필수. placement_verifier: 확장 — 적재 전후 칸 높이 차이로 안착·이탈 확인.
팔이 상자를 가리거나 유효 깊이가 부족하면 valid=False.
"""

from __future__ import annotations

import numpy as np

from aple.schema import BoxOccupancy, SceneSnapshot


def estimate(snap: SceneSnapshot, T_base_camera: np.ndarray, cfg: dict) -> list[BoxOccupancy]:
    raise NotImplementedError


def verify_placement(before: BoxOccupancy, after: BoxOccupancy, cell_index: int,
                     expected_top_z_m: float) -> tuple[bool, float]:
    """(성공 여부, 안착 오차 m). 확장."""
    raise NotImplementedError
