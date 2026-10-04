"""사과 3D 위치·직경 (설계서 apple_pose_estimator, apple_tracker 축약). 필수.

마스크 내 깊이 점군에 RANSAC + 최소제곱 구 피팅 → 중심·반지름.
탑뷰는 윗반구만 보이므로 반지름을 마스크 픽셀 직경(깊이로 m 환산)과 교차검증한다.
깊이 결측이 많으면 confidence 를 낮추고 depth_valid_ratio 에 기록한다 (G1).
apple_id 는 scene 간 최근접 매칭으로 유지한다 (displacement_m).
"""

from __future__ import annotations

import numpy as np

from aple.schema import AppleDetection2D, AppleState, SceneSnapshot


def estimate(snap: SceneSnapshot, dets: list[AppleDetection2D], T_base_camera: np.ndarray,
             prev: list[AppleState] | None, cfg: dict) -> list[AppleState]:
    raise NotImplementedError
