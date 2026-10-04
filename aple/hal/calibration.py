"""카메라–로봇 변환 (설계서 calibration_node). 역할 A.

체커보드 / ChArUco eye-to-hand 캘리브레이션 결과(4×4, base_link ← camera_top_optical_frame)를
configs/workcell.yaml 에 저장·로드한다. 스크립트 실행의 위치 정확도는 이 값에 직접 좌우된다.
TCP 오프셋(핑거 어댑터 반영)도 여기서 관리한다.
"""

from __future__ import annotations

import numpy as np


def load_T_base_camera(path: str) -> np.ndarray:
    raise NotImplementedError


def pixel_to_base(u: float, v: float, depth_m: float, intrinsics: tuple, T_base_camera: np.ndarray) -> np.ndarray:
    """픽셀 + 깊이 → base_link 좌표 (m)."""
    raise NotImplementedError
