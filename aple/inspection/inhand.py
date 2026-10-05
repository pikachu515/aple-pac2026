"""모드 B 다면 검사의 자세 계산 — 로봇 자세로부터 "카메라가 사과의 어느 면을 보는가".

파지 중인 사과는 TCP 에 고정되어 함께 움직인다. 그래서 사과 좌표계 = TCP 좌표계로 두면
  R_cam_apple = R_base_cam^T · R_base_tcp
  사과 중심(카메라) = T_base_cam^-1 · T_base_tcp · apple_in_tcp
가 된다. 이 둘이 있으면 각 뷰의 픽셀을 사과 표면의 같은 칸에 모을 수 있다 (coloration.accumulate_views).

- T_base_cam: 카메라–로봇 캘리브레이션 결과 (역할 A, aple/hal/calibration.py)
- T_base_tcp: 각 촬영 시점의 TCP 자세 (역할 B, 로봇 관절 상태 → 순기구학)
- apple_in_tcp: 파지 직후 측정한 TCP 기준 사과 중심 (집기 전 탑뷰 위치 → TCP 좌표로 변환)

모든 변환은 4×4 동차 행렬, 단위 m.
"""

from __future__ import annotations

import numpy as np

from aple.inspection.coloration import InHandView, fibonacci_sphere
from aple.inspection.geometry import Sphere
from aple.schema import SceneSnapshot


def inv(T: np.ndarray) -> np.ndarray:
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def rot_cam_apple(T_base_cam: np.ndarray, T_base_tcp: np.ndarray) -> np.ndarray:
    return T_base_cam[:3, :3].T @ T_base_tcp[:3, :3]


def apple_center_cam(T_base_cam: np.ndarray, T_base_tcp: np.ndarray, apple_in_tcp: np.ndarray) -> np.ndarray:
    p = np.append(np.asarray(apple_in_tcp, float), 1.0)
    return (inv(T_base_cam) @ T_base_tcp @ p)[:3]


def apple_in_tcp_from_base(T_base_tcp: np.ndarray, center_base: np.ndarray) -> np.ndarray:
    """파지 시점의 사과 중심(base, 탑뷰 측정값) → TCP 좌표. 이후 모든 검사 뷰에서 그대로 쓴다."""
    return (inv(T_base_tcp) @ np.append(np.asarray(center_base, float), 1.0))[:3]


def make_view(snap: SceneSnapshot, mask: np.ndarray, T_base_cam: np.ndarray, T_base_tcp: np.ndarray,
              apple_in_tcp: np.ndarray, radius_m: float) -> InHandView:
    """검사 자세 한 장 → 누적용 뷰. 구 중심은 실루엣이 아니라 로봇 자세로 정한다 (핑거가 가려도 정확)."""
    c = apple_center_cam(T_base_cam, T_base_tcp, apple_in_tcp)
    return InHandView(snap.color, snap.depth_m, snap.intrinsics, mask,
                      rot_cam_apple(T_base_cam, T_base_tcp), Sphere(c, float(radius_m)))


def expected_coverage(R_cam_apple_list: list[np.ndarray], center_dirs: list[np.ndarray] | None = None,
                      cos_min: float = 0.5, n: int = 20000) -> float:
    """검사 자세들로 볼 수 있는 표면 비율 (핑거 가림 제외). 검사 동작을 설계할 때 각도를 고르는 데 쓴다.

    center_dirs: 카메라→사과 중심 방향 (없으면 광축 정면으로 본다).
    """
    pts = fibonacci_sphere(n)                                     # 사과 좌표 표면 방향
    seen = np.zeros(n, bool)
    for i, R in enumerate(R_cam_apple_list):
        view = np.array([0.0, 0.0, 1.0]) if center_dirs is None else np.asarray(center_dirs[i], float)
        view = view / np.linalg.norm(view)
        seen |= (pts @ R.T) @ (-view) >= cos_min                  # 법선이 카메라 쪽을 향하는 정도
    return float(seen.mean())
