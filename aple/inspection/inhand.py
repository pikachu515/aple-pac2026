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

from dataclasses import dataclass

import numpy as np

from aple.inspection.coloration import InHandView, fibonacci_sphere
from aple.inspection.geometry import Sphere, pixel_rays
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


# ---------------------------------------------------------------- 미끄러짐 검사

@dataclass(frozen=True)
class SlipCheck:
    """로봇 자세로 예측한 사과 중심 vs 깊이로 본 사과 중심 (카메라 좌표, m)."""
    predicted_center: np.ndarray
    observed_center: np.ndarray | None   # 깊이 점이 부족하면 None
    mahalanobis: float                   # 관측이 없으면 nan
    fit_rms_m: float
    num_points: int
    ok: bool                             # False 면 미끄러짐 또는 검증 불가 → 자세 기반 구를 믿지 않는다


class InHandSlip(ValueError):
    """모드 B 촬영 중 사과가 파지 때와 다른 위치에 있다 → 재판정 중단, 재파지 대상."""

    def __init__(self, view_index: int, check: SlipCheck) -> None:
        if check.observed_center is None:
            detail = f"only {check.num_points} depth points, cannot verify the held apple position"
        else:
            offset_mm = 1000 * float(np.linalg.norm(check.observed_center - check.predicted_center))
            detail = f"Mahalanobis {check.mahalanobis:.2f}, offset {offset_mm:.1f} mm"
        super().__init__(f"apple slip suspected at in-hand view {view_index}: {detail}")
        self.view_index = view_index
        self.check = check


def fit_center_fixed_radius(points: np.ndarray, radius_m: float, init: np.ndarray,
                            iters: int = 20) -> tuple[np.ndarray, np.ndarray, float]:
    """반지름을 고정하고 깊이 점에 구 중심을 맞춘다 (Gauss-Newton).

    핑거가 옆을 가려 보이는 면이 작아도 반지름을 알고 있으므로 중심이 흔들리지 않는다.
    → (중심, (JᵀJ)⁻¹, 잔차 RMS)
    """
    c = np.asarray(init, float).copy()
    for _ in range(iters):
        d = points - c
        n = np.linalg.norm(d, axis=1)
        J = -d / n[:, None]
        step, *_ = np.linalg.lstsq(J, -(n - radius_m), rcond=None)
        c += step
        if np.linalg.norm(step) < 1e-7:
            break
    d = points - c
    n = np.linalg.norm(d, axis=1)
    J = -d / n[:, None]
    rms = float(np.sqrt(np.mean((n - radius_m) ** 2)))
    return c, np.linalg.pinv(J.T @ J), rms


def check_slip(depth_m: np.ndarray, mask: np.ndarray, intrinsics: tuple[float, float, float, float],
               predicted_center: np.ndarray, radius_m: float, *, sigma_lateral_m: float,
               sigma_axial_m: float, sigma_pose_m: float, max_mahalanobis: float,
               min_points: int = 200) -> SlipCheck:
    """관측 중심과 예측 중심의 마할라노비스 거리로 미끄러짐을 판단한다.

    관측 공분산: 카메라 시선 방향(깊이)은 sigma_axial, 옆 방향은 sigma_lateral 에
                 피팅 자체의 불확실성 rms²·(JᵀJ)⁻¹ 을 더한다
    예측 공분산: 카메라–로봇 보정·순기구학·파지 측정 오차를 sigma_pose 로 등방 근사
    d = sqrt(rᵀ (S_obs + S_pred)⁻¹ r) 가 max_mahalanobis 를 넘으면 미끄러짐.
    """
    pred = np.asarray(predicted_center, float)
    vs, us = np.nonzero(mask & (depth_m > 0))
    if len(us) < min_points:
        return SlipCheck(pred, None, float("nan"), float("nan"), int(len(us)), False)
    rays = pixel_rays(us, vs, intrinsics)
    points = rays * (depth_m[vs, us] / rays[:, 2])[:, None]

    center, inv_jtj, rms = fit_center_fixed_radius(points, radius_m, pred)
    resid = np.abs(np.linalg.norm(points - center, axis=1) - radius_m)
    keep = resid <= 3 * max(rms, sigma_axial_m)                 # 가장자리 혼합 픽셀 등 이상점 1회 제거
    if keep.sum() >= min_points and not keep.all():
        points = points[keep]
        center, inv_jtj, rms = fit_center_fixed_radius(points, radius_m, center)

    u = center / np.linalg.norm(center)                          # 시선 방향
    S_obs = sigma_lateral_m ** 2 * np.eye(3) + (sigma_axial_m ** 2 - sigma_lateral_m ** 2) * np.outer(u, u)
    S_obs = S_obs + max(rms, 1e-6) ** 2 * inv_jtj
    S = S_obs + sigma_pose_m ** 2 * np.eye(3)
    r = center - pred
    d = float(np.sqrt(r @ np.linalg.solve(S, r)))
    return SlipCheck(pred, center, d, rms, int(len(points)), d <= max_mahalanobis)
