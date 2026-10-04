"""사과 구 모형 (착색비율 면적 가중용). 모든 좌표는 카메라 좌표계(m): x 오른쪽, y 아래, z 앞(시선 방향).

탑뷰에서는 윗반구만 보이므로 깊이 점에 구를 직접 맞추면 반지름이 불안정하다.
그래서 반지름은 마스크 크기(실루엣)와 윗면 깊이로 구하고, 깊이 점 구 피팅은 교차검증에만 쓴다 (설계서 M1.5).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Sphere:
    center: np.ndarray        # (3,) 카메라 좌표 m
    radius: float             # m
    fit_residual: float = 0.0 # 깊이 점과 구 표면의 RMS 차이 (m), 교차검증용


def pixel_rays(us: np.ndarray, vs: np.ndarray, intrinsics: tuple[float, float, float, float]) -> np.ndarray:
    """픽셀 → 단위 시선 벡터 (N×3)."""
    fx, fy, cx, cy = intrinsics
    d = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, dtype=np.float64)], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def sphere_from_mask(mask: np.ndarray, depth_m: np.ndarray, intrinsics: tuple[float, float, float, float],
                     top_percentile: float = 5.0) -> Sphere | None:
    """마스크 실루엣 + 윗면 깊이로 구를 추정한다.

    실루엣 반지름 r_px, 윗면까지 거리 z_top 이면 사과 중심 거리는 z_top + r 이고
    r = r_px · (z_top + r) / f  →  r = r_px · z_top / (f − r_px)
    """
    vs, us = np.nonzero(mask)
    if len(us) < 30:
        return None
    d = depth_m[vs, us]
    d = d[d > 0]
    if len(d) < 10:
        return None
    fx, fy, cx, cy = intrinsics
    f = (fx + fy) / 2
    r_px = np.sqrt(len(us) / np.pi)
    z_top = float(np.percentile(d, top_percentile))
    r = r_px * z_top / (f - r_px)
    ray = pixel_rays(np.array([us.mean()]), np.array([vs.mean()]), intrinsics)[0]
    center = ray * (z_top + r) / ray[2]           # 중심의 z = z_top + r
    sph = Sphere(center, float(r))
    sph.fit_residual = depth_residual(sph, mask, depth_m, intrinsics)
    return sph


def depth_residual(sph: Sphere, mask: np.ndarray, depth_m: np.ndarray, intrinsics) -> float:
    """측정 깊이와 구 앞면 깊이의 RMS 차이. 크면 마스크나 구 추정이 틀린 것."""
    vs, us = np.nonzero(mask & (depth_m > 0))
    if len(us) == 0:
        return float("nan")
    t, ok = ray_sphere(pixel_rays(us, vs, intrinsics), sph)
    if ok.sum() == 0:
        return float("nan")
    z_model = t[ok] * pixel_rays(us[ok], vs[ok], intrinsics)[:, 2]
    return float(np.sqrt(np.mean((depth_m[vs[ok], us[ok]] - z_model) ** 2)))


def ray_sphere(rays: np.ndarray, sph: Sphere) -> tuple[np.ndarray, np.ndarray]:
    """시선과 구 앞면의 교점 거리 t (N,), 교차 여부 (N,)."""
    dc = rays @ sph.center
    disc = dc ** 2 - (sph.center @ sph.center - sph.radius ** 2)
    ok = disc >= 0
    t = dc - np.sqrt(np.where(ok, disc, 0.0))
    return t, ok


def fit_sphere_lsq(points: np.ndarray) -> Sphere:
    """깊이 점군에 대한 대수적 최소제곱 구 피팅 (교차검증·인핸드 뷰용)."""
    A = np.c_[2 * points, np.ones(len(points))]
    b = (points ** 2).sum(axis=1)
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    c = sol[:3]
    r = float(np.sqrt(max(sol[3] + c @ c, 0.0)))
    res = float(np.sqrt(np.mean((np.linalg.norm(points - c, axis=1) - r) ** 2)))
    return Sphere(c, r, res)
