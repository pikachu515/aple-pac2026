"""M1 공통 기하 — 받침면 평면, 높이 지도, base_link 영역의 영상 투영, 사과 구 추정. 역할 A.

모든 함수는 순수 계산이며 장치·설정 파일에 접근하지 않는다.
카메라 좌표: x 오른쪽, y 아래, z 시선 방향 (m). 깊이 0 = 결측.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aple.hal.calibration import backproject, base_to_pixel


# ================================================================ 점군 · 평면

def depth_points(depth_m: np.ndarray, intrinsics) -> np.ndarray:
    """H×W×3 카메라 좌표. 결측 픽셀은 (0, 0, 0)."""
    v, u = np.indices(depth_m.shape)
    return backproject(u, v, depth_m, intrinsics)


def fit_plane_ransac(points: np.ndarray, iters: int = 300, tol_m: float = 0.004,
                     seed: int = 0) -> tuple[np.ndarray, float]:
    """RANSAC 평면 n·p + d = 0. n 은 원점(카메라) 쪽을 향하게 맞춘다 (d > 0).

    마지막에 인라이어로 최소제곱 재추정한다 (깊이 양자화 잡음 평균).
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 50:
        raise ValueError("평면을 맞출 유효 깊이가 부족하다 — 카메라 높이·조명·USB 3 연결 확인")
    rng = np.random.default_rng(seed)
    best_cnt, best = -1, None
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-12:
            continue
        n /= norm
        d = -n @ a
        cnt = int(np.count_nonzero(np.abs(pts @ n + d) < tol_m))
        if cnt > best_cnt:
            best_cnt, best = cnt, (n, d)
    if best is None:
        raise ValueError("평면을 찾지 못했다")
    n, d = best
    inl = pts[np.abs(pts @ n + d) < tol_m]
    if len(inl) >= 3:
        c = inl.mean(axis=0)
        _, _, vt = np.linalg.svd(inl - c, full_matrices=False)
        n = vt[2]
        d = -n @ c
    if d < 0:
        n, d = -n, -d
    return n, float(d)


def height_above_plane(pts: np.ndarray, valid: np.ndarray, n: np.ndarray, d: float) -> np.ndarray:
    """평면에서 카메라 쪽으로 솟은 높이 (m). 결측은 0."""
    h = pts @ n + d
    h[~valid] = 0.0
    return h


# ================================================================ base_link 영역 → 영상 마스크

def rect_base(center_xy, size_xy) -> np.ndarray:
    """base_link 축 정렬 직사각형의 네 꼭짓점 (x, y)."""
    cx, cy = center_xy
    hx, hy = size_xy[0] / 2, size_xy[1] / 2
    return np.array([[cx - hx, cy - hy], [cx + hx, cy - hy], [cx + hx, cy + hy], [cx - hx, cy + hy]])


def project_polygon_mask(corners_xy: np.ndarray, z_m: float, shape: tuple[int, int], intrinsics,
                         T_base_camera: np.ndarray) -> np.ndarray | None:
    """높이 z 의 base_link 다각형을 영상에 칠한 마스크. 카메라 뒤면 None."""
    pts = np.column_stack([corners_xy, np.full(len(corners_xy), z_m)])
    uv, zc = base_to_pixel(pts, intrinsics, T_base_camera)
    if np.any(zc <= 0) or not np.all(np.isfinite(uv)):
        return None
    mask = np.zeros(shape, np.uint8)
    cv2.fillPoly(mask, [np.round(uv).astype(np.int32)], 1)
    return mask.astype(bool)


# ================================================================ 사과 구 추정

@dataclass
class SphereEstimate:
    center_cam: np.ndarray        # (3,) m
    radius_m: float               # 실루엣 반지름 (주 추정)
    radius_depth_m: float         # 깊이 점만으로 맞춘 반지름 (교차검증, 실패 시 nan)
    fit_residual_m: float         # 깊이 점과 구 표면의 RMS 거리
    silhouette_ok: bool           # 실루엣 경계가 충분히 보였는가 (가장자리·이웃에 막히지 않음)
    n_points: int                 # 구 거리 추정에 쓴 유효 깊이 점 수


def _unit_rays(us, vs, intrinsics) -> np.ndarray:
    d = backproject(us, vs, np.ones(len(us)), intrinsics)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def _cone_from_rays(rays: np.ndarray) -> tuple[np.ndarray, float] | None:
    """경계 시선들이 같은 원뿔 위에 있다: rays·w = 1, c = w/|w|, cosα = 1/|w|. 원근 왜곡에 정확하다."""
    if len(rays) < 12:
        return None
    w, *_ = np.linalg.lstsq(rays, np.ones(len(rays)), rcond=None)
    norm = np.linalg.norm(w)
    if norm <= 1.0:
        return None
    return w / norm, float(np.arccos(1.0 / norm))


def sphere_from_silhouette(mask: np.ndarray, depth_m: np.ndarray, intrinsics, *,
                           blocked: np.ndarray | None = None, inlier_tol_m: float = 0.003,
                           min_free_boundary: float = 0.5, min_points: int = 20) -> SphereEstimate | None:
    """마스크 실루엣(원뿔) + 마스크 내부 깊이로 구를 추정한다.

    1. 실루엣: 마스크 안쪽 경계와 바깥 경계 픽셀의 시선으로 원뿔(중심 방향 c, 반각 α)을 맞추고
       두 결과를 평균한다 (경계 픽셀화 반 픽셀 편향 상쇄). 이웃 사과·영상 가장자리에 닿은 경계는 뺀다.
    2. 거리: 표면 점 p 마다 |p − D·c| = D·sinα 를 D 에 대해 풀고 중앙값 → RANSAC 식 인라이어 재추정.
    3. 반지름 r = D·sinα. 깊이 점만으로 맞춘 반지름(축 고정 2변수 구 피팅)은 교차검증용으로 따로 낸다.

    탑뷰에서 깊이 점은 윗반구 일부만 덮어 자유 구 피팅은 반지름이 불안정하다. 실루엣은 깊이 결측과
    무관하게 선명하므로 반지름의 주 근거로 쓴다 (설계서 M1.5 교차검증).
    """
    m = mask.astype(np.uint8)
    H, W = m.shape
    if m.sum() < 30:
        return None
    k3 = np.ones((3, 3), np.uint8)
    inner = m.astype(bool) & ~cv2.erode(m, k3, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
    outer = cv2.dilate(m, k3).astype(bool) & ~m.astype(bool)
    edge = np.zeros((H, W), bool)
    edge[:2, :] = edge[-2:, :] = True
    edge[:, :2] = edge[:, -2:] = True
    bad = edge.copy()
    if blocked is not None:
        bad |= cv2.dilate(blocked.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    n_inner_all = int(inner.sum())
    inner &= ~bad
    outer &= ~bad
    free = inner.sum() / max(n_inner_all, 1)

    cones = []
    for ring in (inner, outer):
        vs, us = np.nonzero(ring)
        cone = _cone_from_rays(_unit_rays(us.astype(float), vs.astype(float), intrinsics))
        if cone is not None:
            cones.append(cone)
    if len(cones) == 0:
        return None
    c = sum(cn[0] for cn in cones)
    c /= np.linalg.norm(c)
    alpha = float(np.mean([cn[1] for cn in cones]))
    silhouette_ok = len(cones) == 2 and free >= min_free_boundary

    vs, us = np.nonzero(mask & (depth_m > 0))
    if len(us) < min_points:
        return SphereEstimate(c, float("nan"), float("nan"), float("nan"), silhouette_ok, int(len(us)))
    p = backproject(us, vs, depth_m[vs, us], intrinsics)
    pr = p / np.linalg.norm(p, axis=1, keepdims=True)
    core = pr @ c > np.cos(0.8 * alpha)                # 가장자리(경사 급하고 깊이 잡음 큰 곳) 제외
    if core.sum() >= min_points:
        p = p[core]
    sa, ca2 = np.sin(alpha), np.cos(alpha) ** 2
    pc = p @ c
    pp = np.einsum("ij,ij->i", p, p)
    D_i = (pc + np.sqrt(np.clip(pc ** 2 - ca2 * pp, 0, None))) / ca2
    D = float(np.median(D_i))
    for _ in range(2):                                  # 인라이어로 다시
        resid = np.linalg.norm(p - D * c, axis=1) - D * sa
        inl = np.abs(resid) < max(inlier_tol_m, 1e-4)
        if inl.sum() < min_points:
            break
        D = float(np.median(D_i[inl]))
    r = D * sa
    center = D * c
    resid = np.linalg.norm(p - center, axis=1) - r
    inl = np.abs(resid) < 3 * inlier_tol_m
    fit_rms = float(np.sqrt(np.mean(resid[inl] ** 2))) if inl.any() else float("inf")

    # 교차검증: 축 c 는 고정, (D, D²−r²) 를 깊이 점만으로 선형 최소제곱
    q = p[inl] if inl.sum() >= min_points else p
    A = np.column_stack([2 * (q @ c), -np.ones(len(q))])
    b = np.einsum("ij,ij->i", q, q)
    (Dd, K), *_ = np.linalg.lstsq(A, b, rcond=None)
    r_depth = float(np.sqrt(Dd ** 2 - K)) if Dd ** 2 > K else float("nan")
    return SphereEstimate(center, float(r), r_depth, fit_rms, silhouette_ok, int(len(p)))
