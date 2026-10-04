"""역할 C 단독 실험용 사과 분할 — 역할 A의 검출기(aple/perception/detector.py)가 준비되기 전 임시 대체.

깊이로 테이블 평면을 찾고, 평면보다 솟은 덩어리를 사과로 본다. 색을 쓰지 않으므로
착색 판정 결과에 영향을 주지 않는다. 사과끼리 붙어 있으면 한 덩어리로 잡히므로 실험할 때는 띄워 놓는다.
"""

from __future__ import annotations

import cv2
import numpy as np

from aple.schema import AppleDetection2D


def fit_table_plane(depth_m: np.ndarray, intrinsics, step: int = 4, iters: int = 300,
                    tol_m: float = 0.004, seed: int = 0) -> tuple[np.ndarray, float]:
    """RANSAC 평면 n·p + d = 0 (n 은 카메라 쪽을 향함)."""
    fx, fy, cx, cy = intrinsics
    v, u = np.mgrid[0:depth_m.shape[0]:step, 0:depth_m.shape[1]:step]
    z = depth_m[v, u]
    ok = z > 0
    pts = np.stack([(u[ok] - cx) / fx * z[ok], (v[ok] - cy) / fy * z[ok], z[ok]], axis=1)
    if len(pts) < 100:
        raise ValueError("유효 깊이가 부족하다 — 카메라 높이·조명·케이블(USB 3) 확인")
    rng = np.random.default_rng(seed)
    best, best_n, best_d = -1, None, 0.0
    for _ in range(iters):
        a, b, c = pts[rng.choice(len(pts), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n /= norm
        d = -n @ a
        cnt = int((np.abs(pts @ n + d) < tol_m).sum())
        if cnt > best:
            best, best_n, best_d = cnt, n, d
    if best_d < 0:                     # 원점(카메라)이 평면의 + 쪽에 오도록
        best_n, best_d = -best_n, -best_d
    return best_n, float(best_d)


def height_map(depth_m: np.ndarray, intrinsics, n: np.ndarray, d: float) -> np.ndarray:
    """테이블 평면 위 높이 (m). 깊이 결측은 0."""
    fx, fy, cx, cy = intrinsics
    v, u = np.indices(depth_m.shape)
    z = depth_m
    pts = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], axis=-1)
    h = pts @ n + d
    h[z <= 0] = 0.0
    return h


def segment_apples(color_bgr: np.ndarray, depth_m: np.ndarray, intrinsics,
                   min_height_m: float = 0.012, diam_range_m: tuple[float, float] = (0.03, 0.12),
                   scene_id: int = 0) -> list[AppleDetection2D]:
    n, d = fit_table_plane(depth_m, intrinsics)
    h = height_map(depth_m, intrinsics, n, d)
    raw = (h > min_height_m).astype(np.uint8)
    # 순서가 중요하다: 깊이 결측 구멍(광택)을 먼저 메우고, 그다음 잡음 점을 지운다.
    # 반대로 하면 구멍이 흩어진 사과가 통째로 지워진다.
    raw = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    contours, _ = cv2.findContours(raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fx, fy, _, _ = intrinsics
    f = (fx + fy) / 2
    dets = []
    for cnt in contours:
        mask = np.zeros(raw.shape, np.uint8)
        cv2.drawContours(mask, [cnt], -1, 1, thickness=-1)
        area = int(mask.sum())
        if area < 50:
            continue
        zs = depth_m[(mask > 0) & (depth_m > 0)]
        if len(zs) == 0:
            continue
        diam = 2 * np.sqrt(area / np.pi) * float(np.median(zs)) / f
        if not (diam_range_m[0] <= diam <= diam_range_m[1]):
            continue
        x, y, w, hh = cv2.boundingRect(cnt)
        dets.append((y + hh / 2, x + w / 2, x, y, w, hh, mask.astype(bool)))

    return [AppleDetection2D(scene_id, i, (x, y, w, hh), 1.0, m)
            for i, (_, _, x, y, w, hh, m) in enumerate(reading_order(dets), start=1)]


def reading_order(dets: list[tuple]) -> list[tuple]:
    """위→아래 줄, 줄 안에서 왼→오 순서 (오버레이 번호와 실제 사과를 맞추기 쉽게).

    중심 높이가 사과 높이의 절반 이내로 차이 나면 같은 줄로 본다. 고정 간격으로 자르면
    나란히 놓인 사과가 경계를 사이에 두고 다른 줄로 갈라진다.
    """
    rows: list[list[tuple]] = []
    for t in sorted(dets, key=lambda t: t[0]):
        if rows and t[0] - rows[-1][0][0] <= 0.5 * max(t[5], rows[-1][0][5]):
            rows[-1].append(t)
        else:
            rows.append([t])
    return [t for row in rows for t in sorted(row, key=lambda t: t[1])]
