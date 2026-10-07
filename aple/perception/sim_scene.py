"""정답을 아는 합성 장면 (개발·시험·RViz 데모용). 역할 A.

base_link 기준으로 장면을 놓고, 위에서 내려다보는 카메라(T_base_camera)로 깊이·컬러를 그린다.
- 테이블: z = 0 평면
- tops: 수평 사각형 (x0, x1, y0, y1, z) — 상자 바닥·벽 윗면, 받침대
- spheres: ((x, y, z) 중심, 반지름) — 사과
깊이는 카메라 z (D435 와 같은 의미, 0 = 결측). 수직 면은 그리지 않는다 (탑뷰에서는 거의 안 보인다).
"""

from __future__ import annotations

import math

import numpy as np

INTR = (600.0, 600.0, 320.0, 240.0)
SIZE = (480, 640)

TABLE_BGR = (150, 150, 150)
TOP_BGR = (70, 120, 170)            # 골판지
APPLE_BGR = (40, 40, 175)


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def camera_looking_down(xy=(0.30, 0.0), height_m: float = 0.65, tilt_rad: float = 0.0) -> np.ndarray:
    """카메라를 (x, y, height) 에 아래로 향하게. 영상 오른쪽 = base −y, 영상 아래 = base −x."""
    R = np.column_stack([[0, -1, 0], [-1, 0, 0], [0, 0, -1]]).astype(float)
    T = np.eye(4)
    T[:3, :3] = rot_x(tilt_rad) @ R
    T[:3, 3] = [xy[0], xy[1], height_m]
    return T


def box_tops(origin, rows: int, cols: int, pitch: float, wall_h: float, wall_t: float = 0.01) -> list:
    """상자 바닥 + 네 벽 윗면 (workcell.yaml boxes 규약: 열 = +x, 행 = +y)."""
    x0, y0, z0 = origin
    x1, y1, zt = x0 + cols * pitch, y0 + rows * pitch, z0 + wall_h
    return [(x0, x1, y0, y1, z0),
            (x0 - wall_t, x1 + wall_t, y0 - wall_t, y0, zt), (x0 - wall_t, x1 + wall_t, y1, y1 + wall_t, zt),
            (x0 - wall_t, x0, y0, y1, zt), (x1, x1 + wall_t, y0, y1, zt)]


def render(T, spheres=(), tops=(), *, intr=INTR, size=SIZE, noise_m: float = 0.0, dropout: float = 0.0,
           drop_only=None, seed: int = 0):
    """→ (depth_m H×W float32, who H×W 사과 번호 또는 −1, color H×W×3 BGR uint8)."""
    H, W = size
    fx, fy, cx, cy = intr
    v, u = np.indices(size)
    dc = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones(size)], axis=-1)
    d = dc @ np.asarray(T)[:3, :3].T            # base 방향 (카메라 z 성분 = 1 → 매개변수 = 깊이)
    o = np.asarray(T)[:3, 3]
    best = (0.0 - o[2]) / d[..., 2]             # 테이블
    color = np.empty(size + (3,), np.float64)
    color[:] = TABLE_BGR
    for x0, x1, y0, y1, z in tops:
        s = (z - o[2]) / d[..., 2]
        p = o + d * s[..., None]
        hit = (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1) & (s < best)
        best[hit] = s[hit]
        color[hit] = TOP_BGR
    who = np.full(size, -1)
    for i, (c, r) in enumerate(spheres):
        oc = o - np.asarray(c)
        a = np.einsum("...i,...i", d, d)
        b = 2 * (d @ oc)
        disc = b * b - 4 * a * (oc @ oc - r * r)
        s = (-b - np.sqrt(np.clip(disc, 0, None))) / (2 * a)
        hit = (disc >= 0) & (s < best)
        best[hit] = s[hit]
        who[hit] = i
        if hit.any():
            pts = o + d[hit] * s[hit][:, None]
            n = (pts - np.asarray(c)) / r
            shade = 0.45 + 0.55 * np.clip(n[:, 2], 0, 1)            # 위에서 빛
            color[hit] = np.asarray(APPLE_BGR) * shade[:, None]
    rng = np.random.default_rng(seed)
    depth = best.astype(np.float32)
    if noise_m:
        depth += rng.normal(0, noise_m, size).astype(np.float32)
    if dropout:
        target = who >= 0 if drop_only is None else np.isin(who, drop_only)
        depth[target & (rng.random(size) < dropout)] = 0.0
    return depth, who, np.clip(color, 0, 255).astype(np.uint8)
