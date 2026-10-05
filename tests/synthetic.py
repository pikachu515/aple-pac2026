"""정답을 아는 합성 장면 — 테이블 위 구형 사과를 카메라로 내려다본 컬러·깊이 영상.

카메라 좌표: x 오른쪽, y 아래, z 앞. 테이블은 z = table_z 평면. 사과마다 착색 패턴을
'사과 좌표 방향 → 착색 여부' 함수로 주므로 전표면·관측 면 착색비율 정답을 계산할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from aple.inspection.color import linear_to_srgb, srgb_to_linear

INTR = (600.0, 600.0, 320.0, 240.0)
SIZE = (480, 640)

RED = srgb_to_linear(np.array([175, 30, 38]) / 255)        # 착색부
YELLOW = srgb_to_linear(np.array([205, 190, 75]) / 255)    # 황록색 바탕
TABLE = srgb_to_linear(np.array([150, 150, 150]) / 255)
CARD = srgb_to_linear(np.array([118, 118, 118]) / 255)     # 18% 회색


@dataclass
class Apple:
    center: tuple[float, float, float]
    radius: float
    pattern: Callable[[np.ndarray], np.ndarray]           # (N×3 사과 좌표 방향) → bool
    R_cam_apple: np.ndarray = field(default_factory=lambda: np.eye(3))


def half_red(n: np.ndarray) -> np.ndarray:
    return n[:, 0] > 0


def top_cap(cos_limit: float) -> Callable[[np.ndarray], np.ndarray]:
    """카메라 쪽(사과 좌표 −z) 꼭대기 원형 영역만 착색."""
    return lambda n: -n[:, 2] > cos_limit


def side_band(x0: float) -> Callable[[np.ndarray], np.ndarray]:
    return lambda n: n[:, 0] > x0


def blush(width: float = 1.2) -> Callable[[np.ndarray], np.ndarray]:
    """실제 사과처럼 적색에서 황색으로 서서히 바뀌는 착색 (0~1 섞임 비율)."""
    return lambda n: np.clip((n[:, 0] + width / 2) / width, 0.0, 1.0)


def full_truth(pattern, samples: int = 400_000, seed: int = 0) -> float:
    v = np.random.default_rng(seed).normal(size=(samples, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return float(pattern(v).mean())


def _visible(apple: Apple, cos_min: float, samples: int, seed: int):
    """원근 카메라 기준: 표면 점마다 그 점으로 가는 시선과 법선의 cosθ ≥ cos_min 인 영역."""
    v = np.random.default_rng(seed).normal(size=(samples, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)          # 사과 좌표 방향
    n_cam = v @ apple.R_cam_apple.T
    p = np.asarray(apple.center) + apple.radius * n_cam
    ray = p / np.linalg.norm(p, axis=1, keepdims=True)
    return v, -(n_cam * ray).sum(axis=1) >= cos_min


def visible_truth(apple: Apple, cos_min: float = 0.5, samples: int = 400_000, seed: int = 0) -> float:
    """관측 영역(넓이 기준)의 착색비율 정답."""
    v, vis = _visible(apple, cos_min, samples, seed)
    return float(apple.pattern(v[vis]).mean())


def coverage_truth(apple: Apple, cos_min: float = 0.5, samples: int = 400_000, seed: int = 0) -> float:
    """관측 영역 / 전체 표면. 카메라가 무한히 멀면 0.25, 0.6 m 거리 r 40 mm 이면 약 0.226."""
    return float(_visible(apple, cos_min, samples, seed)[1].mean())


def render(apples: list[Apple], *, table_z: float = 0.65, tint=(1.0, 1.0, 1.0),
           card_rect: tuple[int, int, int, int] | None = None, depth_dropout: float = 0.0,
           highlight_on: int | None = None, color_noise: float = 0.0, depth_noise_m: float = 0.0,
           shadow_rect: tuple[int, int, int, int] | None = None, shadow_factor: float = 0.08,
           intr=INTR, size=SIZE, seed: int = 0):
    """→ (color BGR uint8, depth_m, [정답 마스크])"""
    H, W = size
    fx, fy, cx, cy = intr
    v, u = np.indices((H, W))
    rays = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, float)], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    t_table = table_z / rays[..., 2]
    best_t = t_table.copy()
    who = np.full((H, W), -1)
    for i, a in enumerate(apples):
        c = np.asarray(a.center)
        dc = rays @ c
        disc = dc ** 2 - (c @ c - a.radius ** 2)
        t = dc - np.sqrt(np.clip(disc, 0, None))
        hit = (disc >= 0) & (t < best_t)
        best_t[hit] = t[hit]
        who[hit] = i

    lin = np.broadcast_to(TABLE, (H, W, 3)).copy()
    if card_rect is not None:
        x, y, w, h = card_rect
        lin[y:y + h, x:x + w] = CARD
    masks = []
    for i, a in enumerate(apples):
        m = who == i
        masks.append(m)
        p = rays[m] * best_t[m][:, None]
        n_cam = (p - np.asarray(a.center)) / a.radius
        n_apple = n_cam @ a.R_cam_apple
        col = np.asarray(a.pattern(n_apple), dtype=float)          # bool(착색 여부) 또는 0~1(적색 섞임 정도)
        shade = 0.45 + 0.55 * np.clip(-(n_cam * rays[m]).sum(axis=1), 0, 1)   # 밝기만 바꾸고 색상각은 유지
        lin[m] = (YELLOW + (RED - YELLOW) * col[:, None]) * shade[:, None]
    lin *= np.asarray(tint)
    if shadow_rect is not None:           # 팔 등이 드리운 그림자 (사과·테이블 모두 어두워짐)
        x, y, w, h = shadow_rect
        lin[y:y + h, x:x + w] *= shadow_factor
    if highlight_on is not None:          # 사과 꼭대기 근처 정반사 하이라이트
        m = masks[highlight_on]
        cyx = np.argwhere(m).mean(axis=0)
        hl = m & ((v - cyx[0]) ** 2 + (u - cyx[1]) ** 2 < 8 ** 2)
        lin[hl] = 0.95
    rng = np.random.default_rng(seed + 1)
    srgb = linear_to_srgb(lin)[..., ::-1] * 255
    if color_noise > 0:
        srgb = srgb + rng.normal(0, color_noise, srgb.shape)
    color = np.clip(srgb + 0.5, 0, 255).astype(np.uint8)

    depth = (best_t * rays[..., 2]).astype(np.float32)
    if depth_noise_m > 0:
        depth = depth + rng.normal(0, depth_noise_m, depth.shape).astype(np.float32)
    if depth_dropout > 0:
        drop = np.random.default_rng(seed).random((H, W)) < depth_dropout
        depth[drop & (who >= 0)] = 0.0
    return color, depth, masks


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
