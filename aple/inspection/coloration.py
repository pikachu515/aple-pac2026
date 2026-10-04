"""착색비율 (설계서 apple_crop_extractor · color_segmentation_node · surface_area_corrector · surface_map_accumulator).

판정 카메라는 D435 하나 (전략 S5-4). 웹캠은 판정에 쓰지 않는다.

픽셀 판정   h = atan2(b*, a*) ∈ hue_range  AND  C* ≥ min_chroma  → 착색
            L* ≥ highlight.min_L AND C* < highlight.max_chroma → 정반사 하이라이트, 판정에서 제외
구 법선     n_i = (p_i − c) / r            (p_i: 시선과 구 앞면의 교점 — 깊이 결측에도 동작)
경사        cosθ_i = −n_i · v_i
면적 요소   dA_i = t_i² · v_z³ / (fx·fy) / cosθ_i,   cosθ_i ≥ cos_theta_min 만 포함
ratio_visible = Σ dA_i · colored_i / Σ dA_i
coverage      = Σ dA_i / (4π r²)          (탑뷰 단일 ≈ 0.25)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from aple.inspection.color import bgr_to_lab, colored_mask, hue_chroma
from aple.inspection.geometry import Sphere, pixel_rays, ray_sphere, sphere_from_mask
from aple.schema import AppleDetection2D, ColorRatio, InspectMode, SceneSnapshot


# ---------------------------------------------------------------- 파라미터

@dataclass
class InspectParams:
    variety: str
    coloring_threshold: float          # '상' 기준 T (0~1)
    delta_up: float
    delta_mid: float
    hue_range_deg: tuple[float, float]
    min_chroma: float
    cos_theta_min: float = 0.5
    highlight_min_L: float = 90.0
    highlight_max_chroma: float = 12.0
    mask_erode_px: int = 2
    top_view_error: float = 0.10       # 탑뷰 ↔ 전표면 오차 (실측 전 임시값)
    min_coverage: float = 0.15
    max_view_disagreement: float = 0.25
    min_confidence: float = 0.5
    surface_bins: int = 600
    surface_margin: float = 0.02       # 구면 칸 이산화 오차 (판정 구간 여유)


def params_from_config(cfg: dict, variety: str | None = None) -> InspectParams:
    """configs/varieties.yaml (+ 병합된 설정 dict) → InspectParams."""
    ins = cfg["inspection"]
    variety = variety or ins["default_variety"]
    v = cfg["varieties"][variety]
    hl = ins.get("highlight", {})
    return InspectParams(
        variety=variety,
        coloring_threshold=v["coloring_threshold"],
        delta_up=v["delta_up"],
        delta_mid=v["delta_mid"],
        hue_range_deg=tuple(v["hue_range_deg"]),
        min_chroma=v["min_chroma"],
        cos_theta_min=ins.get("cos_theta_min", 0.5),
        highlight_min_L=hl.get("min_L", 90.0),
        highlight_max_chroma=hl.get("max_chroma", 12.0),
        mask_erode_px=ins.get("mask_erode_px", 2),
        top_view_error=ins.get("top_view_error", 0.10),
        min_coverage=ins.get("min_coverage", 0.15),
        max_view_disagreement=ins.get("max_view_disagreement", 0.25),
        min_confidence=ins.get("min_confidence", 0.5),
        surface_bins=ins.get("surface_map", {}).get("bins", 600),
        surface_margin=ins.get("surface_map", {}).get("margin", 0.02),
    )


# ---------------------------------------------------------------- 단일 뷰 측정

@dataclass
class ViewMeasure:
    ratio_visible: float
    coverage: float
    sphere: Sphere
    n_pixels: int
    colored: np.ndarray            # H×W bool — 판정 대상 중 착색 픽셀 (오버레이용)
    valid: np.ndarray              # H×W bool — 판정 대상 픽셀
    normals: np.ndarray = field(repr=False)   # N×3 카메라 좌표 법선 (판정 대상 픽셀)
    weights: np.ndarray = field(repr=False)   # N   면적 요소 dA (m²)
    is_colored: np.ndarray = field(repr=False)  # N bool
    cos: np.ndarray = field(repr=False)         # N   cosθ (시선과 법선)


def measure_view(color_bgr: np.ndarray, depth_m: np.ndarray, mask: np.ndarray,
                 intrinsics: tuple[float, float, float, float], p: InspectParams,
                 sphere: Sphere | None = None) -> ViewMeasure | None:
    """한 시점에서 사과 1개의 관측 면 착색비율과 커버리지. color_bgr 은 회색 카드 보정 후 영상."""
    sph = sphere or sphere_from_mask(mask, depth_m, intrinsics)
    if sph is None:
        return None
    m = mask.astype(np.uint8)
    if p.mask_erode_px > 0:
        k = np.ones((2 * p.mask_erode_px + 1,) * 2, np.uint8)
        m = cv2.erode(m, k)
    vs, us = np.nonzero(m)
    if len(us) == 0:
        return None

    rays = pixel_rays(us, vs, intrinsics)
    t, hit = ray_sphere(rays, sph)
    pts = rays * t[:, None]
    n = (pts - sph.center) / sph.radius
    cos = -(n * rays).sum(axis=1)

    lab = bgr_to_lab(color_bgr[vs, us])
    h, c = hue_chroma(lab)
    highlight = (lab[:, 0] >= p.highlight_min_L) & (c < p.highlight_max_chroma)
    keep = hit & (cos >= p.cos_theta_min) & ~highlight
    if keep.sum() == 0:
        return None

    fx, fy, _, _ = intrinsics
    dA = t[keep] ** 2 * rays[keep, 2] ** 3 / (fx * fy) / cos[keep]
    col = colored_mask(lab[keep], p.hue_range_deg, p.min_chroma)

    H, W = mask.shape
    valid = np.zeros((H, W), bool)
    colored = np.zeros((H, W), bool)
    valid[vs[keep], us[keep]] = True
    colored[vs[keep][col], us[keep][col]] = True

    total = dA.sum()
    return ViewMeasure(
        ratio_visible=float((dA * col).sum() / total),
        coverage=float(total / (4 * np.pi * sph.radius ** 2)),
        sphere=sph, n_pixels=int(keep.sum()), colored=colored, valid=valid,
        normals=n[keep], weights=dA, is_colored=col, cos=cos[keep],
    )


# ---------------------------------------------------------------- 모드 A: 탑뷰 단일

def color_ratio(snap: SceneSnapshot, det: AppleDetection2D, apple_id: int, p: InspectParams) -> ColorRatio | None:
    """탑뷰 단일 뷰 착색비율. 판정 구간은 탑뷰–전표면 실측 오차(top_view_error)로 정한다."""
    vm = measure_view(snap.color, snap.depth_m, det.mask, snap.intrinsics, p)
    return None if vm is None else ratio_from_view(vm, p, snap.scene_id, apple_id)


def ratio_from_view(vm: ViewMeasure, p: InspectParams, scene_id: int = 0, apple_id: int = 0) -> ColorRatio:
    r = vm.ratio_visible
    return ColorRatio(
        scene_id=scene_id, apple_id=apple_id, inspect_mode=InspectMode.TOP_SINGLE,
        ratio=r, ratio_lo=max(0.0, r - p.top_view_error), ratio_hi=min(1.0, r + p.top_view_error),
        coverage=vm.coverage, view_disagreement=0.0, num_views=1, variety=p.variety,
    )


# ---------------------------------------------------------------- 모드 B: 인핸드 다면

@dataclass
class InHandView:
    color: np.ndarray                  # 회색 카드 보정 후 BGR
    depth_m: np.ndarray
    intrinsics: tuple[float, float, float, float]
    mask: np.ndarray                   # 사과 마스크 (핑거 영역 제외)
    R_cam_apple: np.ndarray            # 3×3, 사과 좌표 벡터 → 카메라 좌표 (파지 중 사과는 TCP에 고정)
    sphere: Sphere | None = None


def fibonacci_sphere(n: int) -> np.ndarray:
    """구 표면을 거의 같은 넓이로 나누는 n 개 방향 (n×3)."""
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1)


def accumulate_views(views: list[InHandView], p: InspectParams, scene_id: int = 0, apple_id: int = 0) -> ColorRatio | None:
    """여러 뷰를 사과 고정 구면 지도(같은 넓이 칸)에 누적해 전표면 착색비율을 구한다.

    정면에 가까운 픽셀일수록 가중치가 크다 (dA·cosθ). 두 뷰 이상이 본 칸은 뷰 간 착색 차이로 불일치를 잰다.
    """
    bins = fibonacci_sphere(p.surface_bins)
    nb = len(bins)
    w_tot = np.zeros(nb)
    w_col = np.zeros(nb)
    per_view = []
    for v in views:
        vm = measure_view(v.color, v.depth_m, v.mask, v.intrinsics, p, v.sphere)
        if vm is None:
            continue
        n_apple = vm.normals @ v.R_cam_apple            # 카메라 → 사과 좌표 (R 의 전치 적용)
        idx = np.argmax(n_apple @ bins.T, axis=1)
        w = vm.weights * vm.cos
        tv = np.bincount(idx, w, nb)
        cv_ = np.bincount(idx, w * vm.is_colored, nb)
        w_tot += tv
        w_col += cv_
        per_view.append((tv, cv_))
    if not per_view:
        return None

    seen = w_tot > 0
    frac = np.divide(w_col, w_tot, out=np.zeros(nb), where=seen)
    ratio = float(frac[seen].mean())
    coverage = float(seen.mean())

    fracs = np.stack([np.divide(c, t, out=np.full(nb, np.nan), where=t > 0) for t, c in per_view])
    multi = (~np.isnan(fracs)).sum(axis=0) >= 2
    disagreement = float(np.mean(np.nanmax(fracs[:, multi], axis=0) - np.nanmin(fracs[:, multi], axis=0))) if multi.any() else 0.0

    m = p.surface_margin                                     # 칸 경계에서 일부만 본 칸도 '본 칸'으로 세는 오차
    lo = max(0.0, ratio * coverage - m)                      # 안 본 칸이 전부 비착색
    hi = min(1.0, ratio * coverage + (1 - coverage) + m)     # 안 본 칸이 전부 착색
    return ColorRatio(
        scene_id=scene_id, apple_id=apple_id, inspect_mode=InspectMode.IN_HAND_MULTI,
        ratio=ratio, ratio_lo=lo, ratio_hi=hi, coverage=coverage,
        view_disagreement=disagreement, num_views=len(per_view), variety=p.variety,
    )

