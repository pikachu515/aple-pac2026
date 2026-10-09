"""사과 검출 + 후처리 (설계서 apple_detector_yolo, detection_postprocess). 필수.

- Piper Studio YOLO 루프로 학습한 경계상자 모델, 또는 외부 학습 YOLO-seg 가중치
- 경계상자 모델이면 상자 안쪽을 깊이(트레이 면보다 솟은 영역)로 분리해 마스크 생성
- 트레이 ROI 안쪽만 사용, 트레이 면과 높이 차가 없는 검출은 오검출로 제외

동작 (스냅샷 1장마다 1회, 라이브 추론 안 함)
1. 트레이 ROI: workcell.tray.center_m 과 T_base_camera 가 있으면 트레이 사각형을 영상에 투영한다.
   없으면 영상 전체에서 상자(boxes.*.origin_m 이 있으면) 영역만 빼고 쓴다 (개발용).
2. ROI 안 깊이에 RANSAC 평면 → 받침면 위 높이 지도 → perception.detect.min_height_m 보다 솟은 픽셀.
3-a. YOLO 가중치(yolo.weights)가 있으면: 상자마다 '솟은 픽셀 ∩ 상자 내접 타원', 상자끼리 겹친 픽셀은
     중심이 가까운 상자에 준다. 솟은 픽셀이 거의 없는 상자는 오검출로 버린다 (전략 S4-6).
3-b. 없으면: 솟은 덩어리를 사과로 보고, 붙은 사과는 거리 변환 봉우리로 나눈다 (watershed).
4. 투영 직경이 perception.detect.diam_range_m 밖이면 버린다. 모양도 본다 (_apple_shaped):
   꼭대기 높이 > max_height_ratio × 직경(기둥·팔), 둥근 정도 < min_circularity(가늘고 긴 것),
   화면 가장자리에 걸린 덩어리(reject_border). 번호는 위→아래, 왼→오 (det_id 1부터).
   가중치가 COCO 사전학습(yolo11n.pt 등)이면 yolo.classes: [47] 로 apple 만 쓴다 — 학습 없이도
   의자·마우스·팔 같은 다른 물체가 사과 후보에서 빠진다 (2026-10-07 실측).

마스크는 침식하지 않는다 (yolo.mask_erode_px 기본 0). 색 판정(역할 C)이 자체적으로 가장자리를 깎는다.
"""

from __future__ import annotations

from typing import Callable, Sequence

import cv2
import numpy as np

from aple.hal.calibration import T_from_cfg
from aple.hal.depth_correction import DepthCorrection
from aple.perception.geometry import (depth_points, fit_plane_ransac, height_above_plane,
                                      project_polygon_mask, rect_base)
from aple.schema import AppleDetection2D, SceneSnapshot, Source

# (x1, y1, x2, y2, confidence) 픽셀 경계상자 목록을 돌려주는 모델. 테스트에서는 가짜 모델을 넣는다.
BoxModel = Callable[[np.ndarray], Sequence[tuple[float, float, float, float, float]]]

_DETECT_DEFAULTS = {
    "min_height_m": 0.012,
    "diam_range_m": [0.03, 0.12],
    "plane": {"step_px": 4, "iters": 300, "tol_m": 0.004},
    "close_px": 9,
    "open_px": 5,
    "min_area_px": 50,
    "split_min_radius_m": 0.02,
    "min_raised_fraction": 0.3,
    "max_height_ratio": 1.6,
    "min_circularity": 0.5,
    "reject_border": True,
    "roi_margin_m": 0.0,            # 트레이 ROI 를 이만큼 넓혀 덩어리를 찾는다 (경계에 걸친 사과를 통째로)
    "keep_margin_m": 0.0,           # 중심이 트레이 ROI + 이 여유 안인 사과만 남긴다
    "split_by_height": False,       # 붙은 사과를 높이 지도의 꼭대기(정수리)로도 나눈다
    "min_apple_color": 0.0,         # 마스크 안 사과색(빨강~노랑, 채도 있음) 픽셀 비율 하한 (0 = 안 봄)
    "segment": "depth",             # depth: 받침면보다 솟은 픽셀 / color: 사과색 픽셀 (깊이가 흔들리는 현장)
    "color": {"hue_max": 40, "hue_min_wrap": 150, "sat_min": 60, "open_px": 3, "close_px": 7},
}
_MODEL_CACHE: dict[str, BoxModel] = {}


def detect_params(cfg: dict) -> dict:
    p = {**_DETECT_DEFAULTS, **cfg.get("perception", {}).get("detect", {})}
    p["plane"] = {**_DETECT_DEFAULTS["plane"], **p.get("plane", {})}
    p["color"] = {**_DETECT_DEFAULTS["color"], **(p.get("color") or {})}
    return p


def detect(snap: SceneSnapshot, cfg: dict, *, model: BoxModel | None = None) -> list[AppleDetection2D]:
    snap = DepthCorrection.from_cfg(cfg).snapshot(snap)   # 거리별 깊이 오차 (live_camera.depth_correction)
    p = detect_params(cfg)
    ycfg = cfg.get("yolo", {})
    roi = work_roi(snap, cfg)
    tray = cfg.get("workcell", {}).get("tray", {})
    T = T_from_cfg(cfg)
    # 10/9 실기: 사과가 집는 영역 경계(흰 종이↔파란 바닥)에 걸쳐 놓이자 ROI 가 사과를 반으로 잘라 직경이
    # 50~65 mm 로 나왔다. ROI 를 넓혀 덩어리를 통째로 찾고, 중심이 원래 ROI(+keep 여유) 안인 것만 남긴다.
    search, keep = roi, None
    if float(p["roi_margin_m"]) > 0 and T is not None and tray.get("center_m") is not None:
        search = work_roi(snap, cfg, margin_m=float(p["roi_margin_m"]))
        keep = work_roi(snap, cfg, margin_m=float(p["keep_margin_m"]))
    if tray.get("support") == "tray_z" and T is not None and tray.get("center_m") is not None:
        # 자동 검출한 집는 영역: 휘어 보이는 바닥에 2차식 받침면 (광택이면 tray_z 수평면)
        raised, f_px, z_ref, height = raised_mask_floor_fit(snap, roi, p, T, float(tray["center_m"][2]), search)
    else:
        raised, f_px, z_ref, height = raised_mask(snap, roi, p, search=search)
    top_pct = float(cfg.get("perception", {}).get("top_percentile", 95))
    if p["segment"] == "color":
        # 10/9 실기: 사과가 붙어 놓인 파란 트레이에서 깊이는 바닥 ±3 cm·사과 꼭대기 4~8 cm 로 흔들려 붙은 사과를
        # 나누지 못했다. 사과(빨강~노랑)와 파란 트레이·흰 종이는 색 경계가 선명하다 — 색 마스크를 모양으로 나눈다.
        raised = apple_color_mask(snap.color, search, p["color"])

    if model is None and ycfg.get("weights"):
        model = load_yolo(ycfg)
    if model is not None:
        boxes = [b for b in model(snap.color) if b[4] >= float(ycfg.get("conf", 0.0))]
        pieces = _pieces_from_boxes(raised, roi, boxes, p)
    else:
        pieces = [(m, 1.0) for m in _pieces_from_depth(raised, f_px, z_ref, p,
                                                        height if p["split_by_height"] else None)]

    lo, hi = p["diam_range_m"]
    erode = int(ycfg.get("mask_erode_px", 0) or 0)
    found = []
    for mask, conf in pieces:
        area = int(mask.sum())
        if area < p["min_area_px"]:
            continue
        zs = snap.depth_m[mask & (snap.depth_m > 0)]
        z = float(np.median(zs)) if len(zs) else z_ref
        diam = 2 * np.sqrt(area / np.pi) * z / f_px
        if not lo <= diam <= hi:
            continue
        if not _apple_shaped(mask, height, max(diam, _long_side_m(mask, z, f_px)), top_pct, p):
            continue
        if float(p["min_apple_color"]) > 0 and apple_color_fraction(snap.color, mask) < float(p["min_apple_color"]):
            continue                                      # 파란 트레이 벽·흰 종이 같은 솟은 비사과 물체
        if keep is not None:
            m = cv2.moments(mask.astype(np.uint8))
            cu, cv_ = int(round(m["m10"] / m["m00"])), int(round(m["m01"] / m["m00"]))
            if not keep[min(max(cv_, 0), keep.shape[0] - 1), min(max(cu, 0), keep.shape[1] - 1)]:
                continue                                  # 중심이 집는 영역 밖 — 옆 구역(리젝트 종이·상자)의 사과
        if erode > 0:
            mask = cv2.erode(mask.astype(np.uint8), np.ones((2 * erode + 1,) * 2, np.uint8)).astype(bool)
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        found.append((y + h / 2, x + w / 2, h, (x, y, w, h), float(conf), mask))
    return [AppleDetection2D(snap.scene_id, i, bbox, conf, mask)
            for i, (_, _, _, bbox, conf, mask) in enumerate(_reading_order(found), start=1)]


# ================================================================ 영역 · 높이

def work_roi(snap: SceneSnapshot, cfg: dict, margin_m: float = 0.0) -> np.ndarray:
    """사과를 찾을 영역. 트레이가 실측돼 있으면 트레이 사각형(+margin_m 씩 넓혀), 아니면 상자 영역을 뺀 전체."""
    shape = snap.depth_m.shape
    wc = cfg.get("workcell", {})
    T = T_from_cfg(cfg)
    tray = wc.get("tray", {})
    if snap.source == Source.TOP and T is not None and tray.get("center_m") is not None:
        cx, cy, cz = tray["center_m"]
        size = (tray["size_m"][0] + 2 * margin_m, tray["size_m"][1] + 2 * margin_m)
        m = project_polygon_mask(rect_base((cx, cy), size), cz, shape, snap.intrinsics, T)
        if m is not None:
            return m
    roi = np.ones(shape, bool)
    if T is not None:
        for box in wc.get("boxes", {}).values():
            if box.get("origin_m") is None:
                continue
            x0, y0, z0 = box["origin_m"]
            rows, cols = box["grid"]
            pitch = box["cell_pitch_m"]
            sx, sy = cols * pitch, rows * pitch
            band = 0.03                                   # 벽 두께·투영 여유
            corners = rect_base((x0 + sx / 2, y0 + sy / 2), (sx + 2 * band, sy + 2 * band))
            for z in (z0, z0 + box["wall_height_m"]):
                m = project_polygon_mask(corners, z, shape, snap.intrinsics, T)
                if m is not None:
                    roi &= ~m
    return roi


def support_plane_from_base_z(T_base_camera: np.ndarray, z_m: float) -> tuple[np.ndarray, float]:
    """base 높이 z 의 수평면을 카메라 좌표 평면 n·p + d = 0 (n 은 카메라 쪽, d > 0) 으로."""
    T = np.asarray(T_base_camera, float)
    R, t = T[:3, :3], T[:3, 3]
    n = R.T @ np.array([0.0, 0.0, 1.0])
    p0 = R.T @ (np.array([t[0], t[1], z_m]) - t)
    return n, float(-n @ p0)


def raised_mask_floor_fit(snap: SceneSnapshot, roi: np.ndarray, p: dict, T_base_camera: np.ndarray,
                          tray_z_m: float, search: np.ndarray | None = None) -> tuple[np.ndarray, float, float, np.ndarray]:
    """자동 검출한 집는 영역(workcell.tray.support = tray_z)용 — 받침면을 평면 대신 ROI 바닥의 픽셀 2차식으로.

    10/9 실기: 깊이가 영상 가장자리로 갈수록 짧아져 트레이 바닥이 평면이 아니라 휘어 보였다. 평면을 맞추면
    한쪽이 1.2 cm 넘게 솟아 바닥 전체가 사과로 잡혔다. 바닥 깊이가 거의 없으면(광택) tray_z 수평면을 쓴다.
    """
    from aple.hal.calibration import camera_to_base
    from aple.perception.box_occupancy import floor_fit_eval
    from aple.perception.workcell_detect import _fit_floor_px, auto_params
    fx, fy, _, _ = snap.intrinsics
    f = (fx + fy) / 2
    valid = snap.depth_m > 0
    pts = depth_points(snap.depth_m, snap.intrinsics)
    z = np.full(snap.depth_m.shape, np.nan)
    z[valid] = camera_to_base(pts[valid], T_base_camera)[:, 2]
    zz = np.where(roi, z, np.nan)
    coef = None
    if np.isfinite(zz).sum() >= float(p["plane"].get("min_valid_fraction", 0.3)) * max(int(roi.sum()), 1):
        coef = _fit_floor_px(roi, np.dstack([z, z, z]), auto_params({}))
    if coef is None:
        return raised_mask(snap, roi, p, support_plane_from_base_z(T_base_camera, tray_z_m), search=search)
    area = roi if search is None else search          # 받침면은 roi 로 맞추고, 솟은 픽셀은 넓힌 영역에서 찾는다
    v, u = np.indices(z.shape)
    h = np.where(valid, z - floor_fit_eval(coef, u, v), 0.0)
    raised = ((h > p["min_height_m"]) & area).astype(np.uint8)
    if p["close_px"] > 1:
        raised = cv2.morphologyEx(raised, cv2.MORPH_CLOSE, np.ones((p["close_px"],) * 2, np.uint8))
    if p["open_px"] > 1:
        raised = cv2.morphologyEx(raised, cv2.MORPH_OPEN, np.ones((p["open_px"],) * 2, np.uint8))
    floor = roi & valid & (h <= p["min_height_m"])
    z_ref = float(np.median(snap.depth_m[floor])) if floor.any() else float(np.median(snap.depth_m[valid]))
    return raised.astype(bool) & area, f, z_ref, h


def raised_mask(snap: SceneSnapshot, roi: np.ndarray, p: dict,
                plane: tuple[np.ndarray, float] | None = None,
                search: np.ndarray | None = None) -> tuple[np.ndarray, float, float, np.ndarray]:
    """받침면보다 솟은 픽셀 (ROI 안). → (마스크, 초점거리 px, 받침면까지 대표 깊이 m, 높이 지도 m).

    plane 을 주면, ROI 안 유효 깊이가 plane.min_valid_fraction(기본 0.3) 미만일 때 그 받침면을 쓴다 —
    광택 트레이처럼 받침면에 깊이가 없어 평면을 맞출 수 없을 때 (10/9). 깊이가 충분하면 늘 하던 대로 맞춘다.
    """
    fx, fy, _, _ = snap.intrinsics
    f = (fx + fy) / 2
    pts = depth_points(snap.depth_m, snap.intrinsics)
    valid = snap.depth_m > 0
    if plane is not None and (valid & roi).sum() < float(p["plane"].get("min_valid_fraction", 0.3)) * max(int(roi.sum()), 1):
        n, d = plane                                  # 받침면에 깊이가 거의 없다 — 주어진 평면
    else:
        step = int(p["plane"]["step_px"])
        sub = (valid & roi)[::step, ::step]
        n, d = fit_plane_ransac(pts[::step, ::step][sub], iters=int(p["plane"]["iters"]),
                                tol_m=float(p["plane"]["tol_m"]), seed=snap.scene_id)
    h = height_above_plane(pts, valid, n, d)
    area = roi if search is None else search
    raised = ((h > p["min_height_m"]) & area).astype(np.uint8)
    # 순서가 중요하다: 광택으로 생긴 깊이 구멍을 먼저 메우고, 그다음 잡음 점을 지운다.
    if p["close_px"] > 1:
        raised = cv2.morphologyEx(raised, cv2.MORPH_CLOSE, np.ones((p["close_px"],) * 2, np.uint8))
    if p["open_px"] > 1:
        raised = cv2.morphologyEx(raised, cv2.MORPH_OPEN, np.ones((p["open_px"],) * 2, np.uint8))
    z_ref = d / abs(n[2]) if abs(n[2]) > 1e-6 else float(np.median(snap.depth_m[valid]))
    return raised.astype(bool) & area, f, float(z_ref), h


def _long_side_m(mask: np.ndarray, z_m: float, f_px: float) -> float:
    """덩어리 경계상자 긴 변 (m). 트레이 ROI 경계에 잘린 사과도 긴 변은 거의 원래 직경이다."""
    _, _, w, h = cv2.boundingRect(mask.astype(np.uint8))
    return max(w, h) * z_m / f_px


def _apple_shaped(mask: np.ndarray, height: np.ndarray, diam_m: float, top_pct: float, p: dict) -> bool:
    """사과가 아닌 솟은 물체를 거른다 — 팔·상자 벽·트레이 테두리·의자 다리처럼 대회장 화면에 함께 들어오는 것들.

    - 높이/직경: 받침면에 놓인 구는 꼭대기 높이 ≈ 직경. 그보다 훨씬 높으면 기둥·팔이다
    - 둥근 정도(4πA/P²): 원 1, 붙은 사과를 반으로 가른 조각 ≈ 0.7, 가늘고 긴 물체 < 0.5
    - 화면 가장자리에 걸린 덩어리: 잘린 물체라 크기·모양·착색을 믿을 수 없다
    """
    m8 = mask.astype(np.uint8)
    if p["reject_border"] and (m8[0].any() or m8[-1].any() or m8[:, 0].any() or m8[:, -1].any()):
        return False
    hs = height[mask & (height != 0)]                    # 깊이 결측은 0 (height_above_plane)
    if len(hs) and np.percentile(hs, top_pct) > p["max_height_ratio"] * diam_m:
        return False
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perim = sum(cv2.arcLength(c, True) for c in contours)
    return perim > 0 and 4 * np.pi * mask.sum() / perim ** 2 >= p["min_circularity"]


def apple_color_mask(color_bgr: np.ndarray, area: np.ndarray, c: dict) -> np.ndarray:
    """사과색(빨강~주황~노랑, 채도 있음) 픽셀 ∩ area. 작은 잡음은 열고, 꼭지·하이라이트 구멍은 닫는다."""
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0].astype(int), hsv[..., 1].astype(int)
    m = (((hue <= int(c["hue_max"])) | (hue >= int(c["hue_min_wrap"]))) & (sat >= int(c["sat_min"])) & area)
    m = m.astype(np.uint8)
    if int(c["open_px"]) > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((int(c["open_px"]),) * 2, np.uint8))
    if int(c["close_px"]) > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((int(c["close_px"]),) * 2, np.uint8))
    return m.astype(bool) & area


def apple_color_fraction(color_bgr: np.ndarray, mask: np.ndarray) -> float:
    """마스크 안에서 사과색(빨강~주황~노랑, 채도 있음) 픽셀 비율. 파란 트레이·흰 종이·회색 벽은 0 에 가깝다."""
    if not mask.any():
        return 0.0
    hsv = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2HSV)[mask]
    hue, sat = hsv[:, 0].astype(int), hsv[:, 1].astype(int)
    apple = ((hue <= 40) | (hue >= 150)) & (sat >= 60)   # OpenCV H 0~180: 빨강 0/180, 노랑 30, 파랑 ~100
    return float(apple.mean())


# ================================================================ 분할

def _height_seeds(comp: np.ndarray, height: np.ndarray, r_min_px: int) -> list[tuple[float, float]]:
    """덩어리 안 높이 지도의 꼭대기들 (y, x). 맞닿은 구는 투영이 이어져도 정수리는 따로 솟는다 (10/9 실기)."""
    h = np.where(comp > 0, height, 0.0).astype(np.float32)
    sigma = max(1.0, r_min_px / 3)
    hs = cv2.GaussianBlur(h, (0, 0), sigma)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(1.5 * r_min_px) + 1,) * 2)
    top = float(hs[comp > 0].max()) if (comp > 0).any() else 0.0
    peaks = (hs >= cv2.dilate(hs, k) - 1e-6) & (comp > 0) & (hs >= 0.6 * top)
    n, lab = cv2.connectedComponents(peaks.astype(np.uint8))
    seeds = []
    for i in range(1, n):
        ys, xs = np.nonzero(lab == i)
        seeds.append((float(ys.mean()), float(xs.mean()), float(hs[ys, xs].max())))
    out: list[tuple[float, float]] = []
    for y, x, _ in sorted(seeds, key=lambda t: -t[2]):
        if all(np.hypot(y - oy, x - ox) >= 1.5 * r_min_px for oy, ox in out):
            out.append((y, x))
    return out


def _pieces_from_depth(raised: np.ndarray, f_px: float, z_ref: float, p: dict,
                       height: np.ndarray | None = None) -> list[np.ndarray]:
    """솟은 덩어리 → 사과별 마스크. 붙은 사과는 거리 변환 봉우리를 씨앗으로 watershed.

    height(받침면 위 높이 지도)를 주면 정수리 봉우리도 본다 — 거리 변환보다 씨앗이 많으면 그걸로 나눈다
    (맞닿아 한 덩어리처럼 보이는 사과 줄, 10/9 실기)."""
    n_lab, labels = cv2.connectedComponents(raised.astype(np.uint8))
    r_min_px = max(3, int(round(p["split_min_radius_m"] * f_px / z_ref)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_min_px + 1,) * 2)
    out = []
    for lab in range(1, n_lab):
        comp = (labels == lab).astype(np.uint8)
        if comp.sum() < p["min_area_px"]:
            continue
        dist = cv2.distanceTransform(comp, cv2.DIST_L2, 5)
        peak_px = (dist >= cv2.dilate(dist, kernel) - 1e-3) & (dist >= 0.5 * r_min_px) & (comp > 0)
        seeds = _distinct_seeds(peak_px, dist)
        if height is not None:
            tops = _height_seeds(comp, height, r_min_px)
            if len(tops) > len(seeds):                   # 정수리마다 가장 가까운 픽셀을 준다 (비슷한 크기)
                vs, us = np.nonzero(comp)
                d2 = np.stack([(us - x) ** 2 + (vs - y) ** 2 for (y, x) in tops])
                owner = np.argmin(d2, axis=0)
                for k in range(len(tops)):
                    piece = np.zeros(comp.shape, bool)
                    piece[vs[owner == k], us[owner == k]] = True
                    out.append(piece)
                continue
        if len(seeds) <= 1:
            out.append(comp.astype(bool))
            continue
        # 픽셀마다 '자기 원 안으로 가장 깊이 들어간' 씨앗에 준다: argmin(|x − c_k| − R_k).
        # 맞닿은 두 원은 접점에서 정확히 갈리고, 크기가 달라도 경계가 작은 사과 쪽으로 밀리지 않는다.
        vs, us = np.nonzero(comp)
        score = np.stack([np.hypot(us - x, vs - y) - R for (y, x, R) in seeds])
        owner = np.argmin(score, axis=0)
        for k in range(len(seeds)):
            piece = np.zeros(comp.shape, bool)
            piece[vs[owner == k], us[owner == k]] = True
            out.append(piece)
    return out


def _distinct_seeds(peak_px: np.ndarray, dist: np.ndarray) -> list[tuple[float, float, float]]:
    """봉우리 픽셀 → 사과 씨앗 (y, x, 반지름 px). 맞닿은 두 사과의 중심은 R1+R2 만큼 떨어져 있으므로
    그보다 훨씬 가까운 봉우리(한 사과 안의 잡음 봉우리)는 큰 쪽 하나로 합친다."""
    n_pk, lab = cv2.connectedComponents(peak_px.astype(np.uint8))
    cands = []
    for k in range(1, n_pk):
        ys, xs = np.nonzero(lab == k)
        cands.append((float(ys.mean()), float(xs.mean()), float(dist[ys, xs].max())))
    seeds: list[tuple[float, float, float]] = []
    for y, x, R in sorted(cands, key=lambda t: -t[2]):
        if all(np.hypot(y - sy, x - sx) >= 0.75 * (R + sR) for sy, sx, sR in seeds):
            seeds.append((y, x, R))
    return seeds


def _pieces_from_boxes(raised: np.ndarray, roi: np.ndarray, boxes, p: dict) -> list[tuple[np.ndarray, float]]:
    H, W = raised.shape
    v, u = np.indices((H, W))
    owner = np.full((H, W), -1)
    best = np.full((H, W), np.inf)
    kept = []
    for i, (x1, y1, x2, y2, conf) in enumerate(boxes):
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        ax, ay = max((x2 - x1) / 2, 1.0), max((y2 - y1) / 2, 1.0)
        rr = ((u - cx) / ax) ** 2 + ((v - cy) / ay) ** 2   # 내접 타원 안쪽 < 1, 겹치면 상대 거리 작은 쪽
        inside = (rr <= 1.0) & roi
        if inside.sum() == 0 or (raised & inside).sum() < p["min_raised_fraction"] * inside.sum():
            continue                                      # 솟은 물체 없음 → 오검출 (S4-6)
        take = inside & raised & (rr < best)
        best[take] = rr[take]
        owner[take] = i
        kept.append((i, float(conf)))
    out = []
    for i, conf in kept:
        m = (owner == i).astype(np.uint8)
        n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(m)
        if n_lab <= 1:
            continue
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        out.append((labels == big, conf))
    return out


def _reading_order(found: list[tuple]) -> list[tuple]:
    """위→아래 줄, 줄 안에서 왼→오. 중심 높이 차가 사과 높이의 절반 이내면 같은 줄."""
    rows: list[list[tuple]] = []
    for t in sorted(found, key=lambda t: t[0]):
        if rows and t[0] - rows[-1][0][0] <= 0.5 * max(t[2], rows[-1][0][2]):
            rows[-1].append(t)
        else:
            rows.append([t])
    return [t for row in rows for t in sorted(row, key=lambda t: t[1])]


# ================================================================ YOLO

def load_yolo(ycfg: dict) -> BoxModel:
    """ultralytics 가중치 → BoxModel. 같은 가중치는 한 번만 읽는다. ultralytics 는 쓸 때만 import 한다."""
    weights = str(ycfg["weights"])
    if weights not in _MODEL_CACHE:
        try:
            from ultralytics import YOLO
        except ImportError as e:              # pragma: no cover - 설치 환경에 따라
            raise RuntimeError("yolo.weights 가 지정됐지만 ultralytics 가 설치돼 있지 않다 "
                               "(pip install -r requirements.txt)") from e
        net = YOLO(weights)
        kw = {"imgsz": int(ycfg.get("imgsz", 640)), "conf": float(ycfg.get("conf", 0.25)),
              "iou": float(ycfg.get("iou", 0.5)), "verbose": False}
        if ycfg.get("device"):
            kw["device"] = ycfg["device"]
        if ycfg.get("classes") is not None:              # COCO 사전학습 가중치는 apple(47) 만 쓴다
            kw["classes"] = [int(c) for c in ycfg["classes"]]

        def run(color_bgr: np.ndarray):
            res = net.predict(color_bgr, **kw)[0]
            xyxy = res.boxes.xyxy.cpu().numpy()
            conf = res.boxes.conf.cpu().numpy()
            return [(*map(float, b), float(c)) for b, c in zip(xyxy, conf)]

        _MODEL_CACHE[weights] = run
    return _MODEL_CACHE[weights]
