"""상자·리젝트 평면 구역·사과 집는 영역 위치를 탑뷰 컬러·깊이로 찾는다 — 실측 좌표 없이 카메라로 (역할 A).

workcell.auto_detect.enabled 이면 장면마다 이 결과로 workcell.boxes.*.origin_m·grid·wall_height_m 과
workcell.tray(사과 집는 영역)를 채운 설정 사본을 만들어 검출·위치·상자 점유 추정에 쓴다
(SceneRunner.from_config → AutoWorkcell). 설정 파일의 origin_m 은 '검출 실패 때 쓰는 예비값' 이 된다
(tools/workcell_detect.py --save).

10/9 최종 현장 배치 (기본 설정이 이것을 따른다)
  - 벽 있는 골판지 상자 2개: 영상 왼쪽 = UPPER(착색 높은 사과), 오른쪽 = MIDDLE (assign)
  - 가운데 파란 평판 트레이: 로봇 쪽(영상 위) 흰 종이 = REJECT 평면 구역 (벽 없음, 종이 위에 칸 격자로 내려놓는다),
    아래 파란 부분 = 사과를 집는 영역 (workcell.tray 대용)
  - 로봇 팔·사람 손이 화면에 있어도 된다: 테이블보다 높이 떠 있는 것은 색 판단에서 뺀다

방법 (base_link 위에서 내려다본 격자, grid_res_m)
  1. 깊이 점 → base_link → 격자. 칸마다 최고·최저 높이, 색 종류별 픽셀 비율 (회색 카드 보정된 영상 기준)
  2. 높이: 테이블 면을 2차 곡면으로 맞추고, 그래도 남는 완만한 휨은 국소 열림(local_window_m)으로 지운
     '테이블 위 높이' 로 판단한다 — 10/9 실기에서 깊이가 영상 가장자리로 갈수록 10 cm 넘게 짧아져 base 높이가
     휘었다 (D435 깊이 왜곡, 카메라 재보정 필요). 테이블 면보다 크게 낮은 칸(테이블 밖 바닥)은 버린다
  3. 상자: 골판지 색 칸(바닥·벽, 사과 자리는 구멍 메움) 덩어리 → 최소 외접 사각형 = 바깥.
     붙어 있는 상자는 덩어리가 한 상자보다 길면, 낮은 골판지(바닥) 폭이 좁아지는 곳(사이 벽)에서 자른다.
     바닥 = 사각형 안 낮은 골판지 칸의 절대 높이 중앙값, 벽 높이 = 사각형 안쪽 둘레 띠의 상위 높이 − 바닥
  4. 트레이: 파란 칸 덩어리의 사각형. 리젝트 구역 = 트레이 안의 흰(종이) 칸 사각형, 바닥 = 종이 높이, 벽 0.
     집는 영역 = 트레이 사각형에서 종이 쪽을 뺀 나머지 — 색이 아니라 기하로 정하므로 사람 팔이 위에 있어도 된다
     (트레이를 못 찾으면 상자 둘레를 뺀 테이블 평탄 영역 중 가장 큰 덩어리)
  5. 칸 격자: 지금 배치 코드는 구역이 base 축에 정렬돼 있다고 본다 (열 +x, 행 +y, 원점 = 최소 모서리).
     안쪽 사각형 안에 축 정렬로 들어가는 cell_pitch_m 칸 수를 세어 가운데에 놓는다.
     |yaw| 가 max_yaw_deg 를 넘으면 그 구역은 쓰지 않는다 (경고) — 상자·트레이를 로봇 정면 축에 맞춰 놓는다
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from aple.hal.calibration import camera_to_base, validate_T
from aple.hal.depth_correction import DepthCorrection
from aple.inspection.color import bgr_to_lab, hue_chroma
from aple.perception.geometry import depth_points
from aple.schema import SceneSnapshot, Source

_DEFAULTS = {
    "enabled": False,
    "area_m": [[-0.25, 0.95], [-0.65, 0.65]],   # 찾을 범위 base x, y (m)
    "grid_res_m": 0.005,
    # 색 (회색 카드 보정 뒤 Lab, 10/9 실기): 골판지 L 40~67·C 11~15·h 58~79°, 파란 트레이 C 15~23·h −145~−160°,
    # 흰 종이 L 83~86·C < 8 (흰 테이블과 같아 트레이 안에 있을 때만 종이로 본다), 사과 C ≥ 50
    # 골판지 채도 상한 17.5: 테이블 끝 너머 나무 바닥(C 19~20)과 가른다. 그늘진 벽 h ≈ 38°
    "cardboard": {"max_L": 75.0, "chroma": [9.0, 17.5], "hue_deg": [30.0, 95.0]},
    "tray_color": {"chroma": [10.0, 45.0], "hue_deg": [-175.0, -125.0]},
    "paper_color": {"min_L": 72.0, "max_chroma": 10.0},
    "min_color_fraction": 0.5,                   # 칸 안 픽셀 중 그 색 비율 하한
    "surface_iters": 8,                          # 테이블 곡면 반복 맞춤 횟수
    "surface_keep_m": 0.004,                     # 곡면 위로 이만큼(+잔차 중앙값)까지 테이블로 남긴다
    "local_window_m": 0.06,                      # 국소 바닥 창 (벽 폭보다 넓고 상자 안쪽보다 좁게)
    "below_table_m": 0.05,                       # 테이블 면보다 이만큼 낮으면 테이블 밖 (바닥의 나무색 제외)
    "wall_min_m": 0.025,                         # 국소 높이 이 이상 = 벽·사과 (바닥 아님)
    "rim_height_m": 0.06,                        # 상자 벽 윗면 높이 (윤곽을 이 높이 평면에 투영한다)
    "box_floor_m": 0.005,                        # 상자 바닥 판 두께 — 바닥 높이 = 테이블 기준 높이 + 이것
    "max_wall_height_m": 0.12,                   # 잰 벽 높이 상한 (깊이 휨으로 부풀어도 이 이상은 아니다)
    "box_max_height_m": 0.35,                    # 이보다 높은 것은 상자가 아니다 (로봇·사람). 가장자리 깊이 휨(~0.2 m)보다 크게
    "outer_size_m": [0.12, 0.50],                # 상자 바깥 한 변 허용 범위
    "min_rectangularity": 0.65,                  # 볼록껍질 넓이 / 최소 외접 사각형 넓이 (가장자리 사과·영상 끝에 깎인다, 10/9 실기 0.71)
    "wall_band_m": 0.02,                         # 벽 높이를 재는 사각형 안쪽 둘레 띠
    "wall_percentile": 75.0,                     # 둘레 골판지 칸 높이의 이 분위 = 벽 윗면 (안쪽 벽면이 섞인다)
    "wall_thickness_m": 0.005,                   # 골판지 두께 (바깥 → 안쪽)
    "tray_size_m": [0.15, 0.90],                 # 트레이 한 변 허용 범위 (시야 끝에 잘려도 된다)
    "tray_rim_m": 0.035,                         # 트레이 테두리 띠 (집는 영역에서 뺀다 — 벽이 사과와 붙어 한 덩어리가 되지 않게)
    "paper_min_size_m": 0.08,                    # 종이 구역 한 변 하한
    "image_close_px": 4,                         # 영상 색 마스크 틈 메우기 (픽셀)
    "min_box_pixels": 3000,                      # 상자로 볼 골판지 덩어리 최소 픽셀 수
    "max_depth_m": 1.0,                          # 카메라에서 이보다 먼 골판지 색은 테이블 밖 (테이블 0.87~0.93 m)
    "image_open_px": 20,                         # 상자와 옆 물건(과자 봉지·전선·그늘)을 잇는 다리·돌기 끊기 (픽셀, 상자 폭 ≈ 170 px)
    "min_tray_pixels": 2000,                     # 트레이·종이로 볼 최소 픽셀 수
    "max_yaw_deg": 8.0,                          # 이보다 비뚤어진 구역은 쓰지 않는다 (축 정렬 배치)
    "fit_margin_m": 0.005,                       # 칸 격자를 안쪽 경계에서 띄우는 여유
    # 벽 있는 상자 이름 배정 {이름: 쪽}. 쪽 = left | right | top | bottom (카메라 영상 기준 — T_base_camera 로 base
    #   방향을 구한다) 또는 +x | -x | +y | -y (base). 적힌 순서대로 남은 상자 중 그 쪽 끝에 있는 상자를 준다.
    #   대회장에서 상·중 상자를 반대로 부르면 한 줄만 바꾼다: {UPPER: right, MIDDLE: left}
    "assign": {"UPPER": "left", "MIDDLE": "right"},
    "flat_zone": "REJECT",                       # 트레이 위 흰 종이 구역에 줄 상자 이름 (null = 안 씀)
    "pick_flat_tol_m": 0.015,                    # 트레이가 없을 때: 국소 높이가 이 안이면 평탄
    "pick_clear_m": 0.03,                        # 상자·구역 바깥 이만큼은 집는 영역에서 뺀다
    "pick_percentile": 2.0,                      # 영역 범위를 이 분위로 (튀는 칸 무시)
}

_COLOR_KEYS = ("cardboard", "tray_color", "paper_color")


def auto_params(cfg: dict) -> dict:
    user = cfg.get("workcell", {}).get("auto_detect") or {}
    out = {**_DEFAULTS, **user}
    for key in _COLOR_KEYS:
        out[key] = {**_DEFAULTS[key], **(user.get(key) or {})}
    return out


def auto_enabled(cfg: dict) -> bool:
    return bool((cfg.get("workcell", {}).get("auto_detect") or {}).get("enabled"))


@dataclass
class BoxDetection:
    center_xy: tuple[float, float]               # 사각형 중심 (base, m)
    outer_size_m: tuple[float, float]            # yaw 를 뺀 바깥 크기 (x 쪽 변, y 쪽 변)
    yaw_rad: float                               # x 쪽 변의 기울기 (−45°, 45°]
    floor_z_m: float                             # 바닥(종이) 윗면 (base 절대 높이)
    wall_height_m: float                         # 벽 윗면 − 바닥 (평면 구역 0)
    flat: bool = False                           # 벽 없는 평면 구역 (리젝트 종이)
    grid: tuple[int, int] = (0, 0)               # (rows, cols) — 축 정렬로 들어가는 칸 수
    origin_m: tuple[float, float, float] | None = None
    name: str | None = None
    usable: bool = True
    note: str = ""
    floor_fit_px: list[float] | None = None      # 바닥의 잰 base 높이 = 픽셀 2차식 (box_occupancy 가 깊이 휨을 지운다)

    def box_cfg(self, pitch: float) -> dict:
        out = {"grid": list(self.grid), "cell_pitch_m": pitch, "wall_height_m": round(self.wall_height_m, 4),
               "origin_m": [round(float(v), 4) for v in self.origin_m]}
        if self.floor_fit_px is not None:
            out["floor_fit_px"] = [float(v) for v in self.floor_fit_px]
        return out


@dataclass
class WorkcellDetection:
    boxes: list[BoxDetection]                    # 쓸 수 있는 것·없는 것 모두 (usable)
    pick_center_m: tuple[float, float, float] | None = None
    pick_size_m: tuple[float, float] | None = None
    tray: tuple | None = None                    # (중심 xy, 크기, yaw) — 찾았으면
    warnings: list[str] = field(default_factory=list)
    grid: dict | None = None                     # 시각화·점검용

    def named(self) -> dict[str, BoxDetection]:
        return {b.name: b for b in self.boxes if b.usable and b.name}


# ================================================================ 색

def color_masks(bgr: np.ndarray, p: dict) -> dict[str, np.ndarray]:
    """골판지·파란 트레이·흰 종이 픽셀 (회색 카드 보정된 영상 기준)."""
    lab = bgr_to_lab(bgr)
    h, c = hue_chroma(lab)
    L = lab[..., 0]
    cb, tr, pa = p["cardboard"], p["tray_color"], p["paper_color"]
    return {
        "cardboard": ((L <= float(cb["max_L"])) & (c >= float(cb["chroma"][0])) & (c <= float(cb["chroma"][1]))
                      & (h >= float(cb["hue_deg"][0])) & (h <= float(cb["hue_deg"][1]))),
        "tray": ((c >= float(tr["chroma"][0])) & (c <= float(tr["chroma"][1]))
                 & (h >= float(tr["hue_deg"][0])) & (h <= float(tr["hue_deg"][1]))),
        "paper": (L >= float(pa["min_L"])) & (c <= float(pa["max_chroma"])),
    }


# ================================================================ 격자

def _grids(snap: SceneSnapshot, T: np.ndarray, p: dict):
    """→ (x0, y0, res, hmax, hmin, {색: 칸 비율}, px) — 칸이 비면 nan.

    px: 픽셀 단위 보조값 {"pts": H×W×3 base 점 (깊이 없으면 nan), "colors": 색 마스크, "cell": (iy, ix) 또는 −1}.
    """
    H, W = snap.depth_m.shape
    valid = snap.depth_m > 0
    pts_px = np.full((H, W, 3), np.nan)
    pts_px[valid] = camera_to_base(depth_points(snap.depth_m, snap.intrinsics)[valid], T)
    colors_px = color_masks(snap.color, p)
    (x0, x1), (y0, y1) = (tuple(map(float, r)) for r in p["area_m"])
    res = float(p["grid_res_m"])
    nx, ny = int(math.ceil((x1 - x0) / res)), int(math.ceil((y1 - y0) / res))
    X, Y = pts_px[..., 0], pts_px[..., 1]
    inside = valid & (X >= x0) & (X < x1) & (Y >= y0) & (Y < y1)
    cell_x = np.full((H, W), -1)
    cell_y = np.full((H, W), -1)
    cell_x[inside] = np.clip(((X[inside] - x0) / res).astype(int), 0, nx - 1)
    cell_y[inside] = np.clip(((Y[inside] - y0) / res).astype(int), 0, ny - 1)
    iy, ix, z = cell_y[inside], cell_x[inside], pts_px[..., 2][inside]
    hmax = np.full((ny, nx), -np.inf)
    hmin = np.full((ny, nx), np.inf)
    count = np.zeros((ny, nx))
    np.maximum.at(hmax, (iy, ix), z)
    np.minimum.at(hmin, (iy, ix), z)
    np.add.at(count, (iy, ix), 1)
    has = count > 0
    hmax[~has], hmin[~has] = np.nan, np.nan
    fracs = {}
    for name, flag in colors_px.items():
        n = np.zeros((ny, nx))
        np.add.at(n, (iy, ix), flag[inside].astype(float))
        fracs[name] = np.where(has, n / np.maximum(count, 1), np.nan)
    return x0, y0, res, hmax, hmin, fracs, {"pts": pts_px, "colors": colors_px, "cell": (cell_y, cell_x)}


def _table_surface(x0, y0, res, hmin, p) -> np.ndarray | None:
    """칸 최저 높이에 2차 곡면 z = c0 + c1 x + c2 y + c3 x² + c4 xy + c5 y² — 위로 튀는 칸(벽·사과·물건)을 빼며 반복."""
    gy, gx = np.nonzero(np.isfinite(hmin))
    if len(gx) < 50:
        return None
    X, Y, Z = x0 + (gx + 0.5) * res, y0 + (gy + 0.5) * res, hmin[gy, gx]
    A = np.column_stack([np.ones_like(X), X, Y, X * X, X * Y, Y * Y])
    keep = np.ones(len(Z), bool)
    for _ in range(int(p["surface_iters"])):
        c = np.linalg.lstsq(A[keep], Z[keep], rcond=None)[0]
        r = Z - A @ c
        s = float(np.median(np.abs(r[keep]))) * 1.5 + float(p["surface_keep_m"])
        keep = (r < s) & (r > -3 * s)
        if keep.sum() < 50:
            return None
    ny, nx = hmin.shape
    gy, gx = np.indices((ny, nx))
    X, Y = x0 + (gx + 0.5) * res, y0 + (gy + 0.5) * res
    return c[0] + c[1] * X + c[2] * Y + c[3] * X * X + c[4] * X * Y + c[5] * Y * Y


def _local_rel(rel: np.ndarray, res: float, p: dict) -> np.ndarray:
    """곡면을 빼고도 남는 완만한 휨을 지운다 — 높이 격자의 열림(최소 → 최대 필터, 창 local_window_m)을 뺀 값.

    벽(폭 1~2 cm)처럼 창보다 좁고 솟은 것만 남고, 창보다 넓은 바닥·테이블은 0 근처가 된다.
    """
    k = max(1, int(round(float(p["local_window_m"]) / res)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1))
    big = 1e3
    a = np.where(np.isfinite(rel), rel, big).astype(np.float32)
    eroded = cv2.erode(a, kernel)
    eroded[eroded >= big / 2] = -big
    opened = cv2.dilate(eroded, kernel)
    out = rel - opened
    out[~np.isfinite(rel) | (opened <= -big / 2)] = np.nan
    return out


# ================================================================ 사각형 도구

def _rect_metric(rect, x0, y0, res) -> tuple[tuple[float, float], float, float, float]:
    """cv2 최소 외접 사각형(격자 좌표) → (중심 xy m, x 쪽 변 m, y 쪽 변 m, yaw rad ∈ (−π/4, π/4])."""
    (cx, cy), _, _ = rect
    pts = cv2.boxPoints(rect)
    e1, e2 = pts[1] - pts[0], pts[2] - pts[1]

    def wrap(a):                                         # 변 방향을 (−π/2, π/2] 로
        while a <= -math.pi / 2:
            a += math.pi
        while a > math.pi / 2:
            a -= math.pi
        return a
    a1, a2 = wrap(math.atan2(e1[1], e1[0])), wrap(math.atan2(e2[1], e2[0]))
    l1, l2 = float(np.hypot(*e1)) * res, float(np.hypot(*e2)) * res
    sx, sy, yaw = (l1, l2, a1) if abs(a1) <= abs(a2) else (l2, l1, a2)
    return (x0 + (cx + 0.5) * res, y0 + (cy + 0.5) * res), sx, sy, yaw


def _rect_cells(shape, center_xy, size, yaw, x0, y0, res) -> np.ndarray:
    """base 사각형(중심·크기·yaw)을 격자 마스크로."""
    mask = np.zeros(shape, np.uint8)
    if min(size) <= 0:
        return mask.astype(bool)
    rect = (((center_xy[0] - x0) / res - 0.5, (center_xy[1] - y0) / res - 0.5),
            (size[0] / res, size[1] / res), math.degrees(yaw))
    cv2.fillPoly(mask, [np.round(cv2.boxPoints(rect)).astype(np.int32)], 1)
    return mask.astype(bool)


def _ray_to_plane(uv: np.ndarray, intrinsics, T: np.ndarray, z: float) -> np.ndarray:
    """픽셀 (N×2) → base 높이 z 평면과 시선의 교점 xy (N×2). 깊이를 쓰지 않는다 (광택 트레이는 깊이가 비고,
    깊이 왜곡도 피한다 — 높이 오차 1 cm 가 옆으로 약 3 mm)."""
    fx, fy, cx, cy = intrinsics
    rays = np.column_stack([(uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy, np.ones(len(uv))])
    d = rays @ np.asarray(T)[:3, :3].T
    o = np.asarray(T)[:3, 3]
    s = (z - o[2]) / d[:, 2]
    return (o[None, :2] + d[:, :2] * s[:, None])


def _metric_rect(xy: np.ndarray):
    """base xy 점들 → (중심, x 쪽 변, y 쪽 변, yaw) — _rect_metric 을 mm 격자로."""
    rect = cv2.minAreaRect((xy * 1000.0).astype(np.float32))
    return _rect_metric(rect, -0.0005, -0.0005, 0.001)


def _largest_filled(mask: np.ndarray, close_px: int, *, hull: bool = False) -> np.ndarray | None:
    """가장 큰 덩어리를 메운 마스크. hull: 볼록껍질 — 시야 끝에 잘려 테두리가 열린 트레이도 안쪽(종이)을 덮는다."""
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (2 * close_px + 1, 2 * close_px + 1)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, [cv2.convexHull(c) if hull else c], -1, 1, thickness=-1)
    return out.astype(bool)


def _fit_floor_px(mask: np.ndarray, pts_px: np.ndarray, p: dict) -> list[float] | None:
    """바닥 픽셀(깊이 있음)의 잰 base 높이를 픽셀 2차식으로 (u·v 는 /1000). 바닥 위 물건(사과)은 반복해서 뺀다."""
    z_all = pts_px[..., 2]
    v, u = np.nonzero(mask & np.isfinite(z_all))
    if len(u) < 100:
        return None
    Z = z_all[v, u]
    uu, vv = u / 1000.0, v / 1000.0
    A = np.column_stack([np.ones_like(uu), uu, vv, uu * uu, uu * vv, vv * vv])
    keep = np.ones(len(Z), bool)
    for _ in range(int(p["surface_iters"])):
        c = np.linalg.lstsq(A[keep], Z[keep], rcond=None)[0]
        r = Z - A @ c
        s_ = float(np.median(np.abs(r[keep]))) * 1.5 + float(p["surface_keep_m"])
        keep = (r < s_) & (r > -3 * s_)
        if keep.sum() < 50:
            return None
    return [float(x) for x in c]


def _hull_rect(mask: np.ndarray, snap, T, z):
    """영상 마스크 → 볼록껍질 꼭짓점을 높이 z 평면에 투영한 (중심, x 쪽 변, y 쪽 변, yaw, 직사각형 정도)."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    hull = cv2.convexHull(np.vstack(contours)).reshape(-1, 2).astype(float)
    xy = _ray_to_plane(hull, snap.intrinsics, T, z)
    center, sx, sy, yaw = _metric_rect(xy)
    area = 0.5 * abs(float(np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))))
    lo, hi = xy.min(0), xy.max(0)                     # 축 정렬 사각형 (yaw 0)
    aligned = ((lo + hi) / 2, float(hi[0] - lo[0]), float(hi[1] - lo[1]), 0.0,
               area / float(np.prod(hi - lo)) if np.prod(hi - lo) > 0 else 0.0)
    return (center, sx, sy, yaw, (area / (sx * sy) if sx * sy > 0 else 0.0)), aligned


def _find_boxes(snap, T, colors, above_px, rel_px, pts_px, z_ref, p) -> list[BoxDetection]:
    """골판지 색 덩어리(영상) → 볼록껍질(사과 자리 포함)을 테이블 면에 투영한 사각형 = 상자 바깥.

    위치는 시선 투영으로 — 영상 가장자리에서 깊이가 15~20 % 짧게 나와 깊이로 옮기면 상자가 찌그러진다 (10/9).
    높이는 '테이블 곡면 위 높이' 로 재고 기준 높이 z_ref(카메라 바로 아래 테이블) 에 더한다.
    """
    known = np.isfinite(above_px)
    tall = known & (above_px > float(p["box_max_height_m"]))
    below = known & (above_px < -float(p["below_table_m"]))
    # 깊이가 있고 max_depth_m 안인 골판지만 — 테이블 끝 너머 나무 바닥은 깊이가 비거나 멀다 (10/9: 0 또는 1.07 m)
    near = (snap.depth_m > 0) & (snap.depth_m <= float(p["max_depth_m"]))
    brown = colors["cardboard"] & ~tall & ~below & near
    k, o = int(p["image_close_px"]), int(p["image_open_px"])
    brown_c = cv2.morphologyEx(brown.astype(np.uint8), cv2.MORPH_CLOSE,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1)))
    filled = np.zeros_like(brown_c)                                # 상자 안 사과 자리(구멍)를 먼저 메운다 —
    contours, _ = cv2.findContours(brown_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)   # 열림이 사과 사이 바닥을 지우지 않게
    cv2.drawContours(filled, contours, -1, 1, thickness=-1)
    brown_c = cv2.morphologyEx(filled, cv2.MORPH_OPEN,             # 옆 물건(과자 봉지·전선)과 잇는 가는 다리를 끊는다
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * o + 1, 2 * o + 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(brown_c, connectivity=8)
    lo, hi = (float(v) for v in p["outer_size_m"])
    fx = float(snap.intrinsics[0])
    low = np.isfinite(rel_px) & (rel_px < float(p["wall_min_m"]))
    out = []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < int(p["min_box_pixels"]):
            continue
        comp = labels == i
        # 볼록껍질 바깥 경계는 벽 윗면이다 — 바닥 높이에 투영하면 시차만큼 바깥으로 커진다 (0.9 m 에서 약 1.5 cm)
        r = _hull_rect(comp, snap, T, z_ref + float(p["rim_height_m"]))
        if r is None:
            continue
        (center, sx, sy, yaw, rect), aligned = r
        if abs(math.degrees(yaw)) > float(p["max_yaw_deg"]) and aligned[4] >= float(p["min_rectangularity"]):
            # 상자 안 사과·벽 가림으로 볼록껍질 모서리가 깎이면 최소 외접 사각형이 크게 기운다 (10/9: −17°).
            # 상자는 로봇 축에 맞춰 테이프로 붙여 두므로 축 정렬 사각형이 잘 맞으면 그것을 쓴다
            center, sx, sy, yaw, rect = aligned
        if min(sx, sy) < lo or max(sx, sy) > hi or rect < float(p["min_rectangularity"]):
            continue
        hull = np.zeros(comp.shape, np.uint8)
        contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(hull, [cv2.convexHull(np.vstack(contours))], -1, 1, thickness=-1)
        hull = hull.astype(bool)
        floor = hull & brown & low & known
        if floor.sum() < 50:
            continue
        depth = float(np.median(snap.depth_m[floor & (snap.depth_m > 0)])) if (floor & (snap.depth_m > 0)).any() else 1.0
        band = max(2, int(round(float(p["wall_band_m"]) * fx / depth)))
        inner = cv2.erode(hull.astype(np.uint8), np.ones((2 * band + 1, 2 * band + 1), np.uint8)).astype(bool)
        ring = hull & ~inner & brown & known
        floor_rel = float(np.median(above_px[floor]))
        wall_h = float(np.percentile(above_px[ring], float(p["wall_percentile"])) - floor_rel) if ring.sum() >= 20 \
            else float("nan")
        # 상자는 테이블 위에 놓인다 — 바닥 = 기준 테이블 높이 + 골판지 바닥 두께. 영상 가장자리의 깊이 휨이
        # 곡면 맞춤 뒤에도 수 cm~20 cm 남아 잰 바닥 높이는 믿지 않는다 (10/9: 67·133 mm 로 나왔다)
        wall_h = min(wall_h, float(p["max_wall_height_m"])) if math.isfinite(wall_h) else wall_h
        b = BoxDetection(center, (sx, sy), yaw, z_ref + float(p["box_floor_m"]), wall_h)
        b.floor_fit_px = _fit_floor_px(floor, pts_px, p)
        b.note = f"바깥 {sx * 1000:.0f}×{sy * 1000:.0f} mm"
        x_, y_, w_, h_ = cv2.boundingRect(comp.astype(np.uint8))
        H_, W_ = comp.shape
        if x_ <= 1 or y_ <= 1 or x_ + w_ >= W_ - 1 or y_ + h_ >= H_ - 1:
            b.note += " — 영상 끝에 걸림: 보이는 부분에만 칸을 놓는다"
        out.append(b)
    return out


def _find_tray(snap, T, above_px, pts_px, colors, z_ref, x0, y0, res, shape, p):
    """영상에서 파란 트레이와 그 안의 흰 종이를 색으로 찾아 시선을 트레이 면에 투영한다.

    → (트레이 (중심, 크기, yaw) | None, 종이 구역 BoxDetection | None, 집는 칸 마스크 | None, 트레이 면 높이).
    파란 바닥은 광택으로 깊이가 거의 없고, 로봇 팔·사람 손이 위를 지나갈 수 있다 — 테이블보다 높이 뜬 픽셀은 뺀다.
    """
    known = np.isfinite(above_px)
    # 로봇 팔·사람처럼 크게 떠 있는 것만 뺀다 — 사과는 파랗거나 희지 않다. 영상 가장자리 깊이 휨이 수 cm~20 cm 라
    # 낮은 문턱을 쓰면 트레이 끝이 잘린다
    tall = known & (above_px > float(p["box_max_height_m"]))
    blue = colors["tray"] & ~tall
    close_px = int(p["image_close_px"])
    tray_img = _largest_filled(blue, close_px, hull=True)
    if tray_img is None or blue[tray_img].sum() < int(p["min_tray_pixels"]):
        return None, None, None, z_ref
    paper_img = _largest_filled(colors["paper"] & tray_img & ~blue & ~tall, close_px)
    rel = None
    for src in (paper_img, tray_img & blue):
        if src is not None and (src & known).sum() >= 20:
            rel = float(np.median(above_px[src & known]))
            break
    z = z_ref + (rel if rel is not None else 0.0)
    v, u = np.nonzero(tray_img)
    center, sx, sy, yaw = _metric_rect(_ray_to_plane(np.column_stack([u, v]).astype(float), snap.intrinsics, T, z))
    lo, hi = (float(x) for x in p["tray_size_m"])
    if min(sx, sy) < lo or max(sx, sy) > hi:
        return None, None, None, z
    rim = 2 * float(p["tray_rim_m"])
    zone = None
    pick = (np.asarray(center, float), (sx - rim, sy - rim), yaw)
    if paper_img is not None and paper_img.sum() >= int(p["min_tray_pixels"]):
        v, u = np.nonzero(paper_img)
        pc, px, py, pyaw = _metric_rect(_ray_to_plane(np.column_stack([u, v]).astype(float), snap.intrinsics, T, z))
        if min(px, py) >= float(p["paper_min_size_m"]):
            zone = BoxDetection(pc, (px, py), pyaw, z, 0.0, flat=True, note=f"종이 {px * 1000:.0f}×{py * 1000:.0f} mm",
                                floor_fit_px=_fit_floor_px(paper_img, pts_px, p))
            pick = _pick_strip(center, (sx, sy), yaw, pc, (px, py), rim, float(p["pick_clear_m"]))
    cells = _rect_cells(shape, pick[0], pick[1], pick[2], x0, y0, res) if pick is not None else None
    return (center, (sx, sy), yaw), zone, cells, z


def _pick_strip(tc, tsize, yaw, pc, psize, rim, clear):
    """트레이 사각형에서 종이 바깥쪽 네 띠(트레이 좌표계 ±x, ±y) 중 가장 넓은 것 = 사과 집는 영역.

    10/9 배치: 종이는 트레이의 로봇 쪽 반, 집는 영역은 나머지 반 — 종이 옆 좁은 틈은 고르지 않는다.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(pc, float) - np.asarray(tc, float)
    px_, py_ = d[0] * c + d[1] * s, -d[0] * s + d[1] * c                # 종이 중심 (트레이 좌표)
    hx, hy = tsize[0] / 2 - rim / 2, tsize[1] / 2 - rim / 2
    strips = []
    for axis, sign in ((0, 1), (0, -1), (1, 1), (1, -1)):
        h_along, h_across = (hx, hy) if axis == 0 else (hy, hx)
        paper_c = px_ if axis == 0 else py_
        paper_h = (psize[0] if axis == 0 else psize[1]) / 2
        lo, hi = (paper_c + paper_h + clear, h_along) if sign > 0 else (-h_along, paper_c - paper_h - clear)
        if hi - lo <= 0:
            continue
        mid = (lo + hi) / 2
        local = (mid, 0.0) if axis == 0 else (0.0, mid)
        size = (hi - lo, 2 * h_across) if axis == 0 else (2 * h_across, hi - lo)
        strips.append(((hi - lo) * 2 * h_across, local, size))
    if not strips:
        return None
    _, (lx, ly), size = max(strips, key=lambda t: t[0])
    center = np.asarray(tc, float) + np.array([lx * c - ly * s, lx * s + ly * c])
    return center, size, yaw


def _pick_from_cells(cells, x0, y0, res, z, p):
    gy, gx = np.nonzero(cells)
    if len(gx) < 20:
        return None, None
    q = float(p["pick_percentile"])
    xs, ys = x0 + (gx + 0.5) * res, y0 + (gy + 0.5) * res
    xl, xh = np.percentile(xs, [q, 100 - q])
    yl, yh = np.percentile(ys, [q, 100 - q])
    return (float((xl + xh) / 2), float((yl + yh) / 2), float(z)), (float(xh - xl), float(yh - yl))


def _flat_pick_area(x0, y0, res, rel, boxes, p):
    """트레이가 없을 때: 상자 둘레를 뺀 테이블 평탄 영역 중 가장 큰 덩어리."""
    valid = np.isfinite(rel)
    flat = valid & (np.abs(np.nan_to_num(rel, nan=1.0)) < float(p["pick_flat_tol_m"]))
    clear = 2 * float(p["pick_clear_m"])
    for b in boxes:
        sx, sy = b.outer_size_m
        flat &= ~_rect_cells(rel.shape, b.center_xy, (sx + clear, sy + clear), b.yaw_rad, x0, y0, res)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(flat.astype(np.uint8), connectivity=4)
    if n <= 1:
        return None
    return labels == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))


# ================================================================ 칸 격자 · 이름

def _fit_grid(box: BoxDetection, pitch: float, p: dict) -> None:
    """축 정렬 칸 격자를 안쪽 사각형 안 가운데에 놓는다. |yaw| 가 크면 쓰지 않는다."""
    yaw = box.yaw_rad
    if abs(math.degrees(yaw)) > float(p["max_yaw_deg"]):
        box.usable = False
        box.note = f"yaw {math.degrees(yaw):.0f}° > {float(p['max_yaw_deg']):g}° — 로봇 축에 맞춰 놓는다"
        return
    if not box.flat and (not math.isfinite(box.wall_height_m) or box.wall_height_m <= 0):
        box.usable = False
        box.note = "벽 높이를 재지 못했다"
        return
    t = 0.0 if box.flat else float(p["wall_thickness_m"])
    a, b = box.outer_size_m[0] - 2 * t, box.outer_size_m[1] - 2 * t
    c, s = math.cos(yaw), abs(math.sin(yaw))
    ax, ay = a * c - b * s, b * c - a * s               # 기울어진 사각형 안의 축 정렬 사각형 (작은 yaw)
    m = 2 * float(p["fit_margin_m"])
    cols, rows = int((ax - m) // pitch), int((ay - m) // pitch)
    if cols < 1 or rows < 1:
        box.usable = False
        box.note = f"안쪽 {a * 1000:.0f}×{b * 1000:.0f} mm 에 칸 {pitch * 1000:.0f} mm 가 들어가지 않는다"
        return
    box.usable = True
    box.grid = (rows, cols)
    cx, cy = box.center_xy
    box.origin_m = (cx - cols * pitch / 2, cy - rows * pitch / 2, box.floor_z_m)


def side_vector(T: np.ndarray, side: str) -> np.ndarray:
    """배정 쪽 → base xy 단위 벡터. 영상 쪽은 카메라 x(오른쪽)/y(아래) 축을 base 평면에 투영한다."""
    d = str(side).lower().strip()
    base = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}
    if d in base:
        return np.array(base[d])
    image = {"right": (0, 1.0), "left": (0, -1.0), "bottom": (1, 1.0), "top": (1, -1.0)}
    if d not in image:
        raise ValueError(f"workcell.auto_detect.assign 의 쪽은 left|right|top|bottom|+x|-x|+y|-y: {side}")
    axis, sign = image[d]
    v = sign * np.asarray(T, float)[:2, axis]
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        raise ValueError(f"영상 {side} 방향이 base 평면에 거의 수직이다")
    return v / n


def _assign(boxes: list[BoxDetection], T: np.ndarray, p: dict, ref_xy) -> list[str]:
    """assign {이름: 쪽} — 그 쪽에 있는(기준점 ref_xy 보다 그 방향으로 나가 있는) 상자 중 가장 끝에 있는 것.

    기준점 = 트레이 중심 (없으면 카메라 바로 아래). 한 상자를 못 찾았을 때 다른 상자가 엉뚱한 이름을 받지 않는다
    (10/9: 왼쪽 상자를 놓치자 오른쪽 상자가 UPPER 가 될 뻔했다) — 못 찾은 이름은 직전 장면 값을 이어 쓴다.
    """
    remaining = [b for b in boxes if b.usable and not b.flat]
    rule = dict(p["assign"] or {})
    ref = np.asarray(ref_xy, float)
    warnings = []
    for name, side in rule.items():
        v = side_vector(T, side)
        cands = [b for b in remaining if float(np.dot(np.asarray(b.center_xy) - ref, v)) > 0]
        if not cands:
            warnings.append(f"{name}({side}) 쪽 벽 있는 상자를 찾지 못했다")
            continue
        b = max(cands, key=lambda b: float(np.dot(b.center_xy, v)))
        b.name = name
        remaining = [r for r in remaining if r is not b]   # dataclass 의 == 는 ndarray 필드에서 모호하다 — 같은 객체로 뺀다
    for b in remaining:
        b.usable = False
        b.note = "배정 규칙(assign)에 맞는 자리가 없다 — 쓰지 않는다"
    return warnings


# ================================================================ 진입점

def detect_workcell(snap: SceneSnapshot, T_base_camera: np.ndarray, cfg: dict) -> WorkcellDetection:
    """탑뷰 스냅샷 1장 (회색 카드 보정된 컬러) → 상자들·리젝트 종이 구역·사과 집는 영역 (base_link)."""
    p = auto_params(cfg)
    if snap.source != Source.TOP:
        return WorkcellDetection([], warnings=["탑뷰 스냅샷이 아니다"])
    snap = DepthCorrection.from_cfg(cfg).snapshot(snap)
    T = validate_T(T_base_camera)
    x0, y0, res, hmax, hmin, fracs, px = _grids(snap, T, p)
    surface = _table_surface(x0, y0, res, hmin, p)
    if surface is None:
        return WorkcellDetection([], warnings=["테이블 면을 찾지 못했다 (유효 깊이 부족)"])
    above = hmax - surface
    rel = _local_rel(above, res, p)
    # 기준 높이: 카메라 바로 아래 테이블 (깊이 왜곡이 가장 작다). 바닥·종이 높이 = z_ref + 곡면 위 높이
    nx_, ny_ = int((T[0, 3] - x0) / res), int((T[1, 3] - y0) / res)
    ny, nx = surface.shape
    z_ref = float(surface[min(max(ny_, 0), ny - 1), min(max(nx_, 0), nx - 1)])
    cy_, cx_ = px["cell"]
    on_grid = cy_ >= 0
    above_px = np.full(cy_.shape, np.nan)
    rel_px = np.full(cy_.shape, np.nan)
    above_px[on_grid] = above[cy_[on_grid], cx_[on_grid]]
    rel_px[on_grid] = rel[cy_[on_grid], cx_[on_grid]]
    boxes = _find_boxes(snap, T, px["colors"], above_px, rel_px, px["pts"], z_ref, p)
    tray, zone, pick_cells, tray_z = _find_tray(snap, T, above_px, px["pts"], px["colors"], z_ref, x0, y0, res,
                                                rel.shape, p)
    warnings = []
    flat_name = p.get("flat_zone")
    if zone is not None and flat_name:
        zone.name = flat_name
        boxes.append(zone)
    elif flat_name:
        warnings.append(f"트레이 위 흰 종이 구역({flat_name})을 찾지 못했다")

    pitch_of = {name: float(b["cell_pitch_m"]) for name, b in cfg["workcell"]["boxes"].items()}
    default_pitch = min(pitch_of.values()) if pitch_of else 0.09
    for b in boxes:
        _fit_grid(b, pitch_of.get(b.name, default_pitch), p)
    ref_xy = tray[0] if tray is not None else T[:2, 3]
    warnings += _assign(boxes, T, p, ref_xy)
    for b in boxes:                                      # 이름이 정해진 뒤 그 상자 칸 크기로 다시
        if b.usable and b.name in pitch_of:
            _fit_grid(b, pitch_of[b.name], p)
    for b in boxes:
        if not b.usable:
            warnings.append(f"구역 후보 ({b.center_xy[0]:.3f}, {b.center_xy[1]:.3f}) 제외: {b.note}")
        elif b.name not in pitch_of:
            warnings.append(f"{b.name} 은 workcell.boxes 에 없는 이름이다")

    if pick_cells is not None and pick_cells.sum() >= 20:
        center, size = _pick_from_cells(pick_cells, x0, y0, res, tray_z, p)
    else:
        cells = _flat_pick_area(x0, y0, res, rel, boxes, p)
        center, size = (None, None) if cells is None else _pick_from_cells(cells, x0, y0, res, z_ref, p)
        if tray is None:
            warnings.append("파란 트레이를 찾지 못해 테이블 평탄 영역을 집는 영역으로 쓴다")
    if center is None:
        warnings.append("사과 집는 영역을 찾지 못했다")
    return WorkcellDetection(boxes, center, size, tray, warnings,
                             {"x0": x0, "y0": y0, "res": res, "rel": rel, "fracs": fracs})


def apply_detection(cfg: dict, det: WorkcellDetection, *, previous: dict | None = None) -> dict:
    """검출 결과로 workcell.boxes·tray 를 채운 설정 사본.

    - 찾은 상자·구역: origin_m·grid·wall_height_m 을 검출값으로
    - 이번에 못 찾은 상자: previous(직전 장면에서 찾은 상자 설정) → 설정 파일 예비값(origin_m) 순서
      (팔이 가리거나 사과로 가득 차 안 보일 때도 이어서 쓴다)
    - 트레이: 설정에 center_m 이 있으면 그대로(실측 트레이), 없으면 검출한 집는 영역 (벽 0)
    """
    out = copy.deepcopy(cfg)
    wc = out["workcell"]
    found = det.named()
    for name, box in wc["boxes"].items():
        if name in found:
            box.update(found[name].box_cfg(float(box["cell_pitch_m"])))
        elif previous and name in previous:
            box.update(copy.deepcopy(previous[name]))
    tray = wc.setdefault("tray", {})
    if tray.get("center_m") is None and det.pick_center_m is not None:
        tray.update(center_m=[round(v, 4) for v in det.pick_center_m],
                    size_m=[round(v, 4) for v in det.pick_size_m], wall_height_m=0.0,
                    support="tray_z")              # 검출기 받침면 = 이 높이의 수평면 (광택 트레이는 깊이가 빈다)
    return out


class AutoWorkcell:
    """장면마다 검출해 설정 사본을 만든다. 직전에 찾은 상자를 기억해 이번에 못 찾으면 이어 쓴다."""

    def __init__(self, cfg: dict, T_base_camera: np.ndarray) -> None:
        self.base_cfg = cfg
        self.T = np.asarray(T_base_camera, float)
        self.cfg = cfg
        self.last: WorkcellDetection | None = None
        self._remembered: dict[str, dict] = {}
        self._pick: tuple | None = None

    def update(self, snap: SceneSnapshot) -> dict:
        if snap.source != Source.TOP:
            return self.cfg
        det = detect_workcell(snap, self.T, self.base_cfg)
        if det.pick_center_m is None and self._pick is not None:
            det.pick_center_m, det.pick_size_m = self._pick
        self.last = det
        self.cfg = apply_detection(self.base_cfg, det, previous=self._remembered)
        for name in det.named():
            self._remembered[name] = copy.deepcopy(self.cfg["workcell"]["boxes"][name])
        if det.pick_center_m is not None:
            self._pick = (det.pick_center_m, det.pick_size_m)
        return self.cfg
