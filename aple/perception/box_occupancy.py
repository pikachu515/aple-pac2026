"""상자 칸별 적재 높이·점유 격자 (설계서 box_occupancy_estimator), 적재 검증 (placement_verifier).

box_occupancy: 필수. placement_verifier: 확장 — 적재 전후 칸 높이 차이로 안착·이탈 확인.
팔이 상자를 가리거나 유효 깊이가 부족하면 valid=False.

상자 기하 (configs/workcell.yaml boxes.<UPPER|MIDDLE|REJECT>, 현장 실측)
- origin_m: 격자 원점 [x, y, z] (base_link). x·y 는 x·y 가 가장 작은 칸 모서리, z 는 상자 바닥 윗면
- 격자는 base_link 축에 정렬: 열(col)은 +x, 행(row)은 +y, 칸 번호 = row·cols + col (row-major)
  → 칸 중심 = origin + ((col+0.5)·pitch, (row+0.5)·pitch). PlacementPlanner 와 같은 규약이다
- origin_m 이 null 인 상자는 측정하지 않는다 (결과 목록에서 빠짐)

출력 높이는 모두 base_link z (m) 절대값이다.
- heights_m[k]: 칸 k 의 상면 높이. 비었으면 바닥(origin z), 사과가 있으면 그 꼭대기
- wall_height_m: 벽 윗면 높이 = origin z + 설정 벽 높이 (PlacementPlanner 가 칸 높이와 같은 기준으로 비교한다)
- wall_height_measured_m: 상자 둘레 띠에서 잰 벽 윗면 높이 (교차검증, 못 재면 nan)
- 깊이가 부족한 칸은 '판단 불가'로 보고 점유·벽 높이로 처리한다 (보수적, 전략 S7-2·S7-3)
"""

from __future__ import annotations

import math

import numpy as np

from aple.hal.calibration import camera_to_base, validate_T
from aple.hal.depth_correction import DepthCorrection
from aple.perception.geometry import depth_points, project_polygon_mask, rect_base
from aple.schema import BoxOccupancy, Grade, SceneSnapshot, Source

_BOX_DEFAULTS = {
    "height_percentile": 90,        # 칸 안 점 높이의 상위 분위 → 칸 상면 (사과 꼭대기 근처)
    "cell_margin_m": 0.01,          # 칸 가장자리·벽 근처 점 제외
    "min_valid_ratio": 0.3,         # 칸 영역 중 유효 깊이 비율 하한
    "min_points": 15,
    "occupied_min_height_m": 0.025, # 바닥보다 이만큼 높으면 점유 (사과 반경보다 작게)
    "wall_band_m": 0.02,            # 벽 윗면을 재는 둘레 띠 폭
    "max_unknown_fraction": 0.5,    # 판단 불가 칸이 이 비율을 넘으면 상자 측정 무효
}


def box_params(cfg: dict) -> dict:
    return {**_BOX_DEFAULTS, **cfg.get("perception", {}).get("box", {})}


def estimate(snap: SceneSnapshot, T_base_camera: np.ndarray, cfg: dict) -> list[BoxOccupancy]:
    if snap.source != Source.TOP:
        return []                                       # 인핸드 스냅샷은 상자를 보지 않는다
    snap = DepthCorrection.from_cfg(cfg).snapshot(snap)
    T = validate_T(T_base_camera)
    p = box_params(cfg)
    valid = snap.depth_m > 0
    pts = camera_to_base(depth_points(snap.depth_m, snap.intrinsics)[valid], T)
    shape = snap.depth_m.shape

    out = []
    for name, box in cfg["workcell"]["boxes"].items():
        if box.get("origin_m") is None:
            continue
        out.append(_estimate_box(snap, Grade[name], box, pts, valid, shape, T, p))
    return out


def _estimate_box(snap, box_id: Grade, box: dict, pts: np.ndarray, valid: np.ndarray, shape, T, p) -> BoxOccupancy:
    rows, cols = (int(v) for v in box["grid"])
    pitch = float(box["cell_pitch_m"])
    x0, y0, z0 = (float(v) for v in box["origin_m"])
    wall_top = z0 + float(box["wall_height_m"])
    margin = float(p["cell_margin_m"])
    heights, occupied, unknown = [], [], 0
    in_view = True

    for k in range(rows * cols):
        row, col = divmod(k, cols)
        cx, cy = x0 + (col + 0.5) * pitch, y0 + (row + 0.5) * pitch
        half = pitch / 2 - margin
        poly = project_polygon_mask(rect_base((cx, cy), (2 * half, 2 * half)), z0, shape, snap.intrinsics, T)
        if poly is None or poly.sum() == 0:
            in_view = False
            heights.append(wall_top)
            occupied.append(True)
            unknown += 1
            continue
        ratio = float(np.count_nonzero(poly & valid)) / float(poly.sum())
        sel = (np.abs(pts[:, 0] - cx) <= half) & (np.abs(pts[:, 1] - cy) <= half)
        z = pts[sel, 2]
        if ratio < p["min_valid_ratio"] or len(z) < p["min_points"]:
            heights.append(wall_top)                        # 판단 불가 → 점유로 (보수적)
            occupied.append(True)
            unknown += 1
            continue
        top = max(float(np.percentile(z, p["height_percentile"])), z0)
        heights.append(top)
        occupied.append(bool(top - z0 >= p["occupied_min_height_m"]))

    sx, sy = cols * pitch, rows * pitch
    band = float(p["wall_band_m"])
    gx, gy = x0 + sx / 2, y0 + sy / 2
    in_outer = (np.abs(pts[:, 0] - gx) <= sx / 2 + band) & (np.abs(pts[:, 1] - gy) <= sy / 2 + band)
    in_inner = (np.abs(pts[:, 0] - gx) < sx / 2) & (np.abs(pts[:, 1] - gy) < sy / 2)
    ring = pts[in_outer & ~in_inner, 2]
    wall_measured = float(np.percentile(ring, 95)) if len(ring) >= p["min_points"] else float("nan")

    ok = in_view and unknown <= p["max_unknown_fraction"] * rows * cols
    return BoxOccupancy(
        scene_id=snap.scene_id, box_id=box_id, rows=rows, cols=cols, cell_pitch_m=pitch,
        wall_height_m=wall_top, wall_height_measured_m=wall_measured, origin_m=(x0, y0, z0),
        heights_m=[float(h) for h in heights], occupied=occupied, valid=bool(ok))


def verify_placement(before: BoxOccupancy, after: BoxOccupancy, cell_index: int,
                     expected_top_z_m: float, *, tol_m: float = 0.01) -> tuple[bool, float]:
    """(성공 여부, 안착 오차 m). 확장.

    성공 = 적재 후 측정이 유효하고, 목표 칸이 점유로 바뀌었고, 칸 상면이 기대 높이(놓인 사과 꼭대기)와
    tol_m 이내. 안착 오차 = 적재 후 칸 상면 − 기대 높이 (측정 무효면 nan).
    """
    if before.box_id != after.box_id:
        raise ValueError("적재 전후 측정의 상자가 다르다")
    if after.scene_id <= before.scene_id:
        raise ValueError("적재 후 측정은 적재 전보다 나중 scene 이어야 한다")
    if not 0 <= cell_index < after.rows * after.cols:
        raise ValueError(f"칸 번호 {cell_index} 가 격자 범위 밖이다")
    if not after.valid:
        return False, float("nan")
    err = float(after.heights_m[cell_index] - expected_top_z_m)
    ok = after.occupied[cell_index] and math.isfinite(err) and abs(err) <= tol_m
    return bool(ok), err
