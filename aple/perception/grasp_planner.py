"""파지 후보 (설계서 grasp_planner — 수직 접근·적도 높이, 간격이 가장 큰 yaw 선택). 역할 A.

역할 B 의 PickPlacePipeline 은 사과마다 GraspCandidate 가 있어야 집기를 계획한다
(aple/execution/motion_executor.py GraspCandidate 프로토콜). 후보가 없거나 collision_free=False 면 DEFER.

계산 (base_link, 위에서 본 2D)
- 핑거는 yaw 방향 u = (cos yaw, sin yaw) 로 닫힌다. 두 핑거 중심 = 사과 중심 ± (r + 개구 여유 + 핑거 두께/2)·u
- 핑거 발자국은 반지름 hypot(두께/2, 폭/2) 원으로 보수적으로 본다
- 간격 = 핑거 발자국과 이웃 사과 원(가려진 사과 포함)·트레이 벽 사이 최소 거리
- yaw 를 grasp_planner.yaw_step_deg 간격으로 [0°, 180°) 훑어 간격 최대를 고른다 (평행 그리퍼는 180° 대칭)
- collision_free = 간격 ≥ grasp_planner.min_clearance_m

출력 자세
- grasp_pose.position = (사과 중심 x, y, 적도 높이 = 중심 z). 높이 오프셋은 역할 B goal_builder 가 더한다
- grasp_pose.orientation = Rz(yaw) · R_down. R_down 은 TCP 가 아래를 보는 자세 (tool_down_rpy_deg, 기본 [180, 0, 0])
  → 역할 B IK 가 기대하는 TCP 축 규약과 같은지 현장 전에 확인한다 (configs/grasp_planner.yaml)
- approach_vector = (0, 0, 1): 집기 전 자세 = 집는 자세 + approach_height_m 위

먼 사과는 기울여 잡는다 (10/9 실기: 수직 파지는 베이스에서 수평 약 0.56 m 까지만 닿는다)
- 베이스에서 수평 거리 d 가 tilt_start_m 을 넘으면 그리퍼 축을 바깥(베이스 반대쪽)으로 기울인다.
  기울기 = max_tilt_deg · (d − tilt_start_m) / (tilt_full_m − tilt_start_m), max_tilt_deg 에서 멈춘다
- 기울 때 핑거는 접선 방향(기울기 축)으로 닫는다 — 두 핑거가 사과 양옆을 같은 높이에서 잡는다
- approach_vector 는 그리퍼 축을 따라 위로 (집기 전 자세도 같은 축 위)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aple.execution.goal_builder import Pose3D
from aple.hal.calibration import rpy_to_matrix
from aple.schema import AppleState

_DEFAULTS = {
    "yaw_step_deg": 15.0,
    "open_margin_m": 0.01,          # 핑거 안쪽 면이 사과 표면에서 떨어지는 여유 (사전 개방 폭 / 2 − r)
    "finger_thickness_m": 0.012,    # 핑거(+어댑터·패드) 닫힘 방향 두께
    "finger_width_m": 0.02,         # 핑거 폭 (닫힘 방향에 수직)
    "min_clearance_m": 0.004,
    "approach_height_m": 0.08,
    "tool_down_rpy_deg": [180.0, 0.0, 0.0],
    "max_clearance_m": 0.2,
    "min_open_margin_m": 0.002,     # 틈이 좁으면 한쪽 여유를 이만큼까지 줄인다 (open_margin_m → 이 값)
    "adaptive_open": False,         # True 면 후보의 open_width_m 으로 그리퍼를 그만큼만 벌린다
    "tilt_yaw_dev_deg": 0.0,        # 기울여 잡을 때 접선 방향에서 yaw 를 이만큼까지 돌려 틈을 찾는다
    "tilt_yaw_free_below_deg": 0.0, # 기울기가 이보다 작으면 yaw 를 제한하지 않는다 (두 핑거 높이 차가 작다)
    "tray_wall_margin_m": 0.0,      # 트레이 사각형을 이만큼 넓혀 벽으로 본다
    "tray_open_sides": [],          # 벽이 없는 변: x_min · x_max · y_min · y_max (base 축)
    "tilt_start_m": None,           # null = 기울이지 않는다 (항상 수직)
    "tilt_full_m": 0.62,
    "max_tilt_deg": 25.0,
}


def tilt_for(center_xy, p: dict) -> float:
    """베이스에서 수평 거리에 따른 그리퍼 기울기 (rad, 바깥쪽)."""
    start = p.get("tilt_start_m")
    if start is None:
        return 0.0
    d = math.hypot(float(center_xy[0]), float(center_xy[1]))
    full = float(p["tilt_full_m"])
    frac = 0.0 if d <= start else min(1.0, (d - start) / max(full - start, 1e-6))
    return math.radians(float(p["max_tilt_deg"])) * frac


def _rot_axis(axis, angle: float) -> np.ndarray:
    k = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


@dataclass(frozen=True)
class GraspCandidate:
    """역할 B GraspCandidate 프로토콜을 만족하는 M1 출력."""
    scene_id: int
    apple_id: int
    grasp_pose: Pose3D
    grasp_yaw_rad: float
    approach_vector: tuple[float, float, float]
    approach_height_m: float
    min_clearance_m: float
    collision_free: bool
    score: float
    open_width_m: float | None = None   # 틈에 맞춘 사전 개방 폭 (핑거 안쪽 면 사이). None = 그리퍼 기본값


def grasp_params(cfg: dict) -> dict:
    return {**_DEFAULTS, **(cfg.get("grasp_planner") or {})}


def plan_grasps(apples: list[AppleState], cfg: dict) -> list[GraspCandidate]:
    """보이는 사과마다 후보 1개 (간격이 가장 큰 yaw). 가려진 사과(occluded)는 후보를 내지 않는다."""
    p = grasp_params(cfg)
    tray = cfg.get("workcell", {}).get("tray", {})
    step = math.radians(float(p["yaw_step_deg"]))
    yaws = np.arange(0.0, math.pi - 1e-9, step)
    R_down = rpy_to_matrix(*np.deg2rad(p["tool_down_rpy_deg"]))
    fr = math.hypot(p["finger_thickness_m"] / 2, p["finger_width_m"] / 2)
    cap = float(p["max_clearance_m"])

    out = []
    for a in apples:
        if a.occluded:
            continue
        r = a.diameter_m / 2
        c = np.asarray(a.center_m[:2])
        others = [(np.asarray(b.center_m[:2]), b.diameter_m / 2) for b in apples if b is not a]
        tilt = tilt_for(c, p)
        phi = math.atan2(float(c[1]), float(c[0]))
        if 0 < tilt and math.degrees(tilt) >= float(p["tilt_yaw_free_below_deg"]):
            # 많이 기울이면 접선 방향 근처로만 닫는다 — 반지름 방향으로 닫으면 두 핑거 높이가 크게 달라진다
            dev = math.radians(float(p["tilt_yaw_dev_deg"]))
            base_yaw = (phi + math.pi / 2) % math.pi
            yaws_here = np.array([base_yaw + d for d in np.arange(-dev, dev + 1e-9, step)]) if dev > 0 \
                else np.array([base_yaw])
        else:
            yaws_here = yaws
        # 틈에 맞춰 벌린다 (10/9 실기: 붙어 놓인 사과 줄). 한쪽 여유 m 을 open_margin_m 부터 min_open_margin_m
        # 까지 줄여 가며, 핑거 발자국이 이웃 사과·트레이 벽과 min_clearance_m 이상 떨어지는 가장 큰 m 을 쓴다.
        margins = np.arange(float(p["open_margin_m"]), float(p["min_open_margin_m"]) - 1e-9, -0.001)
        best = None                                     # (가능, m, 간격, yaw)
        for yaw in yaws_here:
            u = np.array([math.cos(yaw), math.sin(yaw)])
            for m in margins:
                clear = _finger_clearance(c, u, r + m + p["finger_thickness_m"] / 2, others, tray, fr, cap, p)
                ok = clear >= p["min_clearance_m"]
                key = (ok, float(m) if ok else 0.0, clear)
                if best is None or key > best[:3]:
                    best = (*key, float(yaw) % math.pi)
                if ok:
                    break                               # 이 yaw 에서 가능한 가장 넓은 여유를 찾았다
        ok, m_best, best_clear, best_yaw = best
        open_width = 2 * (r + (m_best if ok else float(p["open_margin_m"])))
        R = rpy_to_matrix(0.0, 0.0, best_yaw) @ R_down
        approach = (0.0, 0.0, 1.0)
        if tilt > 0:
            # 축 k = 반지름 방향에 수직(수평). −tilt 회전이면 그리퍼 z(아래)가 베이스 반대쪽으로 기운다
            R = _rot_axis((-math.sin(phi), math.cos(phi), 0.0), -tilt) @ R
            approach = tuple(float(v) for v in -R[:, 2])  # 그리퍼 축을 따라 사과에서 멀어지는 쪽
        pose = Pose3D((float(c[0]), float(c[1]), float(a.center_m[2])), quat_xyzw(R))
        reported = float(min(max(best_clear, 0.0), cap))
        out.append(GraspCandidate(
            scene_id=a.scene_id, apple_id=a.apple_id, grasp_pose=pose, grasp_yaw_rad=best_yaw,
            approach_vector=approach, approach_height_m=float(p["approach_height_m"]),
            min_clearance_m=reported, collision_free=bool(ok), score=reported + (m_best if ok else 0.0),
            open_width_m=float(open_width) if p["adaptive_open"] else None))
    return out


def _finger_clearance(c, u, reach, others, tray, fr, cap, p) -> float:
    """핑거 두 발자국(사과 중심 ± reach·u)과 이웃 사과·트레이 벽 사이 최소 간격."""
    clear = cap
    for s in (1.0, -1.0):
        f = c + s * reach * u
        for oc, orad in others:
            clear = min(clear, float(np.linalg.norm(f - oc)) - orad - fr)
        if tray.get("center_m") is not None:
            tx, ty, _ = tray["center_m"]
            wm = float(p["tray_wall_margin_m"])
            hx, hy = tray["size_m"][0] / 2 + wm, tray["size_m"][1] / 2 + wm
            open_sides = set(p["tray_open_sides"] or ())   # 벽이 없는 변 (자동 집는 영역의 종이 쪽 경계)
            for axis, (ctr, half) in enumerate(((tx, hx), (ty, hy))):
                d = f[axis] - ctr
                side = ("x" if axis == 0 else "y") + ("_min" if d < 0 else "_max")
                if side not in open_sides:
                    clear = min(clear, half - abs(d) - fr)
    return clear


def tool_orientation_at(center_xy, cfg: dict, yaw_rad: float | None = None) -> tuple[float, float, float, float]:
    """그 위치에서 쓸 TCP 자세 (x, y, z, w) — plan_grasps 와 같은 규칙 (먼 곳은 바깥으로 기울이고 접선으로 닫는다).

    놓을 때 쓴다: 사과는 구라서 놓는 위치에 맞춰 그리퍼 방향을 다시 정해도 된다. yaw_rad = 기울이지 않을 때 쓸 yaw."""
    p = grasp_params(cfg)
    R_down = rpy_to_matrix(*np.deg2rad(p["tool_down_rpy_deg"]))
    tilt = tilt_for(center_xy, p)
    phi = math.atan2(float(center_xy[1]), float(center_xy[0]))
    yaw = (phi + math.pi / 2) % math.pi if tilt > 0 else float(yaw_rad or 0.0)
    R = rpy_to_matrix(0.0, 0.0, yaw) @ R_down
    if tilt > 0:
        R = _rot_axis((-math.sin(phi), math.cos(phi), 0.0), -tilt) @ R
    return quat_xyzw(R)


def quat_xyzw(R: np.ndarray) -> tuple[float, float, float, float]:
    """회전 행렬 → 단위 쿼터니언 (x, y, z, w), w ≥ 0."""
    t = float(np.trace(R))
    if t > 0:
        s = 2 * math.sqrt(t + 1)
        q = ((R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4)
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * math.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        v = [0.0, 0.0, 0.0]
        v[i], v[j], v[k] = s / 4, (R[j, i] + R[i, j]) / s, (R[k, i] + R[i, k]) / s
        q = (v[0], v[1], v[2], (R[k, j] - R[j, k]) / s)
    q = np.asarray(q, dtype=float)
    q /= np.linalg.norm(q)
    if q[3] < 0:
        q = -q
    return tuple(float(v) for v in q)
