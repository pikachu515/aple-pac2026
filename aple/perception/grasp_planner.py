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
}


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
        reach = r + p["open_margin_m"] + p["finger_thickness_m"] / 2
        others = [(np.asarray(b.center_m[:2]), b.diameter_m / 2) for b in apples if b is not a]
        best_yaw, best_clear = 0.0, -math.inf
        for yaw in yaws:
            u = np.array([math.cos(yaw), math.sin(yaw)])
            clear = cap
            for s in (1.0, -1.0):
                f = c + s * reach * u
                for oc, orad in others:
                    clear = min(clear, float(np.linalg.norm(f - oc)) - orad - fr)
                if tray.get("center_m") is not None:
                    tx, ty, _ = tray["center_m"]
                    hx, hy = tray["size_m"][0] / 2, tray["size_m"][1] / 2
                    clear = min(clear, hx - abs(f[0] - tx) - fr, hy - abs(f[1] - ty) - fr)
            if clear > best_clear + 1e-9:                    # 동률이면 먼저 본 yaw (0° 쪽) 유지
                best_yaw, best_clear = float(yaw), clear
        R = rpy_to_matrix(0.0, 0.0, best_yaw) @ R_down
        pose = Pose3D((float(c[0]), float(c[1]), float(a.center_m[2])), quat_xyzw(R))
        reported = float(min(max(best_clear, 0.0), cap))
        out.append(GraspCandidate(
            scene_id=a.scene_id, apple_id=a.apple_id, grasp_pose=pose, grasp_yaw_rad=best_yaw,
            approach_vector=(0.0, 0.0, 1.0), approach_height_m=float(p["approach_height_m"]),
            min_clearance_m=reported, collision_free=bool(best_clear >= p["min_clearance_m"]), score=reported))
    return out


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
