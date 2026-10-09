"""탑뷰 카메라에서 팔·그리퍼가 사과를 얼마나 가리는지 — Piper 메시 표면 점을 카메라로 투영해 센다.

10/9 실기: 손목을 반지름 방향으로 닫게 돌리면 먼 쪽 핑거와 그리퍼 몸통이 카메라와 사과 사이에 들어와
들어 올린 사과를 깊이로 확인(holding_verifier)하지 못했다. 계산만 한다 — 장치를 쓰지 않는다.
"""
from __future__ import annotations

from importlib import import_module

import numpy as np

#: 표면 점 하나가 가리는 반지름 (메시 점 간격보다 넉넉히)
POINT_RADIUS_M = 0.008


class ArmOcclusion:
    def __init__(self, T_base_camera, tcp_offset_m, *, finger_length_m: float, finger_thickness_m: float,
                 finger_width_m: float, kinematics_module=None, stride: int = 2):
        self._K = kinematics_module or import_module("piper_robot.kinematics")
        self._geom = self._K.geometry()
        names = list(self._geom.names)
        pts = np.asarray(self._geom.pts, dtype=float)
        link = np.asarray(self._geom.pt_link, dtype=int)
        model_fingers = [names.index(n) for n in ("gripper_link1", "gripper_link2") if n in names]
        keep = ~np.isin(link, model_fingers)            # 모형 핑거 대신 수제 핑거를 넣는다
        self._pts, self._link = pts[keep][::stride], link[keep][::stride]
        self._T_cam_base = np.linalg.inv(np.asarray(T_base_camera, dtype=float))
        self._tcp = np.asarray(tcp_offset_m, dtype=float)
        z = np.linspace(-finger_length_m, 0.0, 12)
        y = np.linspace(-finger_width_m / 2, finger_width_m / 2, 3)
        x = np.linspace(0.0, finger_thickness_m, 2)
        self._finger = np.stack(np.meshgrid(x, y, z, indexing="ij"), -1).reshape(-1, 3)

    @classmethod
    def from_config(cls, cfg: dict) -> "ArmOcclusion | None":
        """T_base_camera·TCP 오프셋·piper_robot 이 모두 있을 때만. 없으면 None (가림을 보지 않는다).

        핑거 두께·폭은 grasp_planner 값, 길이는 execution.camera_view.finger_length_m."""
        workcell = cfg.get("workcell") or {}
        view = (cfg.get("execution") or {}).get("camera_view") or {}
        if workcell.get("T_base_camera") is None or workcell.get("tcp_offset_m") is None or not view:
            return None
        from aple.perception.grasp_planner import grasp_params
        p = grasp_params(cfg)
        try:
            return cls(workcell["T_base_camera"], workcell["tcp_offset_m"],
                       finger_length_m=float(view["finger_length_m"]),
                       finger_thickness_m=float(p["finger_thickness_m"]), finger_width_m=float(p["finger_width_m"]))
        except (ImportError, AttributeError, ValueError):
            return None

    def body_points(self, joints_rad, open_width_m: float) -> np.ndarray:
        T = np.asarray(self._K.link_transforms(np.asarray([joints_rad], dtype=float), self._geom))[0]
        Tl = T[self._link]
        arm = np.einsum("nij,nj->ni", Tl[:, :3, :3], self._pts) + Tl[:, :3, 3]
        flange = T[self._geom.tip_index]
        R = flange[:3, :3]
        tcp = R @ self._tcp + flange[:3, 3]
        fingers = []
        for side in (-1.0, 1.0):
            local = self._finger * np.array([side, 1.0, 1.0]) + np.array([side * open_width_m / 2, 0.0, 0.0])
            fingers.append(local @ R.T + tcp)
        return np.vstack([arm, *fingers])

    def fraction(self, joints_rad, apple_center_m, radius_m: float, open_width_m: float) -> float:
        """사과 원판(카메라에서 본) 중 팔·그리퍼에 가린 비율 0~1."""
        c = self._T_cam_base[:3, :3] @ np.asarray(apple_center_m, float) + self._T_cam_base[:3, 3]
        if c[2] <= radius_m:
            return 1.0
        uc, rc = c[:2] / c[2], radius_m / c[2]          # 정규화 영상 좌표 (내부 파라미터와 무관한 비율)
        g = np.linspace(-rc, rc, 15)
        samples = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
        samples = samples[(samples ** 2).sum(1) <= rc * rc] + uc
        p = self.body_points(joints_rad, open_width_m) @ self._T_cam_base[:3, :3].T + self._T_cam_base[:3, 3]
        p = p[(p[:, 2] > 0.05) & (p[:, 2] < c[2] - 0.3 * radius_m)]   # 사과보다 카메라에 가까운 점만
        if len(p) == 0:
            return 0.0
        uv, pr = p[:, :2] / p[:, 2:3], POINT_RADIUS_M / p[:, 2]
        near = np.abs(uv - uc).max(1) <= rc + pr
        uv, pr = uv[near], pr[near]
        if len(uv) == 0:
            return 0.0
        d2 = ((samples[:, None, :] - uv[None, :, :]) ** 2).sum(-1)
        return float((d2 <= pr[None, :] ** 2).any(1).mean())
