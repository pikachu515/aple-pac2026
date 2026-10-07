"""모드 B 촬영 1장 공급자 — 스냅샷 + 로봇 자세 → InHandCapture (손에 든 사과 마스크, 핑거 제외). 역할 C.

`PiperStudioMotionTransport(in_hand_capture_provider=...)` 에 꽂는다. 트랜스포트는 뷰마다
`provider(apple_id, view_index, joint_state, timeout_s=...)` 를 부른다. 이 호출에는 사과 크기와
파지 위치가 없으므로, 검사를 시작하기 전에 `begin(apple, apple_in_tcp)` 로 한 번 알려 준다.

마스크 (핑거 제외)
  1차  로봇 자세로 예측한 구 표면에서 search_tol_m 안의 깊이 점을 넓게 모은다 (미끄러져도 사과가 잡힐 만큼)
  2차  그 점에 반지름 고정 구를 맞추고, 시선이 그 구를 지나면서 구 앞면까지의 거리와 측정 거리가
       surface_tol_m 안인 픽셀만 사과로 남긴다
  핑거는 사과 앞을 가릴 때 표면보다 카메라 쪽에, 옆을 잡을 때 실루엣 밖에 있으므로 빠진다.
  (실루엣 밖 핑거는 사과 옆면에 붙어 있어 깊이만으로는 구분되지 않지만, 시선이 구를 지나지 않는다)
  마스크는 실제 사과 위치를 따라가고, 예측과의 차이는 재판정의 미끄러짐 검사(check_slip)가 판단한다.

깊이 점이 부족하면(결측) 같은 자세에서 retries 회 다시 찍는다. 그래도 부족하면 그대로 넘기고
판단은 재판정(확인 불가 → 미끄러짐 처리)에 맡긴다.

주입받는 것
  snapshot_fn()          정렬된 컬러·깊이 SceneSnapshot (예: hal.live_snapshot.PiperLiveCamera)
  tool_pose_fn(joints)   TCP 자세 (Pose3D 또는 4×4). 파지 때와 **같은 순기구학**을 써야 구 중심이 맞는다
  T_base_camera          workcell.T_base_camera (4×4)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from aple.execution.goal_builder import Pose3D
from aple.inspection.geometry import Sphere, pixel_rays, ray_sphere
from aple.inspection.in_hand_capture import InHandCapture
from aple.inspection.inhand import apple_center_cam, fit_center_fixed_radius, rot_cam_apple
from aple.schema import AppleState, CaptureMode, SceneSnapshot, Source


@dataclass(frozen=True)
class InHandMaskParams:
    search_tol_m: float      # 1차: 예측 구 표면 ± 이 거리 (미끄러짐 허용 범위)
    surface_tol_m: float     # 2차: 맞춘 구 표면 ± 이 거리만 사과
    roi_margin: float        # 예측 구 투영 반지름에 더하는 여유 (배수)
    min_pixels: int          # 이보다 적으면 다시 찍는다
    retries: int             # 같은 자세에서 다시 찍는 횟수


def mask_params_from_config(cfg: dict) -> InHandMaskParams:
    m = cfg["inspection"]["in_hand_mask"]
    return InHandMaskParams(float(m["search_tol_m"]), float(m["surface_tol_m"]),
                            float(m["roi_margin"]), int(m["min_pixels"]), int(m["retries"]))


def pose_to_matrix(pose: Pose3D | np.ndarray) -> np.ndarray:
    """Pose3D(위치 m, 쿼터니언 xyzw) 또는 4×4 → 4×4."""
    if isinstance(pose, np.ndarray):
        T = np.asarray(pose, float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ValueError("tool pose must be a finite 4x4 matrix")
        return T
    x, y, z, w = (float(v) for v in pose.orientation_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if not math.isfinite(n) or n == 0:
        raise ValueError("tool pose quaternion must be finite and non-zero")
    x, y, z, w = x / n, y / n, z / n, w / n
    T = np.eye(4)
    T[:3, :3] = [[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                 [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                 [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]]
    T[:3, 3] = [float(v) for v in pose.position_m]
    return T


def in_hand_mask(depth_m: np.ndarray, intrinsics: tuple[float, float, float, float],
                 predicted_center: np.ndarray, radius_m: float, p: InHandMaskParams) -> np.ndarray:
    """손에 든 사과의 픽셀 마스크 (H×W bool). 위 모듈 설명의 1차·2차."""
    H, W = depth_m.shape
    mask = np.zeros((H, W), bool)
    c = np.asarray(predicted_center, float)
    if c[2] <= radius_m:
        return mask
    fx, fy, cx, cy = intrinsics
    u0, v0 = fx * c[0] / c[2] + cx, fy * c[1] / c[2] + cy
    r_px = (1 + p.roi_margin) * max(fx, fy) * (radius_m + p.search_tol_m) / (c[2] - radius_m)
    u_lo, u_hi = max(int(u0 - r_px), 0), min(int(u0 + r_px) + 1, W)
    v_lo, v_hi = max(int(v0 - r_px), 0), min(int(v0 + r_px) + 1, H)
    if u_lo >= u_hi or v_lo >= v_hi:
        return mask

    vs, us = np.nonzero(depth_m[v_lo:v_hi, u_lo:u_hi] > 0)
    if len(us) == 0:
        return mask
    vs, us = vs + v_lo, us + u_lo
    rays = pixel_rays(us, vs, intrinsics)
    pts = rays * (depth_m[vs, us] / rays[:, 2])[:, None]

    t_meas = depth_m[vs, us] / rays[:, 2]                                             # 시선 방향 측정 거리

    def along_ray_ok(center: np.ndarray, tol: float) -> np.ndarray:
        # 같은 시선 위에서 구 앞면까지의 거리와 측정 거리를 비교한다. 중심까지 거리로 비교하면
        # 실루엣 가장자리(시선이 구를 스침)에서 앞을 가린 핑거와 표면이 구분되지 않는다
        t_sphere, hits = ray_sphere(rays, Sphere(center, radius_m))
        return hits & (np.abs(t_meas - t_sphere) <= tol)

    near = np.abs(np.linalg.norm(pts - c, axis=1) - radius_m) <= p.search_tol_m      # 1차
    if near.sum() < 10:
        return mask
    center, _, _ = fit_center_fixed_radius(pts[near], radius_m, c)
    for tol in (2 * p.surface_tol_m, p.surface_tol_m, p.surface_tol_m):              # 핑거에 끌린 중심을 다듬는다
        inliers = near & along_ray_ok(center, tol)
        if inliers.sum() < 10:
            break
        center, _, _ = fit_center_fixed_radius(pts[inliers], radius_m, center)

    on_surface = along_ray_ok(center, p.surface_tol_m)                                 # 2차
    mask[vs[on_surface], us[on_surface]] = True
    return mask


class InHandCaptureProvider:
    """PiperStudioMotionTransport 의 in_hand_capture_provider. 사과마다 begin() 후 뷰마다 호출된다."""

    def __init__(self, snapshot_fn: Callable[[], SceneSnapshot],
                 tool_pose_fn: Callable[[object], Pose3D | np.ndarray],
                 T_base_camera: np.ndarray, params: InHandMaskParams, *,
                 scene_id_fn: Callable[[], int] | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        T = np.asarray(T_base_camera, float)
        if T.shape != (4, 4) or not np.all(np.isfinite(T)):
            raise ValueError("T_base_camera must be a finite 4x4 matrix")
        self._snapshot_fn = snapshot_fn
        self._tool_pose_fn = tool_pose_fn
        self._T_base_camera = T
        self._params = params
        self._scene_id_fn = scene_id_fn
        self._clock = clock
        self._apple: AppleState | None = None
        self._apple_in_tcp_m: np.ndarray | None = None
        self._last_scene_id = 0

    @classmethod
    def from_config(cls, cfg: dict, snapshot_fn: Callable[[], SceneSnapshot],
                    tool_pose_fn: Callable[[object], Pose3D | np.ndarray], **kwargs) -> "InHandCaptureProvider":
        T = cfg["workcell"].get("T_base_camera")
        if T is None:
            raise ValueError("workcell.T_base_camera is not set; calibrate the camera before mode-B capture")
        return cls(snapshot_fn, tool_pose_fn, np.asarray(T, float), mask_params_from_config(cfg), **kwargs)

    def begin(self, apple: AppleState, apple_in_tcp: Pose3D) -> None:
        """검사할 사과와 파지 직후 측정한 TCP 기준 사과 위치를 알려 준다 (inspect_views 전에 1회)."""
        p = np.asarray(apple_in_tcp.position_m, float)
        if p.shape != (3,) or not np.all(np.isfinite(p)):
            raise ValueError("apple_in_tcp.position_m must contain three finite values")
        if not math.isfinite(apple.diameter_m) or apple.diameter_m <= 0:
            raise ValueError("apple.diameter_m must be finite and positive")
        self._apple, self._apple_in_tcp_m = apple, p

    def __call__(self, apple_id: int, view_index: int, joint_state, *, timeout_s: float) -> InHandCapture | None:
        apple = self._apple
        if apple is None or self._apple_in_tcp_m is None or apple.apple_id != apple_id:
            raise ValueError("call begin(apple, apple_in_tcp) for this apple before in-hand capture")
        deadline = self._clock() + float(timeout_s)
        T_base_tcp = pose_to_matrix(self._tool_pose_fn(joint_state))
        center = apple_center_cam(self._T_base_camera, T_base_tcp, self._apple_in_tcp_m)
        radius = apple.diameter_m / 2.0

        snap, mask = None, None
        for _ in range(self._params.retries + 1):
            snap = self._snapshot_fn()
            mask = in_hand_mask(snap.depth_m, snap.intrinsics, center, radius, self._params)
            if mask.sum() >= self._params.min_pixels or self._clock() >= deadline:
                break
        if snap is None:
            return None

        snapshot = replace(snap, scene_id=self._next_scene_id(apple), source=Source.IN_HAND,
                           capture_mode=CaptureMode.FULL, apple_id=apple_id, view_index=view_index,
                           tool_pose=T_base_tcp)
        return InHandCapture(snapshot, mask, rot_cam_apple(self._T_base_camera, T_base_tcp))

    def _next_scene_id(self, apple: AppleState) -> int:
        if self._scene_id_fn is not None:
            sid = int(self._scene_id_fn())
        else:
            sid = max(self._last_scene_id, apple.scene_id) + 1
        self._last_scene_id = sid
        return sid
