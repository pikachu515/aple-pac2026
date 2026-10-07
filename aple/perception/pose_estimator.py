"""사과 3D 위치·직경 (설계서 apple_pose_estimator, apple_tracker 축약). 필수.

마스크 내 깊이 점군에 RANSAC + 최소제곱 구 피팅 → 중심·반지름.
탑뷰는 윗반구만 보이므로 반지름을 마스크 픽셀 직경(깊이로 m 환산)과 교차검증한다.
깊이 결측이 많으면 confidence 를 낮추고 depth_valid_ratio 에 기록한다 (G1).
apple_id 는 scene 간 최근접 매칭으로 유지한다 (displacement_m).

구현 (aple/perception/geometry.py sphere_from_silhouette)
- 반지름: 실루엣 원뿔의 반각 α 와 깊이로 구한 중심 거리 D → r = D·sinα. 원근 왜곡에 정확하다
- 중심 거리 D: 마스크 내부 깊이 점마다 구 방정식을 풀어 중앙값, 인라이어(ransac_threshold_m)로 재추정
- 교차검증: 깊이 점만으로 맞춘 반지름과 max_radius_disagreement 이상 다르면 confidence 하향
- 깊이 점이 부족하면(G1) 사과 주변 받침면 깊이와 '구는 받침면에 놓여 있다'로 중심을 추정한다

confidence = 검출 신뢰도 × (아래 문제마다 0.5)
  깊이 유효 비율 < depth.min_valid_ratio (G1) · 실루엣이 이웃·영상 끝에 막힘 ·
  반지름 교차검증 불일치 · fit_residual_m > max_fit_residual_m
→ 한 가지 문제면 대개 inspection.min_confidence(0.5) 근처, 두 가지면 그 아래로 내려가 LOW_CONF 가 된다.

추적: 이전 scene 의 사과와 base_link 거리 tracker.match_radius_m 이내 최근접을 같은 apple_id 로 잇는다.
이번 scene 에서 안 보인 이전 사과는 팔이 트레이를 가린 스냅샷(tray_clear=False)이면 직전 값을 유지하고
occluded=True 로 내보낸다. 가림이 없으면 집어 간 것으로 보고 뺀다.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from aple.hal.calibration import backproject, camera_to_base, validate_T
from aple.perception.geometry import sphere_from_silhouette
from aple.schema import AppleDetection2D, AppleState, SceneSnapshot

_DEFAULTS = {
    "tracker": {"match_radius_m": 0.03},
    "sphere_fit": {"ransac_threshold_m": 0.003, "max_fit_residual_m": None, "max_radius_disagreement": 0.25},
    "depth": {"min_valid_ratio": 0.4},
    "max_clearance_m": 0.2,
    "top_percentile": 95,
}


def pose_params(cfg: dict) -> dict:
    user = cfg.get("perception", {})
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _DEFAULTS.items()}
    for k, v in user.items():
        p[k] = {**p[k], **v} if isinstance(p.get(k), dict) and isinstance(v, dict) else v
    return p


def estimate(snap: SceneSnapshot, dets: list[AppleDetection2D], T_base_camera: np.ndarray,
             prev: list[AppleState] | None, cfg: dict) -> list[AppleState]:
    p = pose_params(cfg)
    T = validate_T(T_base_camera)
    for d in dets:
        if d.scene_id != snap.scene_id:
            raise ValueError(f"검출 scene {d.scene_id} ≠ 스냅샷 scene {snap.scene_id}")
    if prev and any(a.scene_id > snap.scene_id for a in prev):
        raise ValueError("prev 는 이번 스냅샷보다 이전 scene 이어야 한다")

    union = np.zeros(snap.depth_m.shape, bool)
    for d in dets:
        union |= d.mask
    measured = []
    for d in dets:
        m = _measure(snap, d, union & ~d.mask, T, p)
        if m is not None:
            measured.append(m)

    states = _track(snap, measured, prev or [], p)
    _fill_clearance(states, cfg, p)
    return states


# ================================================================ 사과 1개 측정

def _measure(snap: SceneSnapshot, det: AppleDetection2D, others: np.ndarray, T: np.ndarray, p: dict):
    sf = p["sphere_fit"]
    mask = det.mask.astype(bool)
    n_mask = int(mask.sum())
    if n_mask == 0:
        return None
    valid_ratio = float(np.count_nonzero(mask & (snap.depth_m > 0))) / n_mask
    sph = sphere_from_silhouette(mask, snap.depth_m, snap.intrinsics, blocked=others,
                                 inlier_tol_m=float(sf["ransac_threshold_m"]))
    if sph is None:
        return None
    conf = float(det.confidence)
    if valid_ratio < float(p["depth"]["min_valid_ratio"]):
        conf *= 0.5                                                   # G1
    if not sph.silhouette_ok:
        conf *= 0.5

    if math.isfinite(sph.radius_m):
        center_cam, r = sph.center_cam, sph.radius_m
        disagree = abs(sph.radius_depth_m - r) / r if math.isfinite(sph.radius_depth_m) else math.inf
        if disagree > float(sf["max_radius_disagreement"]):
            conf *= 0.5
        lim = sf.get("max_fit_residual_m")
        if lim is not None and sph.fit_residual_m > float(lim):
            conf *= 0.5
        residual = sph.fit_residual_m
    else:                                                             # 깊이 점 부족 → 받침면 기준
        fallback = _center_from_support(snap, mask, others, sph.center_cam, p)
        if fallback is None:
            return None
        center_cam, r = fallback
        residual = float("nan")

    center = camera_to_base(center_cam, T)
    top_z = center[2] + r
    vs, us = np.nonzero(mask & (snap.depth_m > 0))
    if len(us) >= 10:
        z_pts = camera_to_base(backproject(us, vs, snap.depth_m[vs, us], snap.intrinsics), T)[:, 2]
        top_z = float(np.percentile(z_pts, float(p["top_percentile"])))
    return dict(det=det, center=center, r=float(r), top_z=float(top_z), residual=float(residual),
                conf=float(np.clip(conf, 0.0, 1.0)), valid_ratio=valid_ratio)


def _center_from_support(snap, mask, others, c_dir, p):
    """사과 둘레 받침면 깊이 z_s 와 실루엣 방향 c 로: 중심은 받침면보다 r 위 → D = z_s / (c_z + sinα)."""
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool) & ~mask & ~others
    zs = snap.depth_m[ring & (snap.depth_m > 0)]
    if len(zs) < 20:
        return None
    vs, us = np.nonzero(mask)
    rays = backproject(us, vs, np.ones(len(us)), snap.intrinsics)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    sin_a = math.sqrt(max(0.0, 1 - float(np.min(rays @ c_dir)) ** 2))
    D = float(np.median(zs)) / (c_dir[2] + sin_a)
    return D * c_dir, D * sin_a


# ================================================================ 추적 · 간격

def _track(snap: SceneSnapshot, measured: list[dict], prev: list[AppleState], p: dict) -> list[AppleState]:
    radius = float(p["tracker"]["match_radius_m"])
    pairs = sorted(
        ((float(np.linalg.norm(m["center"] - np.asarray(a.center_m))), i, j)
         for i, m in enumerate(measured) for j, a in enumerate(prev)),
        key=lambda t: t[0])
    match: dict[int, int] = {}
    used: set[int] = set()
    for dist, i, j in pairs:
        if dist > radius:
            break
        if i in match or j in used:
            continue
        match[i] = j
        used.add(j)
    next_id = max((a.apple_id for a in prev), default=0) + 1

    out = []
    for i, m in enumerate(measured):
        if i in match:
            old = prev[match[i]]
            apple_id = old.apple_id
            disp = float(np.linalg.norm(m["center"] - np.asarray(old.center_m)))
        else:
            apple_id, next_id, disp = next_id, next_id + 1, 0.0
        out.append(AppleState(
            scene_id=snap.scene_id, apple_id=apple_id, det_id=m["det"].det_id,
            center_m=tuple(float(v) for v in m["center"]), diameter_m=2 * m["r"],
            top_surface_z_m=m["top_z"], clearance_m=0.0, fit_residual_m=m["residual"],
            confidence=m["conf"], depth_valid_ratio=m["valid_ratio"], displacement_m=disp, occluded=False))

    if not snap.tray_clear:                       # 팔에 가려 안 보였을 수 있음 → 직전 값 유지
        for j, a in enumerate(prev):
            if j not in used:
                out.append(AppleState(
                    scene_id=snap.scene_id, apple_id=a.apple_id, det_id=0, center_m=a.center_m,
                    diameter_m=a.diameter_m, top_surface_z_m=a.top_surface_z_m, clearance_m=0.0,
                    fit_residual_m=a.fit_residual_m, confidence=a.confidence,
                    depth_valid_ratio=0.0, displacement_m=0.0, occluded=True))
    return out


def _fill_clearance(states: list[AppleState], cfg: dict, p: dict) -> None:
    """이웃 사과·트레이 벽까지 수평 최소 간격 (m). 0 이상, max_clearance_m 으로 상한."""
    cap = float(p["max_clearance_m"])
    tray = cfg.get("workcell", {}).get("tray", {})
    for a in states:
        r = a.diameter_m / 2
        gaps = [cap]
        for b in states:
            if b is a:
                continue
            dxy = math.hypot(a.center_m[0] - b.center_m[0], a.center_m[1] - b.center_m[1])
            gaps.append(dxy - r - b.diameter_m / 2)
        if tray.get("center_m") is not None:
            cx, cy, _ = tray["center_m"]
            hx, hy = tray["size_m"][0] / 2, tray["size_m"][1] / 2
            gaps += [hx - abs(a.center_m[0] - cx) - r, hy - abs(a.center_m[1] - cy) - r]
        a.clearance_m = float(min(max(min(gaps), 0.0), cap))
