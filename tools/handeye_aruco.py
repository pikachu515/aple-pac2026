"""카메라–로봇 변환 자동 캘리브레이션 — 그리퍼에 붙인 ArUco 마커를 탑뷰 카메라로 찍는다 (eye-to-hand, 역할 A).

touch_calib 은 사람이 끝점을 표식에 대야 해서 14 cm 그리퍼에서는 손목 각도·누르는 힘에 cm 단위로 흔들렸다
(10/9 실기 RMS 5~47 mm). 이 도구는 마커를 카메라가 자동으로 찾고, 같은 순간의 관절각으로 플랜지 자세를
구해 짝지어 푼다. 끝점에 댈 필요도, TCP 오프셋을 알 필요도 없다 (마커 ↔ 플랜지 변환도 같이 풀린다).

준비물
- tools 로 만든 A4 마커 (data/calib/aruco_markers_A4.pdf, DICT_4X4_50). id 0 = 60 mm, id 1 = 45 mm.
  '실제 크기(100%)' 로 인쇄하고 검은 사각형 한 변을 자로 재서 --marker-mm 로 준다.
- 마커를 그리퍼/손목 윗면에 **움직이지 않게** 붙인다 (평평한 판에 붙인 뒤 테이프). 흰 여백을 자르지 않는다.

절차 (Piper Studio 켠 채, 팔로워는 리더암 텔레오퍼레이션으로 움직인다)
  1. 자세마다 Enter — 스냅샷 + 관절각을 함께 기록한다 (10~15 자세)
       python tools/handeye_aruco.py capture --session data/calib/handeye.json --iface can_follower \\
           --marker-id 0 --marker-mm 60
     키: Enter = 기록   u = 마지막 자세 취소   q = 끝 (기록할 때마다 저장된다)
     요령: 작업 영역 전체로 위치를 옮기고, 손목을 ±30° 여러 방향으로 기울이고 돌린다 (회전 폭 30° 이상).
           마커는 늘 탑뷰 카메라에 보여야 한다. 팔이 멈춘 뒤 Enter.
  2. 맞추고 점검한 뒤 저장한다
       python tools/handeye_aruco.py fit --session data/calib/handeye.json                       # 결과만
       python tools/handeye_aruco.py fit --session data/calib/handeye.json --save configs/workcell.yaml
  3. (선택) 저장된 T_base_camera 로 지금 자세를 검증한다
       python tools/handeye_aruco.py check --iface can_follower --marker-id 0 --marker-mm 60

piper_robot(순기구학) 은 PYTHONPATH 로 (docs/simulation.md). 카메라 왜곡은 0 으로 본다 (d435.json 컬러 계수 0).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aple.config import load_config  # noqa: E402
from aple.hal.calibration import (load_T_base_camera, save_T_base_camera, solve_eye_to_hand,  # noqa: E402
                                  validate_T)
from aple.hal.shm_frames import SHM_DIR  # noqa: E402

DICT = cv2.aruco.DICT_4X4_50
MIN_POSES = 8
MIN_SPREAD_DEG = 30.0
MAX_MEDIAN_RES_M = 0.004


# ================================================================ 마커 검출 · 자세

def marker_object_points(size_m: float) -> np.ndarray:
    """마커 좌표계의 네 모서리 (SOLVEPNP_IPPE_SQUARE 순서: 좌상·우상·우하·좌하, z = 0)."""
    h = size_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


def detect_marker(bgr: np.ndarray, marker_id: int) -> np.ndarray | None:
    """→ 4×2 픽셀 모서리 (검출 순서 = 마커 좌상부터 시계방향) 또는 None."""
    gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return None
    for c, i in zip(corners, ids.ravel()):
        if int(i) == marker_id:
            return c.reshape(4, 2).astype(np.float64)
    return None


def _K(intrinsics) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def marker_pose(corners: np.ndarray, size_m: float, intrinsics, depth_m: np.ndarray | None = None
                ) -> tuple[np.ndarray, float, float | None]:
    """→ (T_camera_marker, 재투영 오차 px, 깊이 축척 또는 None).

    작은 마커(0.9 m 에서 40~50 px)의 PnP 는 시선 방향 거리가 가장 흔들린다. 정렬 깊이가 마커 안에 충분하면
    마커 중심 거리를 깊이 중앙값에 맞춰 축척한다 (방향은 PnP 그대로).
    """
    obj = marker_object_points(size_m)
    K = _K(intrinsics)
    ok, rvec, tvec = cv2.solvePnP(obj, corners, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        raise ValueError("solvePnP 실패")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(4, 2) - corners) ** 2, axis=1))))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    scale = None
    if depth_m is not None:
        mask = np.zeros(depth_m.shape, np.uint8)
        center = corners.mean(axis=0)
        inner = (center + 0.6 * (corners - center)).astype(np.int32)       # 가장자리 혼합 픽셀 제외
        cv2.fillConvexPoly(mask, inner, 1)
        d = depth_m[(mask > 0) & (depth_m > 0)]
        if len(d) >= 20:
            scale = float(np.median(d) / T[2, 3])
            if 0.9 < scale < 1.1:                                          # 크게 다르면 깊이 쪽이 틀렸다
                T[:3, 3] *= scale
            else:
                scale = None
    return T, reproj, scale


# ================================================================ 풀이

def rotation_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1) / 2, -1, 1))))


def rotation_spread_deg(flanges: list[np.ndarray]) -> float:
    """기록한 플랜지 자세 사이의 최대 회전각 (deg)."""
    if len(flanges) < 2:
        return 0.0
    return max(rotation_angle_deg(a[:3, :3], b[:3, :3]) for i, a in enumerate(flanges) for b in flanges[i + 1:])


def _mean_transform(Ts: list[np.ndarray]) -> np.ndarray:
    """강체 변환 평균 — 회전은 행렬 합의 SVD 투영, 이동은 평균."""
    U, _, Vt = np.linalg.svd(sum(T[:3, :3] for T in Ts))
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    out = np.eye(4)
    out[:3, :3], out[:3, 3] = R, np.mean([T[:3, 3] for T in Ts], axis=0)
    return out


def _T_from(x: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(x[:3], dtype=np.float64))[0]
    T[:3, 3] = x[3:6]
    return T


def _x_from(T: np.ndarray) -> np.ndarray:
    return np.r_[cv2.Rodrigues(T[:3, :3])[0].ravel(), T[:3, 3]]


def _refine(F, size_m, T_bc, T_fm, corners_px, intrinsics, iters=50):
    """T_base_camera·T_flange_marker 를 함께 마커 모서리 재투영 오차(px)로 다듬는다 (Levenberg–Marquardt).

    재투영 오차는 PnP 의 시선 방향(거리) 흔들림에 영향을 받지 않는다. 카메라 좌표 모서리(3D)로 다듬으면
    그 흔들림이 그대로 들어가 오히려 나빠졌다 (합성 시험: 1.6 → 6.5 mm).
    """
    obj_h = np.c_[marker_object_points(size_m), np.ones(4)]
    K = _K(intrinsics)
    obs = [np.asarray(c, dtype=np.float64) for c in corners_px]

    def resid(x):
        T_cb, T_fm_ = np.linalg.inv(_T_from(x[:6])), _T_from(x[6:])
        out = []
        for Fi, o in zip(F, obs):
            uv = (K @ (T_cb @ Fi @ T_fm_ @ obj_h.T)[:3]).T
            out.append((uv[:, :2] / uv[:, 2:3] - o).ravel())
        return np.concatenate(out)

    x = np.r_[_x_from(T_bc), _x_from(T_fm)]
    r = resid(x)
    lam = 1e-3
    for _ in range(iters):
        J = np.empty((len(r), len(x)))
        for j in range(len(x)):
            dx = np.zeros_like(x)
            dx[j] = 1e-7
            J[:, j] = (resid(x + dx) - r) / 1e-7
        A, g = J.T @ J, J.T @ r
        while True:
            step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), -g)
            r_new = resid(x + step)
            if r_new @ r_new < r @ r:
                x, r, lam = x + step, r_new, max(lam / 3, 1e-9)
                break
            lam *= 4
            if lam > 1e8:
                return _T_from(x[:6]), _T_from(x[6:])
        if np.abs(step).max() < 1e-10:
            break
    return _T_from(x[:6]), _T_from(x[6:])


def solve(flanges: list[np.ndarray], cam_markers: list[np.ndarray], size_m: float,
          corners_px: list | None = None, intrinsics=None) -> tuple[np.ndarray, np.ndarray]:
    """→ (T_base_camera, T_flange_marker).

    OpenCV calibrateHandEye(Park) 로 풀고 — 합성 잡음 시험에서 Tsai·Daniilidis 보다 안정적이었다 —
    픽셀 모서리·내부 파라미터가 있으면 두 변환을 함께 재투영 오차로 다듬는다.
    """
    F = [validate_T(T, "T_base_flange") for T in flanges]
    C = [validate_T(T, "T_camera_marker") for T in cam_markers]
    T_bc = solve_eye_to_hand(F, C, method=cv2.CALIB_HAND_EYE_PARK)
    T_fm = _mean_transform([np.linalg.inv(Fi) @ T_bc @ Ci for Fi, Ci in zip(F, C)])
    if corners_px is not None and intrinsics is not None:
        T_bc, T_fm = _refine(F, size_m, T_bc, T_fm, corners_px, intrinsics)
    return T_bc, T_fm


def residuals(flanges, cam_markers, T_bc: np.ndarray, T_fm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """자세별 (마커 중심 위치 오차 m, 마커 방향 오차 deg) — 카메라로 본 것 vs FK 로 예측한 것."""
    pos, rot = [], []
    for Fi, Ci in zip(flanges, cam_markers):
        seen, pred = T_bc @ np.asarray(Ci), np.asarray(Fi) @ T_fm
        pos.append(np.linalg.norm(seen[:3, 3] - pred[:3, 3]))
        rot.append(rotation_angle_deg(seen[:3, :3], pred[:3, :3]))
    return np.array(pos), np.array(rot)


def fit_warnings(n: int, spread_deg: float, res_m: np.ndarray, T_bc: np.ndarray) -> list[str]:
    warn = []
    if n < MIN_POSES:
        warn.append(f"자세가 {n}개다 — {MIN_POSES}~15개를 찍는다")
    if spread_deg < MIN_SPREAD_DEG:
        warn.append(f"자세 간 최대 회전이 {spread_deg:.0f}° 다 — 손목을 ±30° 여러 방향으로 더 기울인다")
    if len(res_m) and np.median(res_m) > MAX_MEDIAN_RES_M:
        warn.append(f"잔차 중앙값 {np.median(res_m) * 1000:.1f} mm 가 기준 {MAX_MEDIAN_RES_M * 1000:.0f} mm 를 넘는다 "
                    "— 마커가 그리퍼에서 흔들리는지, 팔이 멈춘 뒤 찍었는지 확인")
    if len(res_m) >= 4:
        med = float(np.median(res_m))
        for i in np.nonzero(res_m > max(3 * med, 0.006))[0]:
            warn.append(f"{i + 1}번째 자세 잔차 {res_m[i] * 1000:.1f} mm 가 유난히 크다 — --drop {i + 1} 로 빼고 다시")
    if T_bc[2, 2] > -0.5:
        warn.append("카메라 광축이 아래를 보지 않는다 — 마커 크기(--marker-mm)나 관절각 단위를 확인")
    return warn


# ================================================================ 장치 (실기에서만)

def grab_snapshot(args):
    """Piper Studio 공유 메모리에서 컬러 + 컬러에 정렬한 깊이 한 장 (touch_calib grab 과 같은 경로)."""
    from aple.hal.depth_align import D435Calib, DepthEncoding, snapshot_from_piper
    from aple.hal.shm_frames import ShmError, ShmReader, find_realsense_serials, read_pair, segment_path

    if not args.calib.exists():
        raise SystemExit(f"깊이→컬러 보정 파일이 없다: {args.calib}")
    serial = args.serial or next(iter(find_realsense_serials(args.shm_dir)), None)
    if serial is None:
        raise SystemExit("공유 메모리에 RealSense 영상이 없다 — Piper Studio 에서 D435 를 켰는지 확인")
    try:
        color_r = ShmReader(segment_path(f"rs:{serial}:color", args.shm_dir))
        depth_r = ShmReader(segment_path(f"rs:{serial}:depth", args.shm_dir))
    except ShmError as e:
        raise SystemExit(str(e))
    pair = read_pair(color_r, depth_r, timeout_s=3.0)
    if pair is None:
        raise SystemExit("컬러·깊이 프레임을 받지 못했다 — 스트림이 켜져 있는지 확인")
    color, depth_enc, _ = pair
    return snapshot_from_piper(color, depth_enc, DepthEncoding(args.near, args.far), D435Calib.load(args.calib),
                               0, time.time())


def flange_reader(iface: str | None):
    """→ (fk, joint_state). 관절각 읽기만 한다 — iface 를 주면 이 실행에서만 켜고 설정 파일은 그대로."""
    from importlib import import_module

    from aple.hal.piper_studio import PiperStudio
    try:
        kin = import_module("piper_robot.kinematics")
    except ImportError as e:
        raise SystemExit("piper_robot.kinematics 를 불러올 수 없다 — PYTHONPATH 에 Piper Studio 의 piper_robot 을 "
                         "넣는다 (docs/simulation.md)") from e
    geom = kin.geometry()
    if geom.tip != "link6":
        raise SystemExit(f"piper_robot 기구학의 끝 링크가 link6 이 아니다 ({geom.tip})")

    def fk(q):
        return np.asarray(kin.link_transforms(np.asarray(q, dtype=float)[None, :], geom), dtype=float)[0, geom.tip_index]

    cfg = load_config()
    if iface:
        cfg["piper_web"] = {**cfg["piper_web"], "enabled": True, "iface": iface}
    studio = PiperStudio(cfg)
    return fk, studio.joint_state


# ================================================================ 세션 파일

def load_session(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"marker_id": None, "marker_m": None, "poses": []}


def save_session(path: Path, session: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, ensure_ascii=False, indent=1), encoding="utf-8")


def _arrays(session: dict, drop=()) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    keep = [i for i in range(len(session["poses"])) if (i + 1) not in set(drop)]
    F = [np.asarray(session["poses"][i]["T_base_flange"], dtype=float) for i in keep]
    C = [np.asarray(session["poses"][i]["T_camera_marker"], dtype=float) for i in keep]
    return F, C, keep


# ================================================================ 명령

def record_pose(snap, joints, fk, marker_id: int, size_m: float, use_depth: bool = True) -> dict | None:
    """스냅샷 + 관절각 → 세션 자세 한 개 (마커가 없으면 None)."""
    corners = detect_marker(snap.color, marker_id)
    if corners is None:
        return None
    T_cm, reproj, scale = marker_pose(corners, size_m, snap.intrinsics, snap.depth_m if use_depth else None)
    T_bf = np.asarray(fk(joints), dtype=float)
    return {"joints_rad": [round(float(q), 6) for q in joints], "T_base_flange": T_bf.round(6).tolist(),
            "T_camera_marker": T_cm.round(6).tolist(), "corners_px": corners.round(2).tolist(),
            "reproj_px": round(reproj, 3), "depth_scale": None if scale is None else round(scale, 4),
            "stamp_s": snap.stamp_s}


def cmd_capture(args) -> None:
    session = load_session(args.session)
    size_m = args.marker_mm / 1000
    if session["poses"] and (session["marker_id"] != args.marker_id or abs(session["marker_m"] - size_m) > 1e-6):
        raise SystemExit("세션의 마커 id·크기가 다르다 — 다른 세션 파일을 쓴다")
    session["marker_id"], session["marker_m"] = args.marker_id, size_m
    fk, joint_state = flange_reader(args.iface)
    img_dir = args.session.parent / "handeye_imgs" if args.save_images else None
    print(f"자세마다 팔을 멈추고 Enter (u = 취소, q = 끝). 지금 {len(session['poses'])}개")
    while True:
        text = input(f"자세 {len(session['poses']) + 1} > ").strip().lower()
        if text == "q":
            break
        if text == "u":
            if session["poses"]:
                session["poses"].pop()
                save_session(args.session, session)
                print("  마지막 자세 취소")
            continue
        snap = grab_snapshot(args)
        pose = record_pose(snap, joint_state(), fk, args.marker_id, size_m, not args.no_depth)
        if pose is None:
            print("  [다시] 마커를 못 찾았다 — 카메라에 마커 전체가 보이는지, 반사·흐림이 없는지 확인")
            continue
        session["intrinsics"] = [float(v) for v in snap.intrinsics]
        session["poses"].append(pose)
        save_session(args.session, session)
        if img_dir is not None:
            img_dir.mkdir(parents=True, exist_ok=True)
            img = snap.color.copy()
            cv2.polylines(img, [np.asarray(pose["corners_px"], np.int32)], True, (0, 0, 255), 2)
            cv2.imwrite(str(img_dir / f"pose_{len(session['poses']):02d}.png"), img)
        F = [np.asarray(p["T_base_flange"]) for p in session["poses"]]
        z = pose["T_camera_marker"][2][3]
        print(f"  기록 — 마커 거리 {z * 1000:.0f} mm, 재투영 {pose['reproj_px']:.2f} px"
              + ("" if pose["depth_scale"] is None else f", 깊이 축척 {pose['depth_scale']:.3f}")
              + f" | 자세 {len(F)}개, 최대 회전 {rotation_spread_deg(F):.0f}°")
    print(f"완료 — {len(session['poses'])}개. 다음: fit")


def cmd_fit(args) -> tuple[np.ndarray, np.ndarray]:
    session = load_session(args.session)
    F, C, keep = _arrays(session, args.drop)
    if len(F) < 3:
        raise SystemExit(f"자세가 {len(F)}개다 — capture 로 8~15개 찍는다")
    corners = [session["poses"][i]["corners_px"] for i in keep] if session.get("intrinsics") else None
    T_bc, T_fm = solve(F, C, session["marker_m"], corners, session.get("intrinsics"))
    res, rot = residuals(F, C, T_bc, T_fm)
    spread = rotation_spread_deg(F)
    print(f"{'자세':<5}{'잔차 mm':>9}{'방향 °':>8}{'재투영 px':>11}")
    for i, r, a in zip(keep, res, rot):
        print(f"{i + 1:<5}{r * 1000:>9.1f}{a:>8.2f}{session['poses'][i]['reproj_px']:>11.2f}")
    print(f"\n자세 {len(F)}개  잔차 중앙값 {np.median(res) * 1000:.1f} mm / 최대 {res.max() * 1000:.1f} mm, 최대 회전 {spread:.0f}°")
    print(f"카메라 위치 (base_link) ({', '.join(f'{v * 1000:.1f}' for v in T_bc[:3, 3])}) mm, "
          f"광축 방향 ({', '.join(f'{v:.3f}' for v in T_bc[:3, 2])})")
    print(f"마커 ← 플랜지 ({', '.join(f'{v * 1000:.1f}' for v in T_fm[:3, 3])}) mm (마커를 붙인 위치와 맞는지 본다)")
    tcp = load_config()["workcell"].get("tcp_offset_m")
    if tcp is None:
        print("참고: workcell.tcp_offset_m 이 비어 있다 — 실행 전에 핑거 끝 오프셋을 따로 넣는다 (이 보정과는 무관)")
    warn = fit_warnings(len(F), spread, res, T_bc)
    for w in warn:
        print(f"[경고] {w}")
    session["fit"] = {"T_base_camera": T_bc.round(6).tolist(), "T_flange_marker": T_fm.round(6).tolist(),
                      "residual_median_m": float(np.median(res)), "residual_max_m": float(res.max()),
                      "spread_deg": spread, "used": [i + 1 for i in keep], "warnings": warn}
    save_session(args.session, session)
    if args.save:
        if warn and not args.force:
            raise SystemExit("경고가 있어 저장하지 않았다 — 확인 후 --force 로 저장하거나 자세를 더 찍는다")
        note = (f"handeye_aruco {len(F)}자세 잔차 중앙 {np.median(res) * 1000:.1f} mm "
                f"({time.strftime('%m/%d')}, {args.session.name})")
        line = save_T_base_camera(str(args.save), T_bc, note=note)
        print(f"저장: {args.save}\n  {line[:100]}...")
    return T_bc, T_fm


def cmd_check(args) -> None:
    session = load_session(args.session)
    if not session.get("fit"):
        raise SystemExit("세션에 fit 결과(T_flange_marker)가 없다 — fit 을 먼저 한다")
    T_fm = np.asarray(session["fit"]["T_flange_marker"], dtype=float)
    T_bc = (np.asarray(session["fit"]["T_base_camera"], dtype=float) if args.from_session
            else load_T_base_camera(str(args.workcell)))
    fk, joint_state = flange_reader(args.iface)
    snap = grab_snapshot(args)
    pose = record_pose(snap, joint_state(), fk, args.marker_id, args.marker_mm / 1000, not args.no_depth)
    if pose is None:
        raise SystemExit("마커를 못 찾았다")
    res, rot = residuals([np.asarray(pose["T_base_flange"])], [np.asarray(pose["T_camera_marker"])], T_bc, T_fm)
    print(f"카메라로 본 마커 vs 순기구학 예측: 위치 {res[0] * 1000:.1f} mm, 방향 {rot[0]:.2f}°")


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def device_args(p):
        p.add_argument("--iface", help="관절각을 읽을 팔로워 (예: can_follower)")
        p.add_argument("--marker-id", type=int, default=0)
        p.add_argument("--marker-mm", type=float, default=60.0, help="검은 사각형 한 변 (mm, 인쇄물을 자로 잰 값)")
        p.add_argument("--no-depth", action="store_true", help="정렬 깊이로 마커 거리를 보정하지 않는다")
        p.add_argument("--calib", type=Path, default=ROOT / "configs" / "calib" / "d435.json")
        p.add_argument("--near", type=float, default=150.0, help="Piper Studio 깊이 인코딩 near_mm")
        p.add_argument("--far", type=float, default=1200.0, help="Piper Studio 깊이 인코딩 far_mm")
        p.add_argument("--serial")
        p.add_argument("--shm-dir", type=Path, default=SHM_DIR)

    c = sub.add_parser("capture", help="자세마다 스냅샷 + 관절각 기록")
    c.add_argument("--session", type=Path, default=ROOT / "data" / "calib" / "handeye.json")
    c.add_argument("--save-images", action="store_true", help="자세마다 검출 표시 영상을 handeye_imgs/ 에 저장")
    device_args(c)

    f = sub.add_parser("fit", help="T_base_camera 를 풀고 점검, --save 로 workcell.yaml 에 저장")
    f.add_argument("--session", type=Path, default=ROOT / "data" / "calib" / "handeye.json")
    f.add_argument("--drop", type=int, nargs="*", default=[], help="뺄 자세 번호 (1부터)")
    f.add_argument("--save", type=Path, help="T_base_camera 를 쓸 workcell.yaml")
    f.add_argument("--force", action="store_true", help="경고가 있어도 저장")

    k = sub.add_parser("check", help="저장된 T_base_camera 로 지금 자세 검증")
    k.add_argument("--session", type=Path, default=ROOT / "data" / "calib" / "handeye.json")
    k.add_argument("--workcell", type=Path, default=ROOT / "configs" / "workcell.yaml")
    k.add_argument("--from-session", action="store_true", help="workcell.yaml 대신 세션의 fit 결과를 쓴다")
    device_args(k)

    args = ap.parse_args(argv)
    return {"capture": cmd_capture, "fit": cmd_fit, "check": cmd_check}[args.cmd](args)


if __name__ == "__main__":
    main()
