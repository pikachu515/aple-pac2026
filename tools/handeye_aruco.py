"""카메라–로봇 변환 자동 캘리브레이션 — 그리퍼에 붙인 ArUco 보드(또는 마커)를 탑뷰 카메라로 찍는다 (eye-to-hand, 역할 A).

touch_calib 은 사람이 끝점을 표식에 대야 해서 14 cm 그리퍼에서는 손목 각도·누르는 힘에 cm 단위로 흔들렸다
(10/9 실기 RMS 5~47 mm). 이 도구는 표적을 카메라가 자동으로 찾고, 같은 순간의 관절각으로 플랜지 자세를
구해 짝지어 푼다. 끝점에 댈 필요도, TCP 오프셋을 알 필요도 없다 (표적 ↔ 플랜지 변환도 같이 풀린다).

표적 (기본: 보드)
- **ArUco GridBoard 2×2** (DICT_4X4_50, id 10~13, 마커 35 mm, 간격 8 mm → 약 78 mm 정사각형). 모서리 16개를
  한 번에 쓰므로 마커 하나(모서리 4개)보다 자세 추정이 안정적이고, 하나가 가려지거나 반사돼도 나머지로 잡힌다
  (자세마다 마커 2개 이상 필요).
       python tools/handeye_aruco.py board-pdf --out ~/Desktop/aruco_board_A4.pdf     # A4 실제 크기 PDF
  '실제 크기(100%)' 로 인쇄하고 마커 한 변을 자로 재서 --board-marker-mm (·--board-sep-mm) 로 준다.
- 마커 하나만 쓸 때: --marker-id 0 --marker-mm 60 (data/calib/aruco_markers_A4.pdf). --board 와 같이 쓸 수 없다.
- 표적을 그리퍼/손목 윗면에 **움직이지 않게** 붙인다 (평평한 판에 붙인 뒤 테이프). 흰 여백을 자르지 않는다.

절차 (Piper Studio 켠 채, 팔로워는 리더암 텔레오퍼레이션으로 움직인다)
  1. 자세마다 Enter — 스냅샷 + 관절각을 함께 기록한다 (10~15 자세)
       python tools/handeye_aruco.py capture --session data/calib/handeye.json --iface can_follower \\
           --board 2x2 --board-marker-mm 35 --board-sep-mm 8 --board-first-id 10
     키: Enter = 기록   u = 마지막 자세 취소   q = 끝 (기록할 때마다 저장된다)
     요령: 작업 영역 전체로 위치를 옮기고, 손목을 ±30° 여러 방향으로 기울이고 돌린다 (회전 폭 30° 이상).
           표적은 늘 탑뷰 카메라에 보여야 한다. 팔이 멈춘 뒤 Enter. 표적 설정은 세션에 남아 fit·check 가 그대로 쓴다.
  2. 맞추고 점검한 뒤 저장한다
       python tools/handeye_aruco.py fit --session data/calib/handeye.json                       # 결과만
       python tools/handeye_aruco.py fit --session data/calib/handeye.json --save configs/workcell.yaml
  3. (선택) 저장된 T_base_camera 로 지금 자세를 검증한다
       python tools/handeye_aruco.py check --iface can_follower

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


# ================================================================ 표적 (보드 · 마커)

def marker_object_points(size_m: float) -> np.ndarray:
    """마커 하나의 네 모서리 (SOLVEPNP_IPPE_SQUARE 순서: 좌상·우상·우하·좌하, z = 0, 중심 원점)."""
    h = size_m / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float64)


class Target:
    """보드(GridBoard) 또는 마커 하나. 세션에 dict 로 남겨 fit·check 가 같은 표적을 다시 만든다.

    보드 좌표계는 OpenCV GridBoard 그대로: 원점 = 좌상 모서리, x 오른쪽, y 아래.
    """

    def __init__(self, kind: str, *, cols: int = 2, rows: int = 2, marker_m: float = 0.035, sep_m: float = 0.008,
                 first_id: int = 10, marker_id: int = 0, square_m: float = 0.028) -> None:
        if kind not in ("board", "marker", "charuco"):
            raise ValueError("kind 는 board, marker, charuco")
        self.kind, self.cols, self.rows = kind, int(cols), int(rows)
        self.marker_m, self.sep_m = float(marker_m), float(sep_m)
        self.first_id, self.marker_id = int(first_id), int(marker_id)
        self.square_m = float(square_m)
        self.min_markers = 1 if kind == "marker" else 2

    @property
    def ids(self) -> list[int]:
        if self.kind == "marker":
            return [self.marker_id]
        if self.kind == "charuco":
            return [int(i) for i in self.board().getIds().ravel()]
        return list(range(self.first_id, self.first_id + self.cols * self.rows))

    def board(self):
        if self.kind == "charuco":                     # 다른 사람이 만든 ChArUco 를 잘라 써도 된다 — 찾은 마커만 쓴다
            return cv2.aruco.CharucoBoard((self.cols, self.rows), self.square_m, self.marker_m,
                                          cv2.aruco.getPredefinedDictionary(DICT))
        return cv2.aruco.GridBoard((self.cols, self.rows), self.marker_m, self.sep_m,
                                   cv2.aruco.getPredefinedDictionary(DICT), np.array(self.ids))

    @property
    def size_m(self) -> tuple[float, float]:
        """표적 전체 가로·세로 (m)."""
        if self.kind == "marker":
            return self.marker_m, self.marker_m
        if self.kind == "charuco":
            return self.cols * self.square_m, self.rows * self.square_m
        return (self.cols * self.marker_m + (self.cols - 1) * self.sep_m,
                self.rows * self.marker_m + (self.rows - 1) * self.sep_m)

    def to_dict(self) -> dict:
        d = {"kind": self.kind, "dict": "DICT_4X4_50", "marker_m": self.marker_m}
        if self.kind == "board":
            d.update(cols=self.cols, rows=self.rows, sep_m=self.sep_m, first_id=self.first_id)
        elif self.kind == "charuco":
            d.update(cols=self.cols, rows=self.rows, square_m=self.square_m)
        else:
            d.update(marker_id=self.marker_id)
        return d

    @staticmethod
    def from_dict(d: dict) -> "Target":
        if d["kind"] == "board":
            return Target("board", cols=d["cols"], rows=d["rows"], marker_m=d["marker_m"], sep_m=d["sep_m"],
                          first_id=d["first_id"])
        if d["kind"] == "charuco":
            return Target("charuco", cols=d["cols"], rows=d["rows"], marker_m=d["marker_m"], square_m=d["square_m"])
        return Target("marker", marker_m=d["marker_m"], marker_id=d["marker_id"])

    @staticmethod
    def from_args(args) -> "Target":
        if getattr(args, "charuco", None):
            try:
                cols, rows = (int(v) for v in args.charuco.lower().split("x"))
            except ValueError:
                raise SystemExit(f"--charuco 는 '칸 열x행' (예: 7x5): {args.charuco}") from None
            return Target("charuco", cols=cols, rows=rows, marker_m=args.charuco_marker_mm / 1000,
                          square_m=args.square_mm / 1000)
        if getattr(args, "marker_id", None) is not None:
            return Target("marker", marker_m=args.marker_mm / 1000, marker_id=args.marker_id)
        board = args.board or "2x2"
        try:
            cols, rows = (int(v) for v in board.lower().split("x"))
        except ValueError:
            raise SystemExit(f"--board 는 '열x행' (예: 2x2): {board}") from None
        return Target("board", cols=cols, rows=rows, marker_m=args.board_marker_mm / 1000,
                      sep_m=args.board_sep_mm / 1000, first_id=args.board_first_id)

    def describe(self) -> str:
        if self.kind == "charuco":
            return (f"ChArUco {self.cols}×{self.rows}칸 id {self.ids[0]}~{self.ids[-1]}, 칸 {self.square_m * 1000:g} mm, "
                    f"마커 {self.marker_m * 1000:g} mm")
        if self.kind == "board":
            w, h = self.size_m
            return (f"보드 {self.cols}×{self.rows} id {self.ids[0]}~{self.ids[-1]}, 마커 {self.marker_m * 1000:g} mm, "
                    f"간격 {self.sep_m * 1000:g} mm ({w * 1000:.0f}×{h * 1000:.0f} mm)")
        return f"마커 id {self.marker_id}, {self.marker_m * 1000:g} mm"


def _detector():
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(DICT), params)


def detect_target(bgr: np.ndarray, target: Target) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
    """→ (표적 좌표계 모서리 N×3, 픽셀 모서리 N×2, 찾은 id) — 마커가 target.min_markers 보다 적으면 None.

    모서리는 마커마다 4개씩, 찾은 id 순서대로 이어진다.
    """
    gray = bgr if bgr.ndim == 2 else cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _detector().detectMarkers(gray)
    if ids is None:
        return None
    flat = [int(i) for i in ids.ravel()]
    if any(flat.count(i) > 1 for i in flat if i in target.ids):
        return None                                    # 같은 표적이 두 장 보인다 — 어느 것인지 모른다
    keep = [k for k, i in enumerate(flat) if i in target.ids]
    if len(keep) < target.min_markers:
        return None
    corners = [corners[k] for k in keep]
    found = [int(ids.ravel()[k]) for k in keep]
    if target.kind == "marker":
        return marker_object_points(target.marker_m), corners[0].reshape(4, 2).astype(np.float64), found
    if target.kind == "charuco":                       # CharucoBoard.matchImagePoints 는 체커 모서리용 — 마커 모서리는 직접
        b = target.board()
        by_id = {int(i): np.asarray(o, dtype=np.float64).reshape(4, 3) for i, o in zip(b.getIds().ravel(), b.getObjPoints())}
        obj = np.vstack([by_id[i] for i in found])
        img = np.vstack([c.reshape(4, 2) for c in corners]).astype(np.float64)
        return obj, img, found
    obj, img = target.board().matchImagePoints(corners, np.array(found).reshape(-1, 1))
    return obj.reshape(-1, 3).astype(np.float64), img.reshape(-1, 2).astype(np.float64), found


def detect_marker(bgr: np.ndarray, marker_id: int) -> np.ndarray | None:
    """마커 하나 → 4×2 픽셀 모서리 (좌상부터 시계방향) 또는 None."""
    found = detect_target(bgr, Target("marker", marker_id=marker_id))
    return None if found is None else found[1]


def _K(intrinsics) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


def target_pose(obj: np.ndarray, img: np.ndarray, intrinsics, depth_m: np.ndarray | None = None
                ) -> tuple[np.ndarray, float, float | None]:
    """→ (T_camera_target, 재투영 오차 px, 깊이 축척 또는 None).

    모서리 4개(마커 하나)는 IPPE_SQUARE, 그보다 많으면(보드) IPPE 로 풀고 모든 모서리로 LM 다듬기.
    작은 표적(0.9 m 에서 40~80 px)의 PnP 는 시선 방향 거리가 가장 흔들린다. 정렬 깊이가 표적 안에 충분하면
    보이는 모서리 중심까지 거리를 깊이 중앙값에 맞춘다 (방향은 PnP 그대로).
    """
    obj, img = np.asarray(obj, dtype=np.float64), np.asarray(img, dtype=np.float64)
    K = _K(intrinsics)
    if len(obj) == 4:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE)
        if ok:
            rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, None, rvec, tvec)
    if not ok:
        raise ValueError("solvePnP 실패")
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, None)
    reproj = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]
    T[:3, 3] = tvec.ravel()
    scale = None
    if depth_m is not None:
        mask = np.zeros(depth_m.shape, np.uint8)
        center = img.mean(axis=0)
        hull = cv2.convexHull(img.astype(np.float32)).reshape(-1, 2)
        cv2.fillConvexPoly(mask, (center + 0.6 * (hull - center)).astype(np.int32), 1)   # 가장자리 혼합 픽셀 제외
        d = depth_m[(mask > 0) & (depth_m > 0)]
        c_obj = obj.mean(axis=0)
        c_cam = T[:3, :3] @ c_obj + T[:3, 3]
        if len(d) >= 20:
            scale = float(np.median(d) / c_cam[2])
            if 0.9 < scale < 1.1:                                          # 크게 다르면 깊이 쪽이 틀렸다
                T[:3, 3] = scale * c_cam - T[:3, :3] @ c_obj
            else:
                scale = None
    return T, reproj, scale


def marker_pose(corners: np.ndarray, size_m: float, intrinsics, depth_m: np.ndarray | None = None
                ) -> tuple[np.ndarray, float, float | None]:
    """마커 하나 → (T_camera_marker, 재투영 오차 px, 깊이 축척 또는 None). target_pose 참고."""
    return target_pose(marker_object_points(size_m), corners, intrinsics, depth_m)


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


def _refine(F, T_bc, T_ft, obj_pts, img_pts, intrinsics, iters=50):
    """T_base_camera·T_flange_target 를 함께 모든 모서리 재투영 오차(px)로 다듬는다 (Levenberg–Marquardt).

    재투영 오차는 PnP 의 시선 방향(거리) 흔들림에 영향을 받지 않는다. 카메라 좌표 모서리(3D)로 다듬으면
    그 흔들림이 그대로 들어가 오히려 나빠졌다 (합성 시험: 1.6 → 6.5 mm).
    """
    K = _K(intrinsics)
    objs = [np.c_[np.asarray(o, dtype=np.float64), np.ones(len(o))] for o in obj_pts]
    obs = [np.asarray(c, dtype=np.float64) for c in img_pts]

    def resid(x):
        T_cb, T_ft_ = np.linalg.inv(_T_from(x[:6])), _T_from(x[6:])
        out = []
        for Fi, oh, o in zip(F, objs, obs):
            uv = (K @ (T_cb @ Fi @ T_ft_ @ oh.T)[:3]).T
            out.append((uv[:, :2] / uv[:, 2:3] - o).ravel())
        return np.concatenate(out)

    x = np.r_[_x_from(T_bc), _x_from(T_ft)]
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


def solve(flanges: list[np.ndarray], cam_targets: list[np.ndarray], obj_pts: list | None = None,
          img_pts: list | None = None, intrinsics=None) -> tuple[np.ndarray, np.ndarray]:
    """→ (T_base_camera, T_flange_target).

    OpenCV calibrateHandEye(Park) 로 풀고 — 합성 잡음 시험에서 Tsai·Daniilidis 보다 안정적이었다 —
    자세별 모서리(표적 좌표·픽셀)와 내부 파라미터가 있으면 두 변환을 함께 재투영 오차로 다듬는다.
    """
    F = [validate_T(T, "T_base_flange") for T in flanges]
    C = [validate_T(T, "T_camera_target") for T in cam_targets]
    T_bc = solve_eye_to_hand(F, C, method=cv2.CALIB_HAND_EYE_PARK)
    T_ft = _mean_transform([np.linalg.inv(Fi) @ T_bc @ Ci for Fi, Ci in zip(F, C)])
    if obj_pts is not None and img_pts is not None and intrinsics is not None:
        T_bc, T_ft = _refine(F, T_bc, T_ft, obj_pts, img_pts, intrinsics)
    return T_bc, T_ft


def residuals(flanges, cam_targets, T_bc: np.ndarray, T_ft: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """자세별 (표적 원점 위치 오차 m, 방향 오차 deg) — 카메라로 본 것 vs FK 로 예측한 것."""
    pos, rot = [], []
    for Fi, Ci in zip(flanges, cam_targets):
        seen, pred = T_bc @ np.asarray(Ci), np.asarray(Fi) @ T_ft
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
                    "— 표적이 그리퍼에서 흔들리는지, 팔이 멈춘 뒤 찍었는지 확인")
    if len(res_m) >= 4:
        med = float(np.median(res_m))
        for i in np.nonzero(res_m > max(3 * med, 0.006))[0]:
            warn.append(f"{i + 1}번째 자세 잔차 {res_m[i] * 1000:.1f} mm 가 유난히 크다 — --drop {i + 1} 로 빼고 다시")
    if T_bc[2, 2] > -0.5:
        warn.append("카메라 광축이 아래를 보지 않는다 — 표적 크기(--board-marker-mm·--marker-mm)나 관절각 단위를 확인")
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
    """세션 dict. 표적 정의는 session["target"] (Target.to_dict), 자세마다 obj_pts·img_pts 를 남긴다."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"target": None, "poses": []}


def save_session(path: Path, session: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(session, ensure_ascii=False, indent=1), encoding="utf-8")


def _arrays(session: dict, drop=()) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    keep = [i for i in range(len(session["poses"])) if (i + 1) not in set(drop)]
    F = [np.asarray(session["poses"][i]["T_base_flange"], dtype=float) for i in keep]
    C = [np.asarray(session["poses"][i]["T_camera_target"], dtype=float) for i in keep]
    return F, C, keep


def _points(session: dict, keep: list[int]) -> tuple[list | None, list | None]:
    if not session.get("intrinsics"):
        return None, None
    return [session["poses"][i]["obj_pts"] for i in keep], [session["poses"][i]["img_pts"] for i in keep]


# ================================================================ 명령

def record_pose(snap, joints, fk, target: Target, use_depth: bool = True) -> dict | None:
    """스냅샷 + 관절각 → 세션 자세 한 개 (표적을 못 찾으면 None)."""
    found = detect_target(snap.color, target)
    if found is None:
        return None
    obj, img, ids = found
    T_ct, reproj, scale = target_pose(obj, img, snap.intrinsics, snap.depth_m if use_depth else None)
    T_bf = np.asarray(fk(joints), dtype=float)
    return {"joints_rad": [round(float(q), 6) for q in joints], "T_base_flange": T_bf.round(6).tolist(),
            "T_camera_target": T_ct.round(6).tolist(), "n_markers": len(ids), "marker_ids": sorted(ids),
            "obj_pts": obj.round(6).tolist(), "img_pts": img.round(3).tolist(),
            "reproj_px": round(reproj, 3), "depth_scale": None if scale is None else round(scale, 4),
            "stamp_s": snap.stamp_s}


def cmd_capture(args) -> None:
    session = load_session(args.session)
    target = Target.from_args(args)
    if session["poses"] and session.get("target") != target.to_dict():
        raise SystemExit(f"세션의 표적이 다르다 ({session.get('target')}) — 다른 세션 파일을 쓴다")
    session["target"] = target.to_dict()
    print(f"표적: {target.describe()}")
    fk, joint_state = flange_reader(args.iface)
    img_dir = args.session.parent / "handeye_imgs" if args.save_images else None
    if args.auto:
        return _capture_auto(args, session, target, fk, joint_state, img_dir)
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
        pose = record_pose(snap, joint_state(), fk, target, not args.no_depth)
        if pose is None:
            print(f"  [다시] 표적을 못 찾았다 (마커 {target.min_markers}개 이상 필요) — 카메라에 표적이 하나만 보이는지(같은 보드 두 장이면 안 된다), "
                  "반사·흐림이 없는지 확인")
            continue
        session["intrinsics"] = [float(v) for v in snap.intrinsics]
        session["poses"].append(pose)
        save_session(args.session, session)
        if img_dir is not None:
            img_dir.mkdir(parents=True, exist_ok=True)
            img = snap.color.copy()
            for quad in np.asarray(pose["img_pts"]).reshape(-1, 4, 2):
                cv2.polylines(img, [quad.astype(np.int32)], True, (0, 0, 255), 2)
            cv2.imwrite(str(img_dir / f"pose_{len(session['poses']):02d}.png"), img)
        F = [np.asarray(p["T_base_flange"]) for p in session["poses"]]
        z = pose["T_camera_target"][2][3]
        print(f"  기록 — 마커 {pose['n_markers']}개 {pose['marker_ids']}, 거리 {z * 1000:.0f} mm, "
              f"재투영 {pose['reproj_px']:.2f} px"
              + ("" if pose["depth_scale"] is None else f", 깊이 축척 {pose['depth_scale']:.3f}")
              + f" | 자세 {len(F)}개, 최대 회전 {rotation_spread_deg(F):.0f}°")
    print(f"완료 — {len(session['poses'])}개. 다음: fit")


def is_new_pose(T: np.ndarray, recorded: list[np.ndarray], min_move_m: float, min_rot_deg: float) -> bool:
    """이미 기록한 자세들과 충분히 다른가 (위치 min_move_m 또는 회전 min_rot_deg 이상)."""
    return all(np.linalg.norm(T[:3, 3] - R[:3, 3]) >= min_move_m
               or rotation_angle_deg(T[:3, :3], R[:3, :3]) >= min_rot_deg for R in recorded)


def _capture_auto(args, session, target, fk, joint_state, img_dir) -> None:
    """Enter 없이 — 팔이 still_s 동안 멈추고 이전 자세들과 다르면 기록한다 (텔레오퍼레이션으로 옮기기만 하면 된다)."""
    print(f"자동 기록: 팔을 {args.still_s:.1f}초 멈추면 기록 (이전 자세와 {args.min_move_mm:.0f} mm 또는 "
          f"{args.min_rot_deg:.0f}° 이상 달라야 함). {args.max_poses}개 또는 Ctrl+C 로 끝. 지금 {len(session['poses'])}개",
          flush=True)
    recorded = [np.asarray(p["T_base_flange"]) for p in session["poses"]]
    window: list[tuple[float, np.ndarray]] = []
    last_miss = 0.0
    try:
        while len(session["poses"]) < args.max_poses:
            now = time.time()
            q = np.asarray(joint_state(), dtype=float)
            window = [(t, v) for t, v in window if now - t <= args.still_s] + [(now, q)]
            still = (now - window[0][0] >= args.still_s * 0.9
                     and np.degrees(np.ptp([v for _, v in window], axis=0)).max() < args.still_deg)
            T = np.asarray(fk(q), dtype=float)
            if still and is_new_pose(T, recorded, args.min_move_mm / 1000, args.min_rot_deg):
                snap = grab_snapshot(args)
                pose = record_pose(snap, q, fk, target, not args.no_depth)
                if pose is None:
                    if now - last_miss > 3:
                        print("  [못 찾음] 표적이 안 보이거나 같은 보드가 두 장 보인다 — 자세를 바꾸거나 다른 보드를 치운다", flush=True)
                        last_miss = now
                else:
                    session["intrinsics"] = [float(v) for v in snap.intrinsics]
                    session["poses"].append(pose)
                    save_session(args.session, session)
                    recorded.append(T)
                    if img_dir is not None:
                        img_dir.mkdir(parents=True, exist_ok=True)
                        img = snap.color.copy()
                        for quad in np.asarray(pose["img_pts"]).reshape(-1, 4, 2):
                            cv2.polylines(img, [quad.astype(np.int32)], True, (0, 0, 255), 2)
                        cv2.imwrite(str(img_dir / f"pose_{len(session['poses']):02d}.png"), img)
                    print(f"  [{len(session['poses'])}] 기록 — 마커 {pose['n_markers']}개, "
                          f"거리 {pose['T_camera_target'][2][3] * 1000:.0f} mm, 재투영 {pose['reproj_px']:.2f} px, "
                          f"최대 회전 {rotation_spread_deg(recorded):.0f}°", flush=True)
            time.sleep(0.15)
    except KeyboardInterrupt:
        pass
    print(f"완료 — {len(session['poses'])}개. 다음: fit", flush=True)


def cmd_fit(args) -> tuple[np.ndarray, np.ndarray]:
    session = load_session(args.session)
    F, C, keep = _arrays(session, args.drop)
    if len(F) < 3:
        raise SystemExit(f"자세가 {len(F)}개다 — capture 로 8~15개 찍는다")
    if session.get("target"):
        print(f"표적: {Target.from_dict(session['target']).describe()}")
    obj_pts, img_pts = _points(session, keep)
    T_bc, T_ft = solve(F, C, obj_pts, img_pts, session.get("intrinsics"))
    res, rot = residuals(F, C, T_bc, T_ft)
    spread = rotation_spread_deg(F)
    print(f"{'자세':<5}{'잔차 mm':>9}{'방향 °':>8}{'재투영 px':>11}{'마커':>6}")
    for i, r, a in zip(keep, res, rot):
        pose = session["poses"][i]
        print(f"{i + 1:<5}{r * 1000:>9.1f}{a:>8.2f}{pose['reproj_px']:>11.2f}{pose.get('n_markers', 1):>6}")
    print(f"\n자세 {len(F)}개  잔차 중앙값 {np.median(res) * 1000:.1f} mm / 최대 {res.max() * 1000:.1f} mm, 최대 회전 {spread:.0f}°")
    print(f"카메라 위치 (base_link) ({', '.join(f'{v * 1000:.1f}' for v in T_bc[:3, 3])}) mm, "
          f"광축 방향 ({', '.join(f'{v:.3f}' for v in T_bc[:3, 2])})")
    print(f"표적 원점 ← 플랜지 ({', '.join(f'{v * 1000:.1f}' for v in T_ft[:3, 3])}) mm (표적을 붙인 위치와 맞는지 본다)")
    tcp = load_config()["workcell"].get("tcp_offset_m")
    if tcp is None:
        print("참고: workcell.tcp_offset_m 이 비어 있다 — 실행 전에 핑거 끝 오프셋을 따로 넣는다 (이 보정과는 무관)")
    warn = fit_warnings(len(F), spread, res, T_bc)
    for w in warn:
        print(f"[경고] {w}")
    session["fit"] = {"T_base_camera": T_bc.round(6).tolist(), "T_flange_target": T_ft.round(6).tolist(),
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
    return T_bc, T_ft


def cmd_check(args) -> None:
    session = load_session(args.session)
    if not session.get("fit") or not session.get("target"):
        raise SystemExit("세션에 표적·fit 결과(T_flange_target)가 없다 — capture·fit 을 먼저 한다")
    T_ft = np.asarray(session["fit"]["T_flange_target"], dtype=float)
    target = Target.from_dict(session["target"])
    T_bc = (np.asarray(session["fit"]["T_base_camera"], dtype=float) if args.from_session
            else load_T_base_camera(str(args.workcell)))
    fk, joint_state = flange_reader(args.iface)
    snap = grab_snapshot(args)
    pose = record_pose(snap, joint_state(), fk, target, not args.no_depth)
    if pose is None:
        raise SystemExit("표적을 못 찾았다")
    res, rot = residuals([np.asarray(pose["T_base_flange"])], [np.asarray(pose["T_camera_target"])], T_bc, T_ft)
    print(f"카메라로 본 표적 vs 순기구학 예측: 위치 {res[0] * 1000:.1f} mm, 방향 {rot[0]:.2f}° (마커 {pose['n_markers']}개)")


def board_page(target: Target, dpi: int = 300):
    """A4 실제 크기 페이지 (PIL 회색조) — 표적 가운데, 100 mm 자, 영어 설명 (한글 글꼴이 없는 PC 대비)."""
    from PIL import Image, ImageDraw, ImageFont

    def mm(v: float) -> int:
        return int(round(v / 25.4 * dpi))

    page = Image.new("L", (mm(210), mm(297)), 255)
    draw = ImageDraw.Draw(page)
    font = None
    for f in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf"):
        try:
            font = ImageFont.truetype(f, 38)
            break
        except OSError:
            pass
    font = font or ImageFont.load_default()
    w_m, h_m = target.size_m
    w_px, h_px = mm(w_m * 1000), mm(h_m * 1000)
    if target.kind == "board":
        img = target.board().generateImage((w_px, h_px), marginSize=0)
    else:
        img = cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(DICT), target.marker_id, w_px)
    x0, y0 = (page.width - w_px) // 2, (mm(210) - h_px) // 2 + mm(20)
    page.paste(Image.fromarray(img), (x0, y0))
    y = y0 + h_px + mm(15)
    if target.kind == "board":
        lines = [f"ArUco GridBoard {target.cols}x{target.rows}   DICT_4X4_50   ids {target.ids[0]}-{target.ids[-1]}",
                 f"marker side {target.marker_m * 1000:g} mm   separation {target.sep_m * 1000:g} mm   "
                 f"board {w_m * 1000:.0f} x {h_m * 1000:.0f} mm"]
    else:
        lines = [f"ArUco marker   DICT_4X4_50   id {target.marker_id}", f"marker side {target.marker_m * 1000:g} mm"]
    lines += ["PRINT AT 100% (actual size, no fit-to-page). Measure the marker side with a ruler.",
              "Keep the white border (>= 5 mm). Mount flat and rigid on top of the gripper/wrist."]
    for k, line in enumerate(lines):
        draw.text((mm(20), y + k * mm(7)), line, fill=0, font=font)
    ys = mm(255)                                                       # 100 mm 축척 막대
    draw.line((mm(55), ys, mm(155), ys), fill=0, width=4)
    for k in range(0, 101, 10):
        draw.line((mm(55 + k), ys - mm(6 if k % 50 == 0 else 3), mm(55 + k), ys), fill=0, width=3)
    draw.text((mm(55), ys + mm(3)), "100 mm - this bar must measure exactly 100 mm", fill=0, font=font)
    draw.text((mm(20), mm(20)), "PAC2026 hand-eye calibration target (tools/handeye_aruco.py)", fill=0, font=font)
    return page


def cmd_board_pdf(args) -> Path:
    target = Target.from_args(args)
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    board_page(target).save(out, resolution=300)
    print(f"저장: {out} — {target.describe()}. '실제 크기(100%)' 로 인쇄")
    return out


def main(argv: list[str] | None = None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def target_args(p):
        g = p.add_mutually_exclusive_group()
        g.add_argument("--board", help="GridBoard 열x행 (기본 2x2)")    # 기본값을 두면 --marker-id 와 충돌 검사가 빠진다
        g.add_argument("--marker-id", type=int, help="보드 대신 마커 하나 (id)")
        g.add_argument("--charuco", help="ChArUco 칸 수 열x행 (예: 7x5) — 잘라 쓴 조각도 된다")
        p.add_argument("--square-mm", type=float, default=28.0, help="--charuco 칸 한 변 (mm)")
        p.add_argument("--charuco-marker-mm", type=float, default=21.0, help="--charuco 마커 한 변 (mm)")
        p.add_argument("--board-marker-mm", type=float, default=35.0, help="보드 마커 한 변 (mm, 인쇄물을 자로 잰 값)")
        p.add_argument("--board-sep-mm", type=float, default=8.0, help="보드 마커 간격 (mm)")
        p.add_argument("--board-first-id", type=int, default=10, help="보드 첫 마커 id (행 우선으로 이어진다)")
        p.add_argument("--marker-mm", type=float, default=60.0, help="--marker-id 일 때 마커 한 변 (mm)")

    def device_args(p):
        p.add_argument("--iface", help="관절각을 읽을 팔로워 (예: can_follower)")
        p.add_argument("--no-depth", action="store_true", help="정렬 깊이로 표적 거리를 보정하지 않는다")
        p.add_argument("--calib", type=Path, default=ROOT / "configs" / "calib" / "d435.json")
        p.add_argument("--near", type=float, default=150.0, help="Piper Studio 깊이 인코딩 near_mm")
        p.add_argument("--far", type=float, default=1200.0, help="Piper Studio 깊이 인코딩 far_mm")
        p.add_argument("--serial")
        p.add_argument("--shm-dir", type=Path, default=SHM_DIR)

    c = sub.add_parser("capture", help="자세마다 스냅샷 + 관절각 기록")
    c.add_argument("--session", type=Path, default=ROOT / "data" / "calib" / "handeye.json")
    c.add_argument("--save-images", action="store_true", help="자세마다 검출 표시 영상을 handeye_imgs/ 에 저장")
    c.add_argument("--auto", action="store_true", help="Enter 없이 팔이 멈추면 자동 기록")
    c.add_argument("--still-s", type=float, default=1.0, help="--auto: 이만큼 멈추면 기록 (s)")
    c.add_argument("--still-deg", type=float, default=0.3, help="--auto: 멈춤 판정 관절 흔들림 상한 (deg)")
    c.add_argument("--min-move-mm", type=float, default=40.0, help="--auto: 이전 자세와 최소 위치 차")
    c.add_argument("--min-rot-deg", type=float, default=10.0, help="--auto: 이전 자세와 최소 회전 차")
    c.add_argument("--max-poses", type=int, default=15, help="--auto: 이만큼 모이면 끝")
    target_args(c)
    device_args(c)

    f = sub.add_parser("fit", help="T_base_camera 를 풀고 점검, --save 로 workcell.yaml 에 저장")
    f.add_argument("--session", type=Path, default=ROOT / "data" / "calib" / "handeye.json")
    f.add_argument("--drop", type=int, nargs="*", default=[], help="뺄 자세 번호 (1부터)")
    f.add_argument("--save", type=Path, help="T_base_camera 를 쓸 workcell.yaml")
    f.add_argument("--force", action="store_true", help="경고가 있어도 저장")

    k = sub.add_parser("check", help="저장된 T_base_camera 로 지금 자세 검증 (표적은 세션에 남은 것)")
    k.add_argument("--session", type=Path, default=ROOT / "data" / "calib" / "handeye.json")
    k.add_argument("--workcell", type=Path, default=ROOT / "configs" / "workcell.yaml")
    k.add_argument("--from-session", action="store_true", help="workcell.yaml 대신 세션의 fit 결과를 쓴다")
    device_args(k)

    b = sub.add_parser("board-pdf", help="표적 A4 PDF (실제 크기, 300 dpi)")
    b.add_argument("--out", type=Path, default=ROOT / "data" / "calib" / "aruco_board_A4.pdf")
    target_args(b)

    args = ap.parse_args(argv)
    return {"capture": cmd_capture, "fit": cmd_fit, "check": cmd_check, "board-pdf": cmd_board_pdf}[args.cmd](args)


if __name__ == "__main__":
    main()
