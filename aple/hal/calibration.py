"""카메라–로봇 변환 (설계서 calibration_node). 역할 A.

체커보드 / ChArUco eye-to-hand 캘리브레이션 결과(4×4, base_link ← camera_top_optical_frame)를
configs/workcell.yaml 에 저장·로드한다. 스크립트 실행의 위치 정확도는 이 값에 직접 좌우된다.
TCP 오프셋(핑거 어댑터 반영)도 여기서 관리한다.

좌표계
- 카메라 광학 좌표계: x 오른쪽, y 아래, z 시선 방향 (m). 깊이 depth_m 은 이 z 값이다.
- base_link: 로봇 베이스 (m). 위치는 모두 이 좌표계로 내보낸다.
- T_base_camera: p_base = T[:3, :3] @ p_cam + T[:3, 3]

현장 산출 경로 두 가지
1. 점 터치 (기본) — 같은 표식을 카메라(클릭 + 깊이)와 로봇(끝점으로 터치)으로 6~10개 잰다.
   fit_touch_points 가 맞추고 점별 잔차·leave-one-out 오차로 점검한다. 절차는 tools/touch_calib.py
2. solve_eye_to_hand — ChArUco 보드를 그리퍼에 물리고 자세를 바꿔 가며 찍는다 (OpenCV calibrateHandEye).
어느 쪽이든 save_T_base_camera 로 workcell.yaml 에 써 넣는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import yaml

from aple.config import set_config_value

_ORTHO_TOL = 1e-3


# ================================================================ 로드 · 검증 · 저장

def validate_T(T, name: str = "T_base_camera") -> np.ndarray:
    """4×4 강체 변환인지 확인하고 float64 배열로 돌려준다."""
    T = np.asarray(T, dtype=np.float64)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError(f"{name} 는 유한한 4×4 행렬이어야 한다 (받은 모양 {T.shape})")
    if not np.allclose(T[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError(f"{name} 마지막 행은 [0, 0, 0, 1] 이어야 한다")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=_ORTHO_TOL) or np.linalg.det(R) < 0:
        raise ValueError(f"{name} 회전부가 회전 행렬이 아니다 (직교·det=+1 위반)")
    return T


def T_from_cfg(cfg: dict) -> np.ndarray | None:
    """load_config() 결과에서 T_base_camera. 아직 실측 전(null)이면 None."""
    value = cfg.get("workcell", {}).get("T_base_camera")
    return None if value is None else validate_T(value)


def load_T_base_camera(path: str) -> np.ndarray:
    """workcell.yaml(또는 같은 키를 가진 YAML)에서 T_base_camera 를 읽는다. 비어 있으면 ValueError."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    T = T_from_cfg(data)
    if T is None:
        raise ValueError(f"{path}: workcell.T_base_camera 가 비어 있다 — 카메라–로봇 캘리브레이션을 먼저 한다")
    return T


def save_T_base_camera(path: str, T: np.ndarray, note: str | None = None) -> str:
    """workcell.yaml 의 T_base_camera 줄만 바꾼다 (주석 유지). 바뀐 줄을 돌려준다."""
    T = validate_T(T)
    rows = [[round(float(v), 6) for v in row] for row in T]
    return set_config_value(path, "workcell.T_base_camera", rows, note=note)


def tcp_offset_from_cfg(cfg: dict) -> np.ndarray:
    """플랜지 → TCP 오프셋 (m, 플랜지 좌표계). 실측 전이면 0."""
    value = cfg.get("workcell", {}).get("tcp_offset_m")
    if value is None:
        return np.zeros(3)
    off = np.asarray(value, dtype=np.float64)
    if off.shape != (3,) or not np.all(np.isfinite(off)):
        raise ValueError("workcell.tcp_offset_m 은 [x, y, z] (m) 이어야 한다")
    return off


# ================================================================ 투영

def backproject(us, vs, depth_m, intrinsics) -> np.ndarray:
    """픽셀 + 깊이(z) → 카메라 좌표 (N×3, m). 입력은 스칼라 또는 같은 길이의 배열."""
    fx, fy, cx, cy = intrinsics
    us, vs, z = (np.asarray(a, dtype=np.float64) for a in (us, vs, depth_m))
    return np.stack([(us - cx) / fx * z, (vs - cy) / fy * z, z], axis=-1)


def camera_to_base(points_cam, T_base_camera: np.ndarray) -> np.ndarray:
    p = np.asarray(points_cam, dtype=np.float64)
    return p @ T_base_camera[:3, :3].T + T_base_camera[:3, 3]


def base_to_camera(points_base, T_base_camera: np.ndarray) -> np.ndarray:
    p = np.asarray(points_base, dtype=np.float64)
    R, t = T_base_camera[:3, :3], T_base_camera[:3, 3]
    return (p - t) @ R          # = R.T @ (p - t)


def pixel_to_base(u: float, v: float, depth_m: float, intrinsics: tuple, T_base_camera: np.ndarray) -> np.ndarray:
    """픽셀 + 깊이 → base_link 좌표 (m)."""
    if not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError(f"깊이가 유효하지 않다 ({depth_m}) — 결측 픽셀은 역투영할 수 없다")
    return camera_to_base(backproject(u, v, depth_m, intrinsics), T_base_camera)


def base_to_pixel(points_base, intrinsics: tuple, T_base_camera: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """base_link 점 → (픽셀 N×2 [u, v], 카메라 z N). z ≤ 0 이면 카메라 뒤쪽이다."""
    fx, fy, cx, cy = intrinsics
    pc = np.atleast_2d(base_to_camera(points_base, T_base_camera))
    z = pc[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = np.stack([fx * pc[:, 0] / z + cx, fy * pc[:, 1] / z + cy], axis=-1)
    return uv, z


# ================================================================ 산출

def fit_rigid_transform(points_cam, points_base) -> tuple[np.ndarray, float]:
    """대응점으로 T_base_camera (Kabsch/Umeyama, 축척 없음). → (T, RMS 잔차 m).

    points_cam: 카메라가 본 점 (깊이 역투영), points_base: 같은 점을 TCP 로 찍은 base_link 좌표.
    한 직선 위에 있지 않은 점 3개 이상, 실제로는 작업 영역 전체에 흩어진 6~10개를 쓴다.
    """
    P = np.asarray(points_cam, dtype=np.float64)
    Q = np.asarray(points_base, dtype=np.float64)
    if P.shape != Q.shape or P.ndim != 2 or P.shape[1] != 3 or len(P) < 3:
        raise ValueError("대응점은 같은 개수(3개 이상)의 N×3 배열이어야 한다")
    pc, qc = P.mean(axis=0), Q.mean(axis=0)
    H = (P - pc).T @ (Q - qc)
    if np.linalg.matrix_rank(H, tol=1e-9) < 2:
        raise ValueError("대응점이 한 직선 위에 있다 — 작업 영역에 넓게 흩어 잰다")
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, qc - R @ pc
    resid = np.linalg.norm(camera_to_base(P, T) - Q, axis=1)
    return T, float(np.sqrt(np.mean(resid ** 2)))


# ================================================================ 점 터치 간이 캘리브레이션 (tools/touch_calib.py)

def camera_point_at(depth_m: np.ndarray, u: float, v: float, intrinsics, win_px: int = 3,
                    min_valid: int = 5) -> np.ndarray:
    """클릭한 픽셀의 카메라 좌표 (m). 깊이는 (2·win+1)² 창 안 유효 값의 중앙값 (한 픽셀 결측·잡음 대비)."""
    H, W = depth_m.shape
    ui, vi = int(round(u)), int(round(v))
    if not (0 <= ui < W and 0 <= vi < H):
        raise ValueError(f"픽셀 ({u:.1f}, {v:.1f}) 이 영상 밖이다")
    win = depth_m[max(vi - win_px, 0):vi + win_px + 1, max(ui - win_px, 0):ui + win_px + 1]
    z = win[win > 0]
    if len(z) < min_valid:
        raise ValueError(f"픽셀 ({u:.0f}, {v:.0f}) 주변에 유효 깊이가 없다 — 광택 없는 표식을 쓰고 다시 찍는다")
    return backproject(u, v, float(np.median(z)), intrinsics)


def rpy_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """고정축 roll-pitch-yaw (rad): R = Rz(rz)·Ry(ry)·Rx(rx). AgileX 말단 자세(RX, RY, RZ) 표기 기준."""
    cx, sx, cy, sy, cz, sz = np.cos(rx), np.sin(rx), np.cos(ry), np.sin(ry), np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def tip_from_flange(T_base_flange: np.ndarray, tcp_offset_m) -> np.ndarray:
    """플랜지 자세 + 플랜지 좌표계의 끝점 오프셋 → 끝점 base_link 좌표 (m)."""
    T = validate_T(T_base_flange, "T_base_flange")
    return T[:3, :3] @ np.asarray(tcp_offset_m, dtype=np.float64) + T[:3, 3]


@dataclass
class TouchFit:
    T: np.ndarray                 # T_base_camera
    rms_m: float                  # 모든 점으로 맞춘 뒤의 RMS 잔차
    residuals_m: np.ndarray       # 점별 잔차 (맞춤에 쓴 점)
    loo_errors_m: np.ndarray      # 점별 leave-one-out 예측 오차 — 처음 보는 점에 대한 정확도의 정직한 추정
    warnings: list[str] = field(default_factory=list)


def fit_touch_points(points_cam, points_base, *, min_points: int = 6, max_rms_m: float = 0.004,
                     min_spread_m: float = 0.15) -> TouchFit:
    """점 터치 대응점으로 T_base_camera 를 맞추고 품질을 점검한다.

    경고 (저장 여부는 사람이 판단한다)
    - 점이 min_points 보다 적다 / 작업 영역에 넓게 퍼지지 않았다 (가장 먼 두 점 < min_spread_m)
    - RMS 가 max_rms_m 를 넘는다
    - 한 점의 leave-one-out 오차가 다른 점들보다 유난히 크다 → 클릭 실수·터치 실수 의심 (--drop 으로 빼고 다시)
    - 카메라 광축이 아래를 보지 않는다 → 좌표 단위(mm/m)나 점 순서가 어긋났을 가능성
    """
    P = np.asarray(points_cam, dtype=np.float64)
    Q = np.asarray(points_base, dtype=np.float64)
    T, rms = fit_rigid_transform(P, Q)
    resid = np.linalg.norm(camera_to_base(P, T) - Q, axis=1)
    n = len(P)
    loo = np.full(n, np.nan)
    if n >= 4:
        for i in range(n):
            keep = np.arange(n) != i
            try:
                Ti, _ = fit_rigid_transform(P[keep], Q[keep])
            except ValueError:
                continue
            loo[i] = np.linalg.norm(camera_to_base(P[i], Ti) - Q[i])

    warn = []
    if n < min_points:
        warn.append(f"점이 {n}개다 — {min_points}~10개를 작업 영역 전체에 흩어 찍는다")
    spread = max(np.linalg.norm(a - b) for a in Q for b in Q)
    if spread < min_spread_m:
        warn.append(f"점이 좁게 모여 있다 (가장 먼 두 점 {spread * 1000:.0f} mm) — 트레이·상자 양 끝까지 넓힌다")
    if np.ptp(Q[:, 2]) < 0.02:
        warn.append("모든 점이 같은 높이다 — 받침대 위 점을 2~3개 섞으면 기울기 추정이 좋아진다")
    if rms > max_rms_m:
        warn.append(f"RMS {rms * 1000:.1f} mm 가 기준 {max_rms_m * 1000:.1f} mm 를 넘는다")
    finite = loo[np.isfinite(loo)]
    if len(finite) >= 4:
        med = float(np.median(finite))
        for i in np.nonzero(np.isfinite(loo) & (loo > max(3 * med, 0.005)))[0]:
            warn.append(f"{i + 1}번째 점의 leave-one-out 오차 {loo[i] * 1000:.1f} mm 가 유난히 크다 — 클릭·터치를 다시 확인")
    if T[2, 2] > -0.5:
        warn.append("카메라 광축이 아래를 보지 않는다 — 로봇 좌표 단위(mm↔m)나 점 순서가 맞는지 확인")
    return TouchFit(T, float(rms), resid, loo, warn)


def solve_eye_to_hand(T_base_gripper: list[np.ndarray], T_camera_target: list[np.ndarray],
                      method: int = cv2.CALIB_HAND_EYE_TSAI) -> np.ndarray:
    """고정 카메라(eye-to-hand) 캘리브레이션 → T_base_camera.

    T_base_gripper[i]: 촬영 i 시점의 플랜지 자세 (로봇 FK), T_camera_target[i]: 보드 자세 (solvePnP).
    보드는 그리퍼에 고정한다. OpenCV 는 eye-in-hand 형식이므로 로봇 자세를 역으로 넣으면
    결과가 base ← camera 가 된다. 자세는 회전축을 다양하게 10개 이상 쓴다.
    """
    if len(T_base_gripper) != len(T_camera_target) or len(T_base_gripper) < 3:
        raise ValueError("로봇 자세와 보드 자세를 같은 개수(3개 이상) 넣는다")
    R_g2b, t_g2b, R_t2c, t_t2c = [], [], [], []
    for Tbg, Tct in zip(T_base_gripper, T_camera_target):
        Tgb = np.linalg.inv(validate_T(Tbg, "T_base_gripper"))
        Tct = validate_T(Tct, "T_camera_target")
        R_g2b.append(Tgb[:3, :3]); t_g2b.append(Tgb[:3, 3])
        R_t2c.append(Tct[:3, :3]); t_t2c.append(Tct[:3, 3])
    R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_t2c, t_t2c, method=method)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t).ravel()
    return validate_T(T)
