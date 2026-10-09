"""ChArUco 보드 검출·자세 (카메라–로봇 캘리브레이션 보조). 역할 A.

쓰임
- 보드 자세 확인: 카메라에서 본 보드의 거리·기울기·재투영 오차 → 카메라 내부 파라미터·인쇄 배율 점검
- 깊이 오차: PnP 로 구한 보드 평면과 D435 깊이를 비교 → 깊이 편향 (8비트 인코딩·정렬 포함 경로 점검)
- eye-to-hand: 보드를 그리퍼에 물리고 자세를 바꿔 가며 찍은 T_camera_board 를
  aple/hal/calibration.py solve_eye_to_hand 에 넣는다 (tools/charuco_calib.py handeye)

보드 크기 = 설계 × print_scale (configs/charuco.yaml). 배율을 틀리게 넣으면 보드 거리가 같은 비율로
틀린다 (각도는 그대로). 보드 좌표계는 OpenCV 보드 좌표계(한 모서리 원점, z = 보드 법선)를 그대로 쓴다.
eye-to-hand 와 깊이 비교는 좌표계 원점 선택과 무관하다.

OpenCV 4.7 이후(CharucoDetector)와 이전(interpolateCornersCharuco) 둘 다 지원한다. 4.6 이전은 옛 칸 배치만
있어서 행 수가 짝수인 보드는 최신 배치(legacy_pattern: false)를 읽을 수 없다 — 이 보드(5행)는 문제없다.
왜곡 계수는 0 으로 둔다 (스냅샷 내부 파라미터에 왜곡이 없다. D435 컬러 왜곡은 매우 작다).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from aple.hal.calibration import backproject

_NEW_API = hasattr(cv2.aruco, "CharucoDetector")


@dataclass(frozen=True)
class BoardSpec:
    dictionary: str
    squares_x: int
    squares_y: int
    square_m: float              # 실제 칸 한 변 (= 설계 × 배율)
    marker_m: float              # 실제 마커 한 변
    legacy_pattern: bool = False
    min_corners: int = 6
    max_reproj_px: float = 1.0


def board_spec(cfg: dict) -> BoardSpec:
    c = cfg["charuco"]
    scale = float(c.get("print_scale", 1.0))
    if not 0.3 < scale < 3.0:
        raise ValueError(f"charuco.print_scale {scale} 이 이상하다 — 100 mm 눈금 실측값 / 100 을 넣는다")
    sq, mk = float(c["design_square_length_m"]) * scale, float(c["design_marker_length_m"]) * scale
    if not 0 < mk < sq:
        raise ValueError("마커 한 변은 칸 한 변보다 작아야 한다")
    return BoardSpec(str(c["dictionary"]), int(c["squares_x"]), int(c["squares_y"]), sq, mk,
                     bool(c.get("legacy_pattern", False)), int(c.get("min_corners", 6)),
                     float(c.get("max_reproj_px", 1.0)))


def _dictionary(spec: BoardSpec):
    name = spec.dictionary
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"알 수 없는 ArUco 딕셔너리: {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_board(spec: BoardSpec):
    d = _dictionary(spec)
    if _NEW_API:
        board = cv2.aruco.CharucoBoard((spec.squares_x, spec.squares_y), spec.square_m, spec.marker_m, d)
        if hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(spec.legacy_pattern)
        return board
    if not spec.legacy_pattern and spec.squares_y % 2 == 0:
        raise RuntimeError(f"OpenCV {cv2.__version__} 은 옛 ChArUco 배치만 읽는다 — 행 수가 짝수인 보드는 "
                           "OpenCV 4.7 이상이 필요하다 (pip install -r requirements.txt)")
    return cv2.aruco.CharucoBoard_create(spec.squares_x, spec.squares_y, spec.square_m, spec.marker_m, d)


def board_corners_3d(board) -> np.ndarray:
    """내부 모서리 ID 순서의 보드 좌표 (N×3, m)."""
    pts = board.getChessboardCorners() if hasattr(board, "getChessboardCorners") else board.chessboardCorners
    return np.asarray(pts, dtype=np.float64).reshape(-1, 3)


def board_image(spec: BoardSpec, px_per_m: float = 4000.0) -> np.ndarray:
    """보드 그림 (흰 여백 없음, 회색조). 시험용 합성 장면을 만들 때 쓴다."""
    board = make_board(spec)
    size = (int(round(spec.squares_x * spec.square_m * px_per_m)), int(round(spec.squares_y * spec.square_m * px_per_m)))
    if hasattr(board, "generateImage"):
        return board.generateImage(size, marginSize=0, borderBits=1)
    return board.draw(size, marginSize=0, borderBits=1)


def detect_corners(image: np.ndarray, spec: BoardSpec, board=None) -> tuple[np.ndarray, np.ndarray]:
    """→ (모서리 ID (N,), 영상 좌표 (N×2)). 못 찾으면 빈 배열."""
    board = board if board is not None else make_board(spec)
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if _NEW_API:
        cc, cids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
    else:
        params = cv2.aruco.DetectorParameters_create()
        mc, mids, _ = cv2.aruco.detectMarkers(gray, _dictionary(spec), parameters=params)
        if mids is None or len(mids) == 0:
            return np.empty(0, int), np.empty((0, 2))
        _, cc, cids = cv2.aruco.interpolateCornersCharuco(mc, mids, gray, board)
    if cids is None or len(cids) == 0:
        return np.empty(0, int), np.empty((0, 2))
    return np.asarray(cids).ravel().astype(int), np.asarray(cc, dtype=np.float64).reshape(-1, 2)


@dataclass
class BoardPose:
    T_camera_board: np.ndarray   # 4×4, p_cam = R p_board + t
    ids: np.ndarray
    image_points: np.ndarray     # N×2
    reproj_rms_px: float

    @property
    def n_corners(self) -> int:
        return len(self.ids)

    @property
    def distance_m(self) -> float:
        """카메라 원점 → 보드 평면 수직 거리."""
        n, t = self.T_camera_board[:3, 2], self.T_camera_board[:3, 3]
        return float(abs(n @ t))

    @property
    def tilt_deg(self) -> float:
        """보드 법선과 카메라 광축 사이 각 (0 = 정면)."""
        return float(np.degrees(np.arccos(min(1.0, abs(self.T_camera_board[2, 2])))))


def estimate_board_pose(image: np.ndarray, intrinsics, spec: BoardSpec, board=None) -> BoardPose | None:
    """보드 자세. 모서리가 spec.min_corners 보다 적거나 한 줄로만 보이면 None."""
    board = board if board is not None else make_board(spec)
    ids, img = detect_corners(image, spec, board)
    if len(ids) < max(spec.min_corners, 4):
        return None
    obj = board_corners_3d(board)[ids]
    if np.linalg.matrix_rank(obj - obj.mean(axis=0), tol=1e-6) < 2:
        return None                                  # 모서리가 한 직선 위 → 자세가 정해지지 않는다
    fx, fy, cx, cy = intrinsics
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros(5)
    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    if not ok or len(rvecs) == 0:
        return None
    best = None
    for rv, tv in zip(rvecs, tvecs):                 # 평면 PnP 의 두 해 중 재투영 오차가 작은 쪽
        rv, tv = cv2.solvePnPRefineLM(obj, img, K, dist, rv.copy(), tv.copy())
        proj, _ = cv2.projectPoints(obj, rv, tv, K, dist)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, axis=1))))
        if float(tv.ravel()[2]) > 0 and (best is None or rms < best[2]):
            best = (rv, tv, rms)
    if best is None:
        return None
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(best[0])[0]
    T[:3, 3] = best[1].ravel()
    return BoardPose(T, ids, img, best[2])


def board_depth_error(depth_m: np.ndarray, pose: BoardPose, intrinsics, win_px: int = 2) -> tuple[float, int]:
    """보드 모서리마다 (D435 깊이 − PnP 보드 평면 깊이) 중앙값 (m)과 쓴 모서리 수.

    양수 = 깊이가 실제보다 멀게 나온다. 모서리 둘레 (2·win+1)² 창의 유효 깊이 중앙값을 쓴다.
    """
    n = pose.T_camera_board[:3, 2]
    d = float(n @ pose.T_camera_board[:3, 3])            # 평면: n·p = d (카메라 좌표)
    H, W = depth_m.shape
    errs = []
    for u, v in pose.image_points:
        ui, vi = int(round(u)), int(round(v))
        win = depth_m[max(vi - win_px, 0):vi + win_px + 1, max(ui - win_px, 0):ui + win_px + 1]
        z = win[win > 0]
        if len(z) == 0 or not (0 <= ui < W and 0 <= vi < H):
            continue
        ray = backproject(u, v, 1.0, intrinsics)          # z = 1 인 시선
        denom = float(n @ ray)
        if abs(denom) < 1e-9:
            continue
        errs.append(float(np.median(z)) - d / denom)
    if not errs:
        return float("nan"), 0
    return float(np.median(errs)), len(errs)
