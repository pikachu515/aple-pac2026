"""모듈 간 데이터 계약.

기준: 팀 Notion 「아키텍처 및 노드 설계 (1차 10/03)」의 메시지 정의.
ROS 2 메시지를 Python dataclass 로 옮긴 것이며, 이름·필드는 설계서를 따른다.
대회 최소 구성에 필요한 것만 두었다 (docs/architecture.md).

이 파일을 바꾸면 다른 역할의 코드가 깨진다. 변경은 Pull Request로 올리고
관련 역할의 확인을 받는다. 바꾼 뒤 python tools/gen_interfaces.py 로 다이어그램을 다시 만든다.

규약 (설계서 1.4)
- 단위: SI (m, N, s, rad). mm는 설정 파일 키(_mm)에만
- 좌표: 위치는 base_link 기준. 영상 좌표는 top(D435) 컬러 픽셀
- 번호 공유: 등급과 상자는 같은 번호 (UPPER=0, MIDDLE=1, REJECT=2)
- scene_id: 스냅샷마다 1 증가. 위치·등급·파지 데이터는 같은 scene_id끼리만 결합
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import IntEnum

import numpy as np


# ================================================================ 공통 상수

class Source(IntEnum):
    TOP = 0            # 트레이·상자 스캔
    IN_HAND = 1        # 모드 B 검사 자세 캡처


class CaptureMode(IntEnum):
    FULL = 0           # 위치 + 색상
    POSE_ONLY = 1      # 위치·점유만


class Grade(IntEnum):
    """등급 = 상자 번호."""
    UPPER = 0          # 상
    MIDDLE = 1         # 중
    REJECT = 2         # 리젝트


class RejectReason(IntEnum):
    NONE = 0
    BORDERLINE = 1     # 판정 구간이 기준선에 걸침
    VIEW_DISAGREE = 2  # 뷰 간 편차 과다 (모드 B)
    OVERSIZE = 3       # 그리퍼 범위 초과 → 이동하지 않음
    UNDERSIZE = 4      # 그리퍼 범위 미만 → 이동하지 않음
    LOW_CONF = 5       # 검출 신뢰도 부족 또는 조명 이상
    LOW_COVERAGE = 6   # 관측 커버리지 부족
    OCCLUDED = 7       # 리젝트 아님 — 재촬영 대상
    UNSAFE_GRASP = 8   # 파지 불가 → 이동하지 않음


class InspectMode(IntEnum):
    TOP_SINGLE = 0     # 모드 A: D435 탑뷰 단일
    IN_HAND_MULTI = 1  # 모드 B: 파지 후 기울임 + J6 회전


class InspectionPolicy(IntEnum):
    NEVER = 0          # 모드 A만
    ALWAYS = 1         # 모드 B
    ADAPTIVE = 2       # 애매한 사과만 B


class TaskAction(IntEnum):
    PICK_PLACE = 0
    PICK_INSPECT_PLACE = 1
    DEFER = 2          # 재촬영 후 재평가
    SKIP = 3           # 이동하지 않음


class Phase(IntEnum):
    """실행 단계 (설계서 Phase.msg)."""
    IDLE = 0
    APPROACH = 1
    DESCEND = 2
    GRASP = 3
    LIFT = 4
    TRANSIT = 5
    HOVER = 6
    SOFT_LAND = 7
    RELEASE = 8
    RETREAT = 9
    INSPECT = 10
    MOVE_NAMED = 11


class FailureCode(IntEnum):
    """실행 실패 코드 (설계서 FailureCode.msg)."""
    NONE = 0
    IK_FAIL = 1
    COLLISION = 2
    NO_CONTACT = 3
    SLIP = 4
    DROP = 5
    TIMEOUT = 6
    GATE_VIOLATION = 7
    ABORTED = 8
    NOT_HOLDING = 9
    HOVER_NOT_REACHED = 10
    SCENE_MISMATCH = 11


class FailStage(IntEnum):
    """파이프라인 단계별 실패 분류 (전략 II장 2-4). 실패 분포 집계용."""
    NONE = 0
    D1 = 1   # 검출 실패 / 오검출
    D2 = 2   # 마스크 경계 부정확
    G1 = 3   # 깊이 결측으로 기하 산출 불가
    C1 = 4   # 착색비율이 육안과 불일치
    C2 = 5   # 뷰 간 편차 과다
    P1 = 6   # 빈 칸 없음
    F1 = 7   # 접근 실패
    F2 = 8   # 파지 실패
    F3 = 9   # 이송 중 낙하
    F4 = 10  # 적재 실패


class LightState(IntEnum):
    OK = 0
    CORRECTING = 1
    FAULT = 2


# ================================================================ M0 공통 인프라 → M1·M2

@dataclass
class SceneSnapshot:
    """한 시점의 영상·깊이 묶음. 하류 모듈은 라이브 영상 대신 이것만 쓴다."""
    scene_id: int
    source: Source
    capture_mode: CaptureMode
    color: np.ndarray                         # H×W×3 BGR (회색 카드 보정 적용 영상)
    depth_m: np.ndarray                       # H×W float32, 컬러에 정렬, 0 = 결측
    intrinsics: tuple[float, float, float, float]   # fx, fy, cx, cy (px)
    stamp_s: float
    tray_clear: bool = True                   # 팔이 트레이를 가리지 않음
    apple_id: int = 0                         # IN_HAND 대상
    view_index: int = 0                       # IN_HAND 뷰 번호
    tool_pose: np.ndarray | None = None       # IN_HAND: 4×4 TCP 포즈 (base_link)


@dataclass
class LightingStatus:
    """Piper Studio 조명 감시 결과를 옮겨 담은 것."""
    light_state: LightState
    delta_e: float
    card_valid: bool


# ================================================================ M1 인식 · 3D 위치 추정

@dataclass
class AppleDetection2D:
    scene_id: int
    det_id: int
    bbox: tuple[int, int, int, int]           # x, y, w, h (px)
    confidence: float
    mask: np.ndarray                          # H×W bool


@dataclass
class AppleState:
    """사과 1개의 3D 상태 (base_link)."""
    scene_id: int
    apple_id: int
    det_id: int
    center_m: tuple[float, float, float]      # 구 중심
    diameter_m: float
    top_surface_z_m: float
    clearance_m: float                        # 이웃·트레이 벽과의 최소 간격
    fit_residual_m: float
    confidence: float
    depth_valid_ratio: float                  # 마스크 내 깊이 유효 비율 (G1 판정)
    displacement_m: float = 0.0               # 이전 scene 대비 이동량
    occluded: bool = False


@dataclass
class BoxOccupancy:
    """상자 1개의 칸별 적재 높이·점유 격자."""
    scene_id: int
    box_id: Grade
    rows: int
    cols: int
    cell_pitch_m: float
    wall_height_m: float                      # 설정값
    wall_height_measured_m: float             # 측정값 (교차검증)
    origin_m: tuple[float, float, float]      # 격자 원점 (base_link)
    heights_m: list[float]                    # row-major, 칸별 상면 높이
    occupied: list[bool]
    valid: bool                               # 팔 가림·깊이 부족이면 False


# ================================================================ M2 색상 기반 품질 판별

@dataclass
class ColorRatio:
    """사과 1개의 착색비율 추정 (등급 판정 입력). 판정 카메라는 D435 하나."""
    scene_id: int
    apple_id: int
    inspect_mode: InspectMode
    ratio: float                              # 0~1
    ratio_lo: float
    ratio_hi: float
    coverage: float                           # 관측 표면 / 전체 표면
    view_disagreement: float = 0.0            # 모드 B 뷰 간 편차
    num_views: int = 1
    variety: str = "fuji"


@dataclass
class GradeResult:
    scene_id: int
    apple_id: int
    grade: Grade
    ratio: float
    ratio_lo: float
    ratio_hi: float
    coverage: float
    margin: float                             # 기준선까지 거리 (부호 포함)
    reject_reason: RejectReason
    final: bool                               # False: 인핸드 검사 후 재판정 대상
    inspect_mode: InspectMode
    variety: str


# ================================================================ M5 판단 · 계획

@dataclass
class GraspProfile:
    """크기 구간별 파지 프로파일. 역할 D가 실측한 표를 역할 B가 적용한다."""
    scene_id: int
    apple_id: int
    size_bin: int
    est_mass_kg: float
    open_width_m: float
    grasp_height_offset_m: float
    grasp_force_n: float
    effort: int                               # GripperCtrl 토크 설정 (0~5000), 보정표로 환산
    feasible: bool


@dataclass
class TaskItem:
    scene_id: int
    apple_id: int
    action: TaskAction
    grade: Grade | None = None
    grade_scene_id: int = 0                   # 등급을 재사용한 경우 원래 scene


@dataclass
class PlacementPlan:
    scene_id: int
    apple_id: int
    box_id: Grade
    cell_index: int
    place_center_m: tuple[float, float, float]   # 놓을 때 사과 중심
    hover_z_m: float                          # 상자 위 정지 고도 (Phase A 끝)
    release_z_m: float                        # 놓는 고도 (Phase B)
    planned_drop_m: float                     # ≤ 0.03
    feasible: bool


# ================================================================ M3 로봇 실행 · 손상 방지 ↔ M0

@dataclass
class ArmCommand:
    """Piper Studio 로봇 데몬으로 보내는 목표. 안전 필터는 데몬 쪽에 있다."""
    position_rad: tuple[float, float, float, float, float, float]
    gripper_width_m: float
    gripper_effort: int
    cmd_source: str                           # "scripted" | "act" | "soft_landing" | "inspect" | "move_named"


@dataclass
class ExecResult:
    success: bool
    failure_code: FailureCode
    phase_reached: Phase
    slip_count: int = 0


# ================================================================ M4 공정 관리

@dataclass
class CycleRecord:
    """사이클 1행 로그 (설계서 CycleRecord 축약). 발표 지표의 원천."""
    cycle_index: int
    scene_id: int
    apple_id: int
    variety: str
    inspection_policy: InspectionPolicy
    task_action: TaskAction
    backend: str                              # "scripted" | "act"
    diameter_m: float
    ratio: float
    ratio_lo: float
    ratio_hi: float
    coverage: float
    grade: Grade
    reject_reason: RejectReason
    target_box: Grade
    target_cell: int
    grasp_force_set_n: float
    planned_drop_m: float
    settle_error_m: float
    t_scan_s: float
    t_plan_s: float
    t_pick_s: float
    t_inspect_s: float
    t_place_s: float
    t_total_s: float
    success: bool
    failure_code: FailureCode
    fail_stage: FailStage

    def to_dict(self) -> dict:
        return {k: (v.name if isinstance(v, IntEnum) else v) for k, v in asdict(self).items()}


# ================================================================ 모듈 간 데이터 흐름
# docs/interfaces.md 의 다이어그램은 이 표와 위 클래스들로부터 자동 생성된다.
# 흐름을 바꾸려면 다이어그램이 아니라 이 표를 고친다.
# 모듈 번호는 역할 분담 기준 (모듈 2 = 색상, 모듈 3 = 로봇).

MODULES = {
    "M0": "M0 공통 인프라<br/>Piper Studio 연동 · 역할 A·B",
    "M1": "M1 인식 · 3D 위치<br/>역할 A",
    "M2": "M2 색상 품질 판별<br/>역할 C",
    "M5": "M5 판단 · 계획<br/>역할 B·D·A",
    "M3": "M3 로봇 실행 · 손상 방지<br/>역할 B·D",
    "M4": "M4 공정 관리<br/>통합 담당",
}

FLOWS = [
    ("M0", "M1", SceneSnapshot),
    ("M0", "M2", LightingStatus),
    ("M1", "M2", AppleDetection2D),
    ("M1", "M2", AppleState),
    ("M1", "M5", AppleState),
    ("M1", "M5", BoxOccupancy),
    ("M2", "M5", GradeResult),
    ("M5", "M4", TaskItem),
    ("M5", "M4", GraspProfile),
    ("M5", "M4", PlacementPlan),
    ("M4", "M3", PlacementPlan),
    ("M4", "M3", GraspProfile),
    ("M3", "M0", ArmCommand),
    ("M3", "M4", ExecResult),
]

# 모듈 내부에서만 쓰지만 다이어그램에 함께 표시할 데이터
INTERNAL = [ColorRatio, CycleRecord]
