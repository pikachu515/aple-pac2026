"""모듈 간 데이터 계약 (전략 종합 II장 4-3).

이 파일을 바꾸면 다른 역할의 코드가 깨진다. 변경은 Pull Request로 올리고
관련 역할의 확인을 받는다.

약속
- 위치: 로봇 베이스 좌표계, mm
- 영상 좌표: 상부(top) D435 컬러 영상 기준 픽셀
- 착색비율: 0~100 (%)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum

import numpy as np


class Grade(str, Enum):
    HIGH = "high"        # 상
    MID = "mid"          # 중
    REJECT = "reject"    # 리젝트


class RejectReason(str, Enum):
    NONE = "none"
    NEAR_BOUNDARY = "near_boundary"      # 등급 경계 근접 (S6-2)
    VIEW_DISAGREE = "view_disagree"      # 뷰 간 편차 과다 (S6-3)
    OVERSIZE = "oversize"                # 그리퍼 파지 한계 초과 (S6-4)


class FailCode(str, Enum):
    """실패 코드 (II장 2-4)."""
    NONE = "none"
    D1 = "D1"   # 검출 실패 / 오검출
    D2 = "D2"   # 마스크 경계 부정확
    G1 = "G1"   # 깊이 결측으로 기하 산출 불가
    C1 = "C1"   # 착색비율이 육안과 불일치
    C2 = "C2"   # 뷰 간 편차 과다
    P1 = "P1"   # 빈 셀 없음
    F1 = "F1"   # 접근 실패
    F2 = "F2"   # 파지 실패
    F3 = "F3"   # 이송 중 낙하
    F4 = "F4"   # 적재 실패


# ---------------------------------------------------------------- 역할 A → 역할 C

@dataclass
class AppleObservation:
    """사과 1개의 영상 관측. 역할 A가 만들고 역할 C가 소비한다."""
    id: int
    bbox: tuple[int, int, int, int]          # (x1, y1, x2, y2) top 픽셀
    score: float                             # 검출 신뢰도
    mask: np.ndarray                         # top 영상 크기의 bool 배열
    normal_z: np.ndarray | None = None       # 마스크 내부 법선의 카메라 축 성분 (S5-3 면적 가중)
    front_roi: tuple[int, int, int, int] | None = None  # front 영상의 대응 영역 (S5-4)
    depth_valid_ratio: float = 1.0           # 마스크 내 깊이 유효 비율 (G1 판정)


# ---------------------------------------------------------------- 역할 A → 역할 B

@dataclass
class AppleGeometry:
    id: int
    xyz_mm: tuple[float, float, float]       # 사과 중심, 로봇 베이스 좌표계
    diameter_mm: float
    size_bin: str                            # 크기 구간 이름 (configs/grasp_bins.yaml)


@dataclass
class BoxState:
    box_id: str                              # 예: "high", "mid", "reject"
    wall_top_mm: float                       # 벽 상단 높이
    cell_heights_mm: list[float]             # 셀별 적재 상단 높이
    free_cells: list[int]                    # 빈 셀 번호 (far-to-near 순서, S7-4)


# ---------------------------------------------------------------- 역할 C → 역할 B

@dataclass
class GradeResult:
    id: int
    grade: Grade
    ratio_pct: float                         # 다뷰 합산 착색비율
    ratio_by_view: dict[str, float] = field(default_factory=dict)  # {"top": .., "front": ..}
    reject_reason: RejectReason = RejectReason.NONE


# ---------------------------------------------------------------- 역할 D → 역할 B

@dataclass
class GraspParams:
    size_bin: str
    open_width_mm: float                     # 사전 개방 폭
    grasp_height_mm: float                   # 파지 높이 (사과 중심 기준 오프셋)
    effort: int                              # GripperCtrl 토크 설정 (0~5000)


# ---------------------------------------------------------------- 사이클 결과

@dataclass
class PlacePlan:
    apple_id: int
    box_id: str
    cell: int
    descend_to_mm: float                     # 안전 하강 목표 고도 (S7)
    grasp: GraspParams
    task_text: str                           # 학습 문형과 토큰 단위로 동일해야 함 (II장 2-3 ⑧)


@dataclass
class CycleLog:
    cycle: int
    plan: PlacePlan | None
    success: bool
    fail_code: FailCode = FailCode.NONE
    cycle_time_s: float = 0.0
    note: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fail_code"] = self.fail_code.value
        return d


# ---------------------------------------------------------------- 역할 간 데이터 흐름
# docs/interfaces.md 의 다이어그램은 이 표와 위 클래스들로부터 자동 생성된다.
# 흐름을 바꾸려면 다이어그램이 아니라 이 표를 고친다.

ROLES = {
    "A": "역할 A<br/>비전 타깃 인식 · 3D 위치",
    "C": "역할 C<br/>다면 촬영 · 색상 품질 판별",
    "D": "역할 D<br/>하드웨어 고도화",
    "B": "역할 B<br/>로봇 파지 · 손상 방지 제어",
    "S": "공동<br/>오케스트레이터 · 평가",
}

FLOWS = [
    ("A", "C", AppleObservation),
    ("A", "B", AppleGeometry),
    ("A", "B", BoxState),
    ("C", "B", GradeResult),
    ("D", "B", GraspParams),
    ("B", "S", CycleLog),
]
