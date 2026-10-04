"""집기·놓기 실행 (설계서 motion_executor + scripted_backend / act_backend + soft_landing). 필수.

Phase A  접근 → 하강 → 파지 → 상승 → 이송 → hover    : scripted 또는 act 백엔드
전환     hover 도달 + 파지 유지 확인                   : 미충족 → HOVER_NOT_REACHED / NOT_HOLDING
Phase B  감속 하강 → release → 후퇴                    : 항상 scripted (낙하 높이 보장)

scripted: Piper Studio kinematics.ik 로 관절 목표 산출 → PiperStudio.send
act     : PiperStudio.run_policy. ACT 그리퍼 출력은 닫기/열기 신호로만 쓰고 폭·토크는 GraspProfile 값
"""

from __future__ import annotations

from aple.hal.piper_studio import PiperStudio
from aple.schema import AppleState, ExecResult, GraspProfile, PlacementPlan


class MotionExecutor:
    def __init__(self, hal: PiperStudio, cfg: dict) -> None:
        self.hal = hal
        self.cfg = cfg

    def move_named(self, name: str) -> ExecResult:
        """'home' / 'scan' (팔을 시야 밖으로)."""
        raise NotImplementedError

    def pick_place(self, apple: AppleState, profile: GraspProfile, plan: PlacementPlan,
                   backend: str = "scripted") -> ExecResult:
        raise NotImplementedError

    def inspect(self, profile: GraspProfile) -> ExecResult:
        """모드 B (확장): 툴 수평으로 기울인 뒤 J6 90°씩 4회, 각도마다 정지 후 스냅샷."""
        raise NotImplementedError
