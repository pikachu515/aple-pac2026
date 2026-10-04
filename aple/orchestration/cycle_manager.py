"""공정 상태 기계 (설계서 cycle_manager). 필수 (단순형).

BOOT → CALIB_LIGHT → HOME → IDLE → SCAN → PLAN → EXEC → VERIFY → PLAN … → FINISH
(적응형) PLAN → PICK → INSPECT → WAIT_GRADE → PLAN_PLACE → PLACE → VERIFY

예외
- NO_CONTACT / IK_FAIL / COLLISION : POSE_ONLY 재스캔 후 재시도 (max_retry) → 초과 시 SKIP
- SLIP          : 재파지 1회 → 실패 시 안전 위치 복귀
- DROP          : FULL 재스캔
- SCENE_MISMATCH: POSE_ONLY 재스캔
- 조명 FAULT    : 스캔 중단, 회색 카드 재보정
- 개수 불일치·가림: RESCAN (max_rescan)

dry_run=True 면 로봇 없이 판단 경로만 돈다 (전략 IV장 2단계 단독 검증).
"""

from __future__ import annotations

from aple.schema import CycleRecord


class CycleManager:
    def __init__(self, cfg: dict, dry_run: bool = False) -> None:
        self.cfg = cfg
        self.dry_run = dry_run

    def run(self) -> list[CycleRecord]:
        raise NotImplementedError
