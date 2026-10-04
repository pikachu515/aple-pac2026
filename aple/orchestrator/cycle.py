"""한 사이클: 캡처 → 판단 → 태스크 지정 → 완료 대기 → 평가 → 리셋 (S12).

각 단계의 중간 산출물과 실패 코드를 CycleLog 로 남긴다 (원칙 3 · 관측 가능성).
판단 경로는 정책 없이도 실행 가능해야 한다 — 로봇 결합 전 단독 검증용 (IV장 2단계).
"""

from __future__ import annotations

from aple.schema import CycleLog


def run_cycle(cycle: int, dry_run: bool = False) -> CycleLog:
    raise NotImplementedError
