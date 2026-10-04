"""파지력 유지·미끄러짐 감지 (설계서 gripper_force_controller). 필수 (토크 설정 변경 범위).

open_width 까지 연 뒤 저속 폐쇄 → effort 상승·위치 정지로 접촉 판정
개구 폭 이동 > slip_width_m 또는 effort 저하 → SLIP
Piper Studio 기본 토크는 1000 고정 (arm.py · publish.py · piper_motors_bus.py) — 변경 경로 확인 필요
"""

from __future__ import annotations

from aple.schema import GraspProfile


def effort_for(profile: GraspProfile, calib_table: list[tuple[int, float]]) -> int:
    """grasp_force_n → effort (FSR 보정표 선형 보간)."""
    raise NotImplementedError
