"""Piper Studio 연동 계층 — 설계서의 piper_driver · command_gate · camera_top · camera_color_setup ·
color_normalizer · lighting_monitor 자리를 대신한다.

로봇 CAN과 카메라 장치는 프로그램 하나만 열 수 있다. Piper Studio가 이미 열고 있으므로
우리 코드는 직접 열지 않고 Piper Studio를 통해 읽고 보낸다.

| 설계서 노드             | 대체하는 Piper Studio 기능                                  |
|------------------------|------------------------------------------------------------|
| piper_driver           | robotd (CAN 단독 점유)                                      |
| command_gate           | robotd 안전 필터 safety.filter_goal (범위·변화율·바닥·데드맨) |
| camera_top             | rsd (RealSense 데몬)                                        |
| camera_color_setup 외 2 | 회색 카드 보정 · 조명 감시 (feature/gray-card-calibration.md, lighting-watch.md) |

[확인 필요] 아래 메서드가 기대는 Piper Studio 내부 경로는 대회 전에 시뮬레이션에서 검증한다.
- 스냅샷: rsd 공유 메모리 프레임 또는 /api/vision/segments/{name}/snapshot
- 관절 목표: 웹 조그 세션과 같은 ActionWriter 경로 (backend/app/services/jog.py)
- 정책 실행: services/orchestrator.py 의 set_task → wait_done 흐름
"""

from __future__ import annotations

from aple.schema import ArmCommand, CaptureMode, LightingStatus, SceneSnapshot, Source


class PiperStudio:
    def snapshot(self, source: Source = Source.TOP, mode: CaptureMode = CaptureMode.FULL) -> SceneSnapshot:
        """D435 컬러(회색 카드 보정 적용) + 컬러 정렬 깊이 한 장. scene_id 발급."""
        raise NotImplementedError

    def lighting_status(self) -> LightingStatus:
        """Piper Studio 조명 감시 상태."""
        raise NotImplementedError

    def joint_state(self) -> tuple[float, ...]:
        raise NotImplementedError

    def send(self, cmd: ArmCommand) -> None:
        """관절 목표 전달. 데드맨(약 300ms)보다 빠르게 재발행해야 한다."""
        raise NotImplementedError

    def run_policy(self, task_text: str, timeout_s: float) -> bool:
        """ACT 정책으로 Phase A 실행. 학습 문형과 토큰 단위로 같은 문장을 쓴다."""
        raise NotImplementedError
