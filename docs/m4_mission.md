# M4 미션 실행과 복구

`MissionController`는 기존 `SceneRunner` 위에서 한 트레이의 작업을 관리한다.
`tools/run_cycle.py --live --execute`의 실기 런타임에 연결되어 있다.
공유 `schema.py` 계약과 기존 M1/M2/M3/M5 호출 형태는 바꾸지 않았다.
기존 드라이런·시뮬레이션 실행기는 그대로 유지한다.

## 동작과 설정

- `configs/workcell.yaml`의 `workcell.expected_apple_count`(현재 5)만큼 사과가
  보이고, `tray_clear=True`, 가려진 사과 없음, ID 중복 없음일 때 시작한다.
  적재 확인이 끝난 사과 수만큼 예상 잔여 수를 줄인다. 처음 확인한 ID를 유지하며
  사라지거나 다른 ID로 바뀐 사과를 임의로 대체하지 않는다.
- DEFER 또는 목표 칸 부족이면 새 FULL TOP 영상을 받아 위치·등급·점유·계획을
  다시 만든다. `configs/orchestration.yaml`의 `max_rescan`은 미션 전체의
  문제 해결 재촬영 상한, `max_retry`는 사과별 추가 실행 상한이다.
  정상 적재 확인 촬영은 문제 해결 재촬영 예산에 포함하지 않는다.
- 성공한 적재 확인 영상을 다음 사과 계획에 재사용한다. 오래된 좌표로 연속 집기하지 않는다.
- NO_CONTACT/IK_FAIL/COLLISION/SCENE_MISMATCH 복구는 기존 RecoveryPolicy를
  따르되, 빈손 및 안전한 스캔 가능 상태를 확인하는 `prepare_recovery(item)`가
  정확히 `True`를 반환해야 실제 재시도한다. 기존 모드 B 미끄럼 재파지 1회는 유지한다.
- 실기 어댑터는 scripted 실행에서 **닫기 명령 이전의 APPROACH/DESCEND 실패**만
  재시도를 허용한다. 재시도 상한에 도달해도 빈손과 재촬영 예산이 확인되면 해당 사과를
  SKIP하고 알린 뒤 다른 사과를 진행한다. 관절 피드백을 확인하고, 대상 사과도 다시 장애물에 포함한다.
  파지 이후 실패를 실패 코드만 보고 빈손으로 간주하지 않는다.
- DROP은 우선 팔을 움직이지 않고 새 TOP 영상을 기록한다. 현재 실기 연결에서는
  이후 빈손·복구 가능 여부를 보증할 수 없어 중단한다. 낙하 후 자동 집기를 활성화하려면
  검증된 복구 공급자를 연결해야 한다. 위치가 바뀐 과실은 재계획 때 같은 ID로 추적되어야 한다.
- 조명 FAULT는 실행하지 않는다. `recalibrate_light()`가 실제 보정을 수행하고 `True`를
  반환하는 공급자가 없다면 운영자 조치를 요청하며 중단한다. 재시작 전 카드 보정을 확인한다.
- `verify_timeout_s`를 초과한 적재 확인은 TIMEOUT으로 기록하고 다음 집기를 차단한다.
  동기식 촬영 호출 자체를 강제 취소하는 값은 아니다. 카메라/Studio 전송 시간 제한도 필요하다.
- `grade_wait_timeout_s`와 `pipelining.enabled`는 이번 변경에서 활성화하지 않았다.
  현재 판정은 동기식 함수 호출이며 비동기 판정 대기/병렬 작업 계약은 별도 구현 대상이다.

## 기록과 운영

`orchestration.logging.csv_dir` 아래 두 파일을 생성한다.

| 파일 | 내용 |
| --- | --- |
| `cycles.csv` | 각 사과의 실행 시도 결과와 예외·미완료 장면 행 |
| `mission_events.jsonl` | 상태, 재촬영 장면 ID, 복구 선택·시도, 중단 이유 및 운영자 안내 |

미션 안의 모든 CSV 행은 최초 촬영 `scene_id`와 같은 `cycle_index`를 사용한다.
같은 사과가 재시도되면 여러 행이 생기므로 행 수를 과실 수로 세면 안 된다.
재촬영·재계획 시간은 미션 행에 나누어 기록하여 같은 계획 시간을 중복 합산하지 않는다.
`apple_id=0`은 장면 단위 행이다. 최초 촬영 자체가 실패하면 `scene_id=0`도 사용한다.
이 행의 과실 측정값은 NaN이며 실측 과실 데이터로 해석하지 않는다.

촬영·계획·실행 예외도 CSV로 기록한다. 중단 상태이면 다음 사이클과 reset 이동을
실행하지 않으며 CLI는 실패 종료한다. `stop()`은 다음 작업을 막는 협력적 중단이다.
진행 중인 이동의 긴급 정지는 기존 HAL/E-stop을 사용한다. 중단 후 같은 인스턴스를
자동으로 해제하지 않으므로 현장 상태를 확인하고 명시적으로 새 실행을 시작한다.
CLI는 중단 사유와 설정된 로그 경로를 표시한다. 모든 중단을 조명 문제로 안내하지
않으며, 중단은 실패 종료 상태를 유지한다. 파지 상태를 모르면 운영자가 확인해야 한다.

```bash
tail -f logs/mission_events.jsonl
python tools/run_cycle.py --live --execute --cycles 1
```

## 팀 연결부

- A(M1): FULL TOP의 노출별 증가하는 `scene_id`/`stamp_s`, 정확한 `tray_clear`,
  사과 `occluded` 및 장면 간 영구 ID가 필요하다. 개수만 맞고 ID가 바뀌면 M4가 멈춘다.
  인식 추적기의 ID 재사용 문제는 이번 M4 변경으로 해결되는 문제가 아니다.
- C(M2): DEFER 재촬영마다 판정기를 다시 호출한다. 조명 보정 공급자를 연결할 때
  실제 카드 보정 완료 후에만 성공을 반환해야 한다.
- B(M3/M5): 파지 전후 실패 단계가 정확해야 복구가 허용된다. 파지 이후 미확인 상태의
  예약 칸은 비우지 않는다. TCP·속도·접촉량·충돌 외형 등 실측 설정은 기존 요구 사항을 유지한다.
- 통합 담당: 별도 실행 진입점을 쓰면 `MissionController`를 연결하고 `CycleManager`에
  `failure_records=mission.failure_records`, `stop_requested=lambda: mission.halted`를 전달한다.
  `build_live_runtime().runner`는 이제 MissionController이며, 내부 SceneRunner는
  `.runner.runner`, 파이프라인은 기존 `.pipeline`에서 접근한다.
  `event_sink`에 UI 알림을 연결할 수 있다. 기본은 터미널과 JSONL 알림이며 웹 팝업은 아니다.

물리 설정값을 추가로 추정하거나 시뮬레이션 임의값을 실기에 주입하지 않았다.
