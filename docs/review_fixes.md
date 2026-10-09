# 2026-10-09 역할 B 코드 검토 후 개선

검토 보고서 `logs/code-review-20261009/review.md`의 오류와 실행 연결을 기존 구조 안에서 수정했다. 작업 브랜치는 `execution/review-fixes-20261009`다. 실기 명령은 실행하지 않았으며, 미측정 값을 채우거나 시뮬레이션의 가정값을 실기 설정에 옮기지 않았다. 기존 동영상·속도 실험 설정은 보존했다.

## 수정 대응표

| 검토 | 코드 변경 | 남은 확인 |
|---|---|---|
| R1 마지막 하강 | `min(hover_z, release_z + final_descent_m)`로 마지막 구간만 감속. 짧은 하강도 전체 감속. 시뮬레이션 전용 보정 제거 | 실물 사과 손상과 하강 속도 측정 |
| R2 회전축 불일치 | 기울인 툴의 로컬 축 회전으로 공통 검사 자세 생성. 기존 옆 회전 시뮬레이션도 동일 함수 사용. 사과 중심과 측정 apple-in-TCP 유지 | 각 검사 자세의 실물 IK/가림/관측률. 이동하면서 -100° 준비하는 실기 최적화는 추가하지 않음 |
| R3 파지 가정 | 안쪽 닫힘 목표와 접촉 부근 폭 안정 대기, 리프트 후 새 깊이 영상으로 사과 존재 확인. 파지 후 폭 변화를 운반 중 감시 | 압축량·폭·대기시간 실측. 폭과 영상은 접촉력 센서가 아니며 과부하/손상 측정을 대체하지 않음 |
| R4 속도·통신 | 관절별 속도·가속도 제한으로 최소저크 시간 궤적 생성. 검사된 직선의 불필요한 분할점만 제거. 경로 모서리는 유지. 한 관절 경로를 한 jog 세션으로 전송. 측정 속도/추종 오차/폭 이탈 시 중단·세션 정리. 다음 목표 시각과 전체 deadline을 구분해 피드백 지연의 경계 오인 제거 | 실제 HTTP 주기, 모터 추종, 정지 시간 검증. 명령 궤적 한계가 모터의 가속도 보장은 아님 |
| R5 실기 연결 | `hal/live_runtime.py`가 Studio·공통 TCP IK/FK·충돌 검사·실시간 촬영·파지 확인·재촬영·재시도·적재 확인을 조립. `run_cycle --live --execute`에 연결 | 미측정 설정을 모두 채운 뒤 실기 단계별 검증 |
| R6 예약·안착 | 계획 예약은 복사본에서 처리. 실행 시작 때만 확정. 적재 후 새 TOP 영상의 점유·높이 변화로 검증, 실패하면 중단. 남은 ID는 새 좌표로 재계획 | 높이 오차 허용치, 사과 재추적 안정성. 상자 위치가 달라지면 검증 실패로 정지 |
| R7 재파지 이웃 | 전체 관측 사과로 파지 후보를 만든 뒤 대상 ID만 선택. 재시도 때 충돌 환경도 갱신 | 실제 이웃/트레이 간섭 |
| R8 운반 충돌 | 사과 부피, 관측된 다른 사과, 상자 벽/적재 칸, 실측 트레이 벽을 검사. 설정한 비인접 링크 쌍과 사과/어댑터의 링크 간섭도 검사. 릴리스 후 가상 운반 부피 제거 | 링크 쌍·벽 두께·장애물·여유 실측. 구형 근사/유한 샘플링이며 완전한 연속 충돌 인증은 아님 |
| R9 개방 범위 | 사과 직경뿐 아니라 개방 여유까지 실제 접촉 폭 범위와 대조. 폭이 없거나 초과하면 infeasible/SKIP | 장착 후 실제 닫힘/열림 접촉 폭 |
| R10 상자 축 | 가상 상자와 무작위 배치/성공 평가의 x=cols, y=rows 통일 | 실측 축 정렬과 칸 인덱스 확인 |
| R11 최신 영상 | 촬영 시작 이후의 새 노출만 수용. 중복/오래된/미래 시각 프레임 거부. 컬러·깊이 중 이른 노출 시각으로 둘 다 최신인지 확인 | Studio 발행자의 wall clock 동기화와 프레임 지연 |
| R12 기록 | 최초 캡처 시간을 포함하고 장면 공통 시간을 행에 나눠 기록. 실제 pick/inspect/place 시간, 안착 오차 기록. 고정 토크의 미측정 N 값은 NaN | 재시도/복구는 전체 `t_total_s`로 분석. 단계별 값만으로 전체 복구 비용을 추정하지 말 것 |

## 실기 전에 입력할 값

`configs/execution.yaml`의 `execution.motion`:

- `velocity_limits_rad_s`, `acceleration_limits_rad_s2`: 관절별 6개, 단위 rad/s와 rad/s².
- `command_period_s`: 실제 Studio가 안정적으로 처리하는 명령 주기.
- `tracking_tolerance_rad`, `velocity_tolerance_rad_s`: 추종·측정 속도 오차 허용치.
- `capture_joint_tolerance_rad`: 촬영 전후 관절 변화 허용치. 관절이 변하면 해당 캡처를 채택하지 않는다.

`execution.gripper`:

- `closing_compression_m`: 과실/패드 접촉을 반영한 안쪽 닫힘 목표의 실측량.
- `contact_settle_s`, `contact_width_tolerance_m`, `contact_stability_m`: 접촉 폭과 안정 대기의 실측값.
- `control_mode: fixed_effort` 경로만 이번 Studio 실기 조립에 연결했다. `fixed_effort`와 Studio 설정 토크가 일치해야 한다. FSR 또는 크기별 힘 제어는 별도 백엔드가 필요하다.
- `opening_margin_m`, `slip_width_m`, `recovery.lost_grip_width_m`, `placement_verify.height_tol_m`의 기존 초기값도 현장 검증한다.

`configs/safety.yaml`의 `segment_validation`:

- `sample_step_rad`, `floor_z_m`, `clearance_m`, `wall_thickness_m`, `obstacle_aabbs_m`.
- `self_collision_pairs`: 검사할 비인접 링크 이름 쌍. `payload_self_links`: 사과/어댑터와 검사할 링크 이름. `[]`는 명시적으로 확인한 경우만 사용하며 자동으로 빈 목록을 만들지 않는다.
- `support_contact_tolerance_m`: 리프트 시작의 바닥 접촉에서만 허용할 깊이/과실/보정 오차. 일반 운반에서는 이 허용치를 쓰지 않는다. 가상 셀의 0.001 m를 실기에 복사하면 안 된다.
- 측정 추종 오차와 구형 기하 근사 오차를 포함해 여유·샘플 간격을 정한다. 추종 오차 감시는 이미 생긴 오차를 검출하는 것이므로 Studio 보호 기능도 유지한다.

`configs/workcell.yaml` 및 장착 설정:

- `T_base_camera`, `tcp_offset_m`, 트레이 중심/크기/벽 높이, 등급별 상자 격자/원점/칸 간격/벽 높이, `home_joint_rad`, `scan_joint_rad`.
- `piper_web.gripper_width_range_m`: **수제 어댑터 장착 후 접촉면 사이 실제 간격**. 제품 기본 스트로크나 사진의 점 사이 거리를 그대로 쓰지 않는다.
- 사진 치수는 기존 mm 참조값으로 유지했다. 파지 중심·장착 변환은 실측값을 SI 단위로 입력한다.
- `inspection.views.center_m`은 선택적인 실측 검사 중심이다. null이면 집어 올린 사과 위치에서 검사한다. 각도/틸트는 실제 관절 범위에서 가능한 자세로 정한다.

카메라/조명:

- `live_camera.depth_calib`, Studio의 D435 스트림, 정렬 깊이 인코딩 범위와 노출 시각을 확인한다.
- 측정한 `color.card_json`을 사용한다. 조명 FAULT 또는 확인 실패 시 다음 명령을 실행하지 않는다.
- `piper_web.enabled`와 `robotd_safety_confirmed`는 현장 확인 후 켠다. 실행 대상은 실제 follower 이름이어야 한다.

## 담당별 연결 확인

- **B / dbrua3311**: 시간 궤적·추종/중단, TCP·그리퍼 폭·운반 부피, 홈/스캔/검사/하강 경로.
- **A / lsb1017**: 새 프레임 시각, 카메라–로봇 보정, 적재 후 ID 재추적과 박스 점유/높이. 상자 격자는 열 +x, 행 +y. 상자 탐색을 새로 추가한 변경은 아니다.
- **C**: 새 공통 검사 자세에서 실제 보이는 면·핑거 제외 마스크·관측률·재등급. 착색 임계/가중치/리젝트 판정은 이번 수정에서 바꾸지 않았다.
- **D / uhdahmrok 및 공통 통합**: 장착 후 기하/접촉 폭/패드 압축 실측, 보호 범위, 손상·슬립 측정. `grasp_force_set_n=NaN`은 미측정이며 0 N이 아니다. torque 조회는 설정 확인이다.

## 실행과 결과 해석

기존 장치 없는 기본 통합 검증:

```bash
python tools/run_cycle.py --simulate --trace logs/simulation/trace.json
```

실기 드라이런과 실행은 각각 다음 명령이다. 실기는 설정이 완성되지 않으면 누락 항목을 표시하고 카메라/로봇 실행 전에 거부한다.

```bash
python tools/run_cycle.py --live --cycles 1
python tools/run_cycle.py --live --execute --cycles 1
```

`--simulate`의 기본 운동학 실행과 이전 물리 실험 보고서의 성공은 새 실기 루프의 영상 기반 안착 검증과 구별한다. 실기 조립은 한 사과씩 실행→스캔→안착 검증→남은 사과 재인식/재계획으로 진행한다. 기록의 외부 scene_id는 최초 장면이며 각 실행 항목은 내부적으로 최신 장면을 사용할 수 있다. ID를 잃으면 임의의 다른 사과를 대신 집지 않는다.

새 짧은 물리 재확인 결과는 `logs/review-fixes-20261009/physics-recheck/seed-42/`에 저장했다. 사과 **1/1 운반 성공**, 물리 시간 약 **17.9 s**, 관측률 **0.258**, 최종 판정 **REJECT**다. 이는 한 장면의 물리 운반 회귀 확인이며 분류 정확도·전체 장면 성공률·실기 안전 속도 100% 증명이 아니다. 기존 영상은 덮어쓰지 않았고 이 재확인에서는 영상을 새로 만들지 않았다.

전체 테스트에는 보관 실험 2개 폴더의 회귀 테스트도 포함했다. 최종 로그는 `logs/review-fixes-20261009/pr-validation-final.txt`다.

최종 테스트 로그와 초기 코드 백업은 `logs/review-fixes-20261009/`에 있다. 이 폴더와 영상은 커밋 대상이 아니다.

최종 pytest: **636 passed, 1 skipped, 1 warning in 241.89s (0:04:01)**. 환경의 구형 pytest는 `pythonpath` 설정 경고 1개를 출력했다. Python 3.10과 설치된 Piper 수학/물리 모듈에서 검증했다.
