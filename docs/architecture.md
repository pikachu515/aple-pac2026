# 아키텍처 — 설계서와 구현의 대응

- 논리 설계 기준: 팀 Notion 「아키텍처 및 노드 설계 (1차 10/03)」
- 대회 운영·플랫폼 기준: 전략 종합 문서 II장 5절
- 데이터 정의: [`aple/schema.py`](../aple/schema.py), 다이어그램: [interfaces.md](interfaces.md)

설계서의 모듈·노드 경계와 메시지 정의를 그대로 따르고, 대회 구현은 **Piper Studio 위의 Python 모듈**로 한다(1절 D-1).
노드는 함수·클래스로, 토픽은 `schema.py` 의 데이터 클래스로 대응한다.
대회 후 ROS 2로 옮길 때도 경계가 그대로 유지된다.

## 1. 설계 결정

설계서는 대회 조건(현장 개발 약 21시간, 로봇은 현장 지급, Piper Studio 기반 환경)을 전제로 하지 않고 작성된 범용 설계다. 아래 네 가지는 대회 조건에 맞춰 구현 방식을 확정한 것이다. 설계서의 모듈 경계·메시지·판정 규칙은 바꾸지 않았다.

| # | 항목 | 설계서 원안 | 확정 |
|---|---|---|---|
| D-1 | 로봇·카메라 연결 | ROS 2 자체 드라이버 | **Piper Studio 연동** |
| D-2 | 실행 방식 | 스크립트 기본, ACT 선택 | **스크립트 먼저 완성 + ACT 병행, 성공률 비교 후 시연 방식 결정** |
| D-3 | 판정 카메라 | D435 탑뷰 (정책 입력도 탑뷰) | **판정은 D435 탑뷰만, 정면·손목 웹캠은 ACT 입력에 추가** |
| D-4 | 모듈 번호 | 본문 M2 = 실행, M3 = 색상 | **M2 = 색상, M3 = 실행** |

### D-1. 로봇·카메라 연결은 Piper Studio를 통한다

**확정**
판단·경로 계산(M1·M2·M5·M3의 계산 부분)은 설계서 구조대로 직접 구현한다. 모터에 신호를 보내는 마지막 단계와 카메라 장치 접근은 Piper Studio를 통한다. 설계서의 `piper_driver`, `command_gate`, `camera_top`, `camera_color_setup`, `color_normalizer`, `lighting_monitor` 자리는 Piper Studio 기능으로 대체하고, 연동 코드는 `aple/hal/piper_studio.py` 에 모은다.

**검토 과정**
1. 로봇이 움직이는 과정을 네 단계로 나누었다: ① 판단(어떤 사과를 어디로) → ② 경로 계산(IK) → ③ 안전 검사 → ④ CAN 전송
2. 설계서는 ①~④를 모두 ROS 2 노드로 직접 만든다
3. Piper Studio 저장소를 확인한 결과, ③④는 로봇 데몬(robotd)에 이미 구현되어 있다 — 안전 필터 `safety.filter_goal`(관절 범위·변화율·바닥 충돌·데드맨)과 CAN 전송
4. 로봇 데몬은 CAN 버스를 **단독으로 점유**한다. 카메라 장치도 한 프로그램만 열 수 있다
5. 따라서 ④를 자체 드라이버가 잡으면 Piper Studio 전체가 로봇·카메라에 접근할 수 없다

**근거**
자체 드라이버를 택하면 다음 Piper Studio 기능을 모두 잃고, 대회 기간 안에 다시 만들어야 한다.

| 잃는 기능 | 본 시스템에서의 용도 |
|---|---|
| 텔레오퍼레이션 녹화 | ACT 학습 데이터 수집 |
| ACT 학습·추론 | D-2의 ACT 경로 |
| 회색 카드 보정 · 조명 감시 | M2 색 기준 고정 (설계서 C-3) |
| YOLO 캡처·라벨·학습 루프 | M1 검출 모델 |
| 깊이 배경 마스크 | ACT 입력의 배경 제거 |

또한 설계서 규모(노드 34개, 인터페이스 46종, 시뮬레이터 연동)는 준비 기간 4일과 현장 약 21시간 안에 완성하기 어렵고, 설계서 자체도 ROS 2 배포판·시뮬레이터·IK 라이브러리를 미정으로 두고 있다.

**영향**
- ①② 단계는 설계서 그대로다. 노드는 함수·클래스로, 토픽은 `schema.py` 데이터 클래스로 대응한다
- 대회 후 ROS 2로 옮길 때는 `aple/hal` 만 ROS 2 드라이버로 바꾸면 된다

### D-2. 스크립트 실행을 먼저 완성하고 ACT를 병행한다

**확정**
1일차에는 스크립트 실행(IK 기반)으로 전체 흐름(판정 → 적재 → 기록)을 로봇 위에서 끝까지 연결한다. ACT는 데이터 수집과 야간 학습을 병행한다. 2일차 04:00에 두 경로의 자동 루프 성공률을 비교해 시연 방식을 정한다. Phase B(감속 하강 → 놓기 → 후퇴)는 실행 방식과 관계없이 항상 스크립트로 수행한다.

**검토 과정**
1. 설계서는 스크립트 실행을 기본으로, 전략 문서는 ACT를 기본으로 두고 있었다
2. 스크립트 실행에 필요한 요소가 Piper Studio에 있는지 확인했다
   - 역기구학: `robot/piper_robot/kinematics.py` 의 `ik`, `jacobian`
   - 관절 목표 전달: 웹 조그 세션 (`backend/app/services/jog.py`) → 공유 메모리 → 로봇 데몬
   - 말단 자세 조그(`endpose.py`)는 팔 내부 IK를 쓰므로 관절 안전 필터를 거치지 않는다 → **IK를 직접 풀어 관절 목표로 보내는 경로**를 쓴다
3. 두 방식은 장단점이 반대이므로 하나를 고르지 않고 함께 쓰는 쪽이 위험이 작다고 판단했다

**근거**

| 경로 | 장점 | 약점 |
|---|---|---|
| 스크립트 | 데이터 불필요, 결정론적, 즉시 시험 가능 | 카메라–로봇 캘리브레이션 오차에 취약 |
| ACT | 시각 피드백으로 위치 오차 보정, 대회 취지(모방학습) | 데이터 수집·학습 시간, 조건 변화에 취약 |

- 스크립트로 먼저 완성하면 판정·배치·기록이 ACT 학습 완료를 기다리지 않고 검증된다
- 두 경로의 성공률 비교 결과가 그대로 발표의 정량 근거가 된다
- Phase B를 항상 스크립트로 두는 것은 설계서의 2단계 실행 원칙을 따른 것이다. 낙하 높이와 하강 속도를 **보장**해야 손상 방지 근거가 성립한다

**영향**
- 역할 A: 카메라–로봇 캘리브레이션(체커보드 / ChArUco) 절차를 대회 전에 연습한다
- 역할 B: 조그 세션 경로로 연속 궤적 실행이 되는지 시뮬레이션에서 검증한다 (4절 체크리스트)

### D-3. 등급 판정은 D435 탑뷰만 쓴다

**확정**
착색비율과 등급 판정에는 D435 탑뷰만 쓴다. 정면·손목 웹캠은 ACT 입력으로만 추가한다. 탑뷰의 관측 한계는 애매한 사과만 집어서 다면 검사하는 방식(모드 B·적응형)으로 보완한다.

**검토 과정**
1. 탑뷰 단일 뷰는 사과 표면의 약 25%만 관측한다(cosθ ≥ 0.5, 구 표면 균일 표본 계산). 이를 보완하기 위해 처음에는 정면 웹캠을 판정에 합산하는 안(관측 46%)을 검토했다
2. 카메라를 함께 연결하는 것 자체는 문제가 없다. Piper Studio는 RealSense(rsd)와 웹캠(camerad)을 동시에 운용하도록 설계되어 있고, 판정은 팔이 정지한 시점의 스냅샷 한 장으로 하므로 시간 동기화도 필요 없다
3. 문제는 **색 기준**이다. Piper Studio 회색 카드 보정은 카메라마다 회색 한 점을 맞춰 색을 **고정**하는 기능이다. 센서가 다른 두 카메라의 빨강·노랑 반응까지 **서로 맞춰** 주지는 않는다. 카메라 간 색 맞춤은 Piper Studio 개발팀도 미구현 항목으로 남겨 두었다
4. 두 카메라를 합산하려면 카메라별 임계값, 카메라 간 색 변환, 정면 영상의 사과 위치 대응을 새로 만들어야 하고, 대회 웹캠은 기종을 미리 알 수 없어 사전 검증도 불가능하다
5. 따라서 판정은 한 카메라로 고정하고, 관측 범위는 **같은 D435로** 여러 면을 찍는 모드 B로 넓히는 설계서 방식이 더 타당하다고 결론지었다

**근거 — 모드 B 커버리지 (구 표면 균일 표본 100만 점 계산, 핑거 가림 미반영)**

| 구성 | 커버리지 |
|---|---|
| 탑뷰 1장 (모드 A) | 25% |
| + 툴 60° 기울임, J6 90°씩 4회 | 70% |
| **+ 툴 수평(90°), J6 90°씩 4회** | **91%** |

툴을 수평으로 눕히면 J6 회전축이 수평이 되어 사과가 굴러가듯 돌고, 트레이에 닿아 있던 아랫면까지 관측된다.

**영향**
- 회색 카드 보정은 카메라 3대 모두 그대로 사용한다. D435는 판정 색 고정용, 웹캠은 ACT 입력 색 고정용이다
- 탑뷰 판정의 불확실성은 착색비율을 구간(`ratio_lo`, `ratio_hi`)으로 다뤄 표현한다. 구간이 기준선에 걸친 사과만 모드 B로 보낸다(적응형)
- 모드 B는 사과 1개당 약 6~8초가 추가되고, 툴을 눕힐 때 미끄러짐 위험이 있다. 기본 흐름 완성 후 확장 과제로 붙인다
- 설정: `configs/varieties.yaml` 의 `judge_camera: top`, `views.tilt_rad: 1.5708`

### D-4. 모듈 번호는 역할 분담을 따른다

**확정**
M2 = 색상 품질 판별(역할 C), M3 = 로봇 실행·손상 방지(역할 B·D).

**검토 과정**
같은 Notion 페이지 안에서 상단 「역할 분담」·「초안」은 모듈 2 = 색상, 모듈 3 = 로봇인데, 본문 「모듈별 아키텍처 설계서」는 M2 = 로봇 실행(기존 모듈 2), M3 = 색상(기존 모듈 3)으로 반대였다. 본문은 이전 기획서 번호를 따른 것으로 보인다.

**근거**
역할 분담이 가장 최근에 확정된 기준이고 담당자와 직접 연결되므로, 이를 기준으로 통일한다. 코드 폴더 이름은 번호 대신 기능 이름(`inspection`, `execution`)을 써서 번호 혼동이 코드로 번지지 않게 했다.

## 2. 모듈 대응

| 모듈 | 설계서 패키지 | 구현 | 담당 |
|---|---|---|---|
| M0 공통 인프라 | apple_sorter_hal, apple_sorter_sim | `aple/hal` (Piper Studio 연동, 캘리브레이션) | 역할 A·B |
| M1 인식 · 3D 위치 | apple_sorter_perception | `aple/perception` | 역할 A |
| M2 색상 품질 판별 | apple_sorter_inspection | `aple/inspection` | 역할 C |
| M3 로봇 실행 · 손상 방지 | apple_sorter_execution | `aple/execution` | 역할 B·D |
| M4 공정 관리 | apple_sorter_orchestration, apple_sorter_eval | `aple/orchestration` | 통합 담당 |
| M5 판단 · 계획 | apple_sorter_planning | `aple/planning` | 역할 B·D·A |

## 3. 노드 대응과 우선순위

**필수** = 기본 과제 완주에 반드시 필요 · **확장** = 고도화 과제 · **대체** = Piper Studio 기능 사용 · **대회 후** = 발표 시 확장 설계로 제시

| 설계서 노드 | 구현 위치 | 등급 |
|---|---|---|
| piper_driver, command_gate | Piper Studio 로봇 데몬 (`hal/piper_studio.py`) | 대체 |
| camera_top | Piper Studio rsd (`hal/piper_studio.py` `snapshot`) | 대체 |
| camera_color_setup, color_normalizer, lighting_monitor | Piper Studio 회색 카드 보정 · 조명 감시 (Piper Studio 밖 실험용: `inspection/color.py`) | 대체 — ✅ 소프트웨어판 구현·검증 |
| (카드 보정·조명 상태 공급자) | `inspection/card_monitor.py` `CardMonitor` — 원본 스냅샷에 카드 게인 적용(`capture`) + 같은 프레임 기준 조명 상태(`lighting_status_provider`). `color.card_json` 의 card.json 사용 | 필수 — ✅ 구현·검증 (합성) |
| (레퍼런스 색 패치 조건부 보정, S3-5) | `inspection/ref_patch.py` — 적·황·백 + 회색 카드 기준 색으로 3×3 선형 보정. `CardMonitor` 가 조명이 틀어졌을 때만 적용 → 맞으면 CORRECTING 으로 판정 계속, 아니면 FAULT. 기준값은 `color_session patches` → `color.patch_json` | 확장 — ✅ 구현·검증 (합성) |
| calibration_node | `hal/calibration.py` | 필수 |
| workcell_config_server | `configs/*.yaml` | 필수 |
| scene_capture_server | `hal/piper_studio.py` `snapshot` | 필수 |
| apple_detector_yolo, detection_postprocess | `perception/detector.py` | 필수 |
| apple_pose_estimator, apple_tracker | `perception/pose_estimator.py` | 필수 (tracker 단순형) |
| box_occupancy_estimator | `perception/box_occupancy.py` `estimate` | 필수 |
| placement_verifier | `perception/box_occupancy.py` `verify_placement` | 확장 |
| apple_crop_extractor, color_segmentation_node, surface_area_corrector | `inspection/coloration.py` `color_ratio` | 필수 — ✅ 구현·검증 |
| surface_map_accumulator | `inspection/coloration.py` `accumulate_views` | 확장 (모드 B) — ✅ 구현·검증 |
| (모드 B 자세 계산) | `inspection/inhand.py` — 로봇 자세 → 카메라에서 본 사과 방향·중심, 검사 각도별 예상 커버리지 | 확장 (모드 B) — ✅ 구현·검증 |
| (모드 B 촬영 공급자) | `inspection/in_hand_provider.py` `InHandCaptureProvider` — 스냅샷 + 로봇 자세 → `InHandCapture`. 손에 든 사과 마스크(핑거 제외), 깊이 결측 시 재촬영. `PiperStudioMotionTransport(in_hand_capture_provider=...)` 에 꽂고 검사 전 `begin(apple, apple_in_tcp)` | 확장 (모드 B) — ✅ 구현·검증 (합성) |
| grade_classifier | `inspection/grade_classifier.py` | 필수 — ✅ 구현·검증 |
| (M2 진입점) | `inspection/pipeline.py` `inspect_scene` — 스냅샷 1장의 사과 전부 판정, scene_id 불일치 거부 | 필수 — ✅ 구현·검증 |
| top_view_estimator | 탑뷰–전표면 오차를 실측해 `ratio_lo/hi` 폭으로 사용 | 대회 후 (학습형 모델) |
| grasp_profile_selector | `planning/grasp_profile.py` | 필수 |
| task_planner | `planning/task_planner.py` | 필수 (단순형) |
| placement_planner | `planning/placement_planner.py` | 필수 |
| motion_executor, scripted_backend, soft_landing | `execution/motion_executor.py` | 필수 |
| act_backend | `hal/piper_studio.py` `run_policy` | 필수 (병행) |
| inspect_skill | `execution/motion_executor.py` `inspect` | 확장 (모드 B) |
| 미끄러짐 복구 의도 | `execution/pick_place_pipeline.py` `regrade_with_slip_decision` | 확장 (모드 B) — `InHandSlip` 만 잡아 `FailureCode.SLIP` + `RecoveryPolicy` 결정을 반환. 재파지 동작은 하지 않는다. 검사는 뷰마다 촬영 직후 실행(`inspect_held_apple`의 `view_check`) — 미끄러지면 남은 회전 중단, 깊이 점 부족은 같은 자세에서 1회 재촬영 후에도 확인 안 되면 미끄러짐 처리 |
| 미끄러짐 복구 동작 | `execution/pick_place_pipeline.py` `recover_from_slip` · `run_mode_b_item` | 확장 (모드 B) — 전부 저속(`execution.recovery.speed_scale`), 공중에서 그리퍼 열기 금지. 1회차 `RETRY_GRASP_ONCE`: 원래 트레이 자리에 낙하 ≤ `max_drop_m` 로 내려놓고 `rescan=True` (탑뷰 재촬영·재파지, 모드 B 결과 폐기, 탑뷰 등급 유지). 2회차 `PLACE_REJECT_AND_ALERT`: 리젝트 상자에 적재 후 알림·계속. 그리퍼가 완전히 닫힘(`lost_grip_width_m` 이하)이면 이동 없이 `SAFE_STOP_AND_ALERT` |
| gripper_force_controller | `execution/gripper.py` | 확장 (FSR/힘 텔레메트리 사용 시) |
| cycle_manager | `orchestration/cycle_manager.py` | 필수 (단순형) |
| (M4 장면 실행기) | `orchestration/scene_runner.py` `SceneRunner` — CycleManager 의 plan·execute·records 를 각 역할 모듈로 채움 (조명 FAULT 면 실행 안 함 → RECALIBRATE_LIGHT). 실행 파일 `tools/run_cycle.py` (`--replay` 드라이런 · `--live` 판정·계획 · `--execute` 는 이슈 #24 부품이 들어온 뒤) | 필수 — ✅ 구현·검증 (합성, M1 은 #13) |
| metrics_logger | `orchestration/metrics_logger.py` | 필수 |
| grasp_planner (yaw 선택) | — | 확장 (사과 간격이 좁을 때) |
| arm_occlusion_masker | 스냅샷 전 팔을 scan 자세로 보내는 규칙으로 대체 | 대회 후 |
| pipeline_scheduler | — | 대회 후 |
| sim_bridge | — | 대회 후 |
| reject_curve_evaluator | `inspection/evaluation.py`, `tools/color_session.py curve` | 확장 — ✅ 구현·검증 |
| report_generator | 노트북 분석 | 확장 |

`aple/orchestration/cycle_manager.py` 는 하드웨어 독립형 사이클 실행기다. 캡처, 장면 계획(인식·판정 포함), 실행, `CycleRecord` 생성은 콜백으로 주입하며, 드라이런은 별도 replay/synthetic 캡처 공급자를 요구하고 실행·하드웨어 리셋 콜백을 호출하지 않는다. `orchestration.logging.csv_dir` 가 설정되어 있으면 `cycles.csv` 에 기록한다. 실제 Piper 캡처/FK 공급자 및 복구 정책과 콜백의 통합은 팀 검토 후 연결해야 한다.

`aple/orchestration/recovery.py` 는 Notion M4.5 예외 규칙에 따른 복구 **의도**만 반환한다. 실제 재시도·재촬영·안전 복귀를 수행하지 않으므로, 실행에 연결하기 전에 카운터의 사이클 범위와 하드웨어별 안전 동작을 팀이 검토해야 한다. `GATE_VIOLATION`·E-stop은 즉시 `STOP_AND_IDLE`, 미정의 실패는 `SAFE_STOP_AND_ALERT`로 닫힌다. 미끄러짐 2회차는 정지가 아니라 `PLACE_REJECT_AND_ALERT`(리젝트 적재 후 계속)이고, 그리퍼가 이미 닫힌 미끄러짐(`gripper_lost=True`)만 `SAFE_STOP_AND_ALERT`다.

`aple/orchestration/scene_runner.py` 는 PR #25 구조를 바탕으로 단계 콜백을 연결한다.
조명 FAULT는 공용 `is_lighting_fault`로 확인해 실행을 막고, `execute_scene`에 검사 파라미터와
backend를 전달한다. 첫 미끄러짐 후 `capture_for_retry`로 새 탑뷰·위치·점유를 받아 기존 탑뷰
등급을 유지한 채 1회 재파지한다. 재촬영 후 이전 장면의 나머지 작업은 다음 사이클에서 새로
계획한다. 실행 실패 뒤에는 자동 재실행하지 않는다. 최종 등급·적재 대상 및 실패 phase를
기록에 반영한다. 공급자 조립과 PR 요청 대조는 [pr_review_audit.md](pr_review_audit.md) 참조.
이 연결은 실기 검증 완료를 뜻하지 않으며 역할 A/C 공급자·실측 FK·안전 설정은 별도 필요하다.

## 4. Piper Studio 연동 시 확인할 것

대회 전 시뮬레이션에서 검증한다.

- [x] 스냅샷 경로 조사 — 공유 메모리 세그먼트, 깊이는 비정렬·8비트 인코딩 → `aple/hal/depth_align.py` 로 해결, 상세는 [piper_studio_integration.md](piper_studio_integration.md)
- [ ] 스냅샷 실기: `piper_shm.Subscriber` 로 컬러·깊이 세그먼트 동시 읽기
- [ ] 관절 목표 전달: 웹 조그 세션 경로로 연속 궤적 실행 가능 여부 (재발행 주기, 데드맨)
- [ ] `kinematics.ik` 로 목표 자세 → 관절각
- [ ] 고정 effort 1000에서 파지·과실 손상·미끄러짐 여부 실기 검증 (FSR 피드백은 필수 경로 아님)
- [ ] ACT 실행 중 그리퍼 출력을 파지 프로파일 값으로 바꿀 수 있는 지점
- [ ] 스크립트 제어 중 녹화 동작 여부 (스크립트 시연 → ACT 학습 데이터)

Piper Web v0.5.9의 카메라 JPEG 프리뷰와 raw 조명 측정값은 HAL에서 개별 조회할 수 있다.
다만 프리뷰는 동기화된 `SceneSnapshot`이 아니며, 이 API만으로는 정렬 깊이와 내참수를
얻을 수 없다. `snapshot()`은 정렬 컬러·깊이·내참수를 제공하는 프레임 provider를 주입해야
하며, 누락 시 실패를 명시한다. `LightingStatus`도 보정-aware provider를 주입해야 하고,
raw 조명 측정값만으로 보정 상태를 추정하지 않는다.

역할 B의 `PiperTrajectoryMotionAdapter`는 좌표를 m/rad, 자세를 xyzw 단위 쿼터니언으로
받고, 주입된 IK·구간 안전 검사·속도 지정 가능한 전송 계층으로 Pick/Place와 인핸드
검사 순서를 실행한다. Piper Web v0.5.9 HAL은 조그 세션 하나 안에서 관절 목표들을
순서대로 보내고 각 목표 도달을 확인하는 `send_joint_trajectory()`를 제공한다. 그러나
호출별 속도와 TCP 정지 자세는 제공하지 않으며, 실제 연속 궤적 동작도 시뮬레이션에서
검증해야 한다. 따라서 이 API만으로 `PiperTrajectoryMotionAdapter`의 속도 계약이나
Phase B 감속을 보장할 수 없다. 실제 연결에는 속도·정지 계약을 제공하는 Piper 측 전송
계층과 카메라의 정렬 컬러·깊이·내참수 캡처 provider가 필요하다. HAL의
`PiperStudioMotionTransport` 는 Piper Web v0.5.9에서 속도 명령이 없을 때
joint waypoint 를 더 잘게 쪼개 감속을 근사하지만, 이건 실증된 속도 계약이 아니라
시뮬레이션/실기 확인 전의 보조 경로다. 고정 effort 파지 완료는 센서 측정이 아니라
그리퍼 명령 승인에 근거한 가정이다.

모드 B의 각 캡처는 원래 장면보다 큰 고유 `scene_id`, `Source.IN_HAND`, `CaptureMode.FULL`,
측정 TCP 자세, 마스크와 사과 좌표계 회전을 가져야 한다. 현재 재등급 구현은 이를
기존 `InHandView` 누적기와 판정기로 흘려보내며, 상자 점유는 집기 전 장면의 데이터로
계획한다. 이 장면 간 연결을 공유 계약에 반영할지는 팀 리뷰에서 확인한다.
