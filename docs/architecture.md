# 아키텍처 — 설계서와 구현의 대응

- 논리 설계 기준: 팀 Notion 「아키텍처 및 노드 설계 (1차 10/03)」
- 대회 운영·플랫폼 기준: 전략 종합 문서 II장 5절
- 데이터 정의: [`aple/schema.py`](../aple/schema.py), 다이어그램: [interfaces.md](interfaces.md)

설계서의 모듈·노드 경계와 메시지 정의를 그대로 따르되, 대회 구현은 **Piper Studio 위의 Python 모듈**로 한다.
노드는 함수·클래스로, 토픽은 `schema.py` 의 데이터 클래스로 대응한다.
대회 후 ROS 2로 옮길 때도 경계가 그대로 유지된다.

## 1. 대회 조건에 따른 조정 (제안 — 팀 확인 후 확정)

| # | 항목 | 설계서 | 구현 | 이유 |
|---|---|---|---|---|
| 1 | 로봇·카메라 연결 | ROS 2 자체 드라이버 (piper_driver, command_gate, camera_top) | **Piper Studio 연동** (`aple/hal`) | CAN·카메라 장치는 프로그램 하나만 열 수 있다. 자체 드라이버가 잡으면 Piper Studio 녹화·ACT·회색 카드 보정·YOLO 학습을 못 쓴다 |
| 2 | 실행 방식 | 스크립트 기본, ACT 선택 | **스크립트 먼저 완성 + ACT 병행**, 2일차 04:00 성공률 비교 후 결정. Phase B는 항상 스크립트 | 스크립트는 데이터 없이 바로 전체 흐름을 검증한다. Piper Studio에 IK(`kinematics.ik`)가 있다 |
| 3 | 판정 카메라 | D435 탑뷰 | **D435 탑뷰 유지**, 정면·손목 웹캠은 ACT 입력에만 추가 | 회색 카드 보정은 카메라별 색 고정이지 카메라 간 색 맞춤이 아니다. 판정을 한 카메라로 해야 색 기준이 하나로 유지된다 |
| — | 모듈 번호 | 본문 M2 = 실행, M3 = 색상 | **M2 = 색상, M3 = 실행** | 역할 분담(모듈 2 = 색상, 모듈 3 = 로봇)과 통일 |

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
| camera_color_setup, color_normalizer, lighting_monitor | Piper Studio 회색 카드 보정 · 조명 감시 | 대체 |
| calibration_node | `hal/calibration.py` | 필수 |
| workcell_config_server | `configs/*.yaml` | 필수 |
| scene_capture_server | `hal/piper_studio.py` `snapshot` | 필수 |
| apple_detector_yolo, detection_postprocess | `perception/detector.py` | 필수 |
| apple_pose_estimator, apple_tracker | `perception/pose_estimator.py` | 필수 (tracker 단순형) |
| box_occupancy_estimator | `perception/box_occupancy.py` `estimate` | 필수 |
| placement_verifier | `perception/box_occupancy.py` `verify_placement` | 확장 |
| apple_crop_extractor, color_segmentation_node, surface_area_corrector | `inspection/coloration.py` `color_ratio` | 필수 |
| surface_map_accumulator | `inspection/coloration.py` `accumulate_views` | 확장 (모드 B) |
| grade_classifier | `inspection/grade_classifier.py` | 필수 |
| top_view_estimator | 탑뷰–전표면 오차를 실측해 `ratio_lo/hi` 폭으로 사용 | 대회 후 (학습형 모델) |
| grasp_profile_selector | `planning/grasp_profile.py` | 필수 |
| task_planner | `planning/task_planner.py` | 필수 (단순형) |
| placement_planner | `planning/placement_planner.py` | 필수 |
| motion_executor, scripted_backend, soft_landing | `execution/motion_executor.py` | 필수 |
| act_backend | `hal/piper_studio.py` `run_policy` | 필수 (병행) |
| inspect_skill | `execution/motion_executor.py` `inspect` | 확장 (모드 B) |
| gripper_force_controller | `execution/gripper.py` | 필수 (토크 변경 범위) |
| cycle_manager | `orchestration/cycle_manager.py` | 필수 (단순형) |
| metrics_logger | `orchestration/metrics_logger.py` | 필수 |
| grasp_planner (yaw 선택) | — | 확장 (사과 간격이 좁을 때) |
| arm_occlusion_masker | 스냅샷 전 팔을 scan 자세로 보내는 규칙으로 대체 | 대회 후 |
| pipeline_scheduler | — | 대회 후 |
| sim_bridge | — | 대회 후 |
| report_generator, reject_curve_evaluator | 노트북 분석 | 확장 |

## 4. Piper Studio 연동 시 확인할 것

대회 전 시뮬레이션에서 검증한다.

- [ ] 스냅샷: D435 컬러(보정 적용) + 정렬 깊이를 코드에서 받는 경로
- [ ] 관절 목표 전달: 웹 조그 세션 경로로 연속 궤적 실행 가능 여부 (재발행 주기, 데드맨)
- [ ] `kinematics.ik` 로 목표 자세 → 관절각
- [ ] 그리퍼 토크 1000 고정값 변경 경로
- [ ] ACT 실행 중 그리퍼 출력을 파지 프로파일 값으로 바꿀 수 있는 지점
- [ ] 스크립트 제어 중 녹화 동작 여부 (스크립트 시연 → ACT 학습 데이터)
