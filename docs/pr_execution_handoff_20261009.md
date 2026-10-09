# 역할 B 통합 인수인계·대회 설정 (2026-10-09)

이 변경은 #31 이후 미커밋으로 남아 있던 무작위 배치·중앙 검사·속도 탐색·링 그리퍼·독립 관측 실험과, 10/9 코드 검토의 실행 오류 수정을 함께 묶는다. 기존 실기 구조는 Python/Piper Studio 기반이며 위치 입력은 M1의 `AppleState`, 목적지는 관측한 `BoxOccupancy`와 최종 등급으로 결정한다. 무작위 좌표·질량·마찰·고속 배율은 별도 시뮬레이션 설정에만 있다.

변경별 설명: [코드 검토 개선](review_fixes.md), [속도 탐색](simulation_speed_search.md), [중앙 검사 시뮬레이션](in_hand_simulation.md), [곡면 링 모델](gripper_physics.md).

## 다른 담당자가 확인하거나 수정해야 하는 부분

| 담당 | 연결 변경과 필요한 작업 |
|---|---|
| A / @lsb1017 | 사과 ID를 적재 후 재촬영에서도 유지해야 한다. 남은 대상 ID가 사라지면 다른 사과를 대신 집지 않는다. 재파지 후보에는 전체 관측 사과를 장애물로 전달한다. 실제 캡처는 요청 이후의 새 컬러·깊이 노출과 wall-clock 시각을 반환해야 한다. 가짜 카메라 테스트도 오래된 고정 시각을 재사용하면 안 된다. |
| A / @lsb1017 | 상자는 열=+x, 행=+y, 칸=행×cols+열이다. 적재 전후 `rows/cols/origin_m/cell_pitch_m`이 동일해야 하며 후속 `scene_id`는 증가해야 한다. 새 높이와 `occupied`가 일치해야 검증된다. 무효/오래된 관측은 실행 성공으로 처리하지 않는다. 상자 원점은 현재 구조에서 실측 설정으로 공급하며 상자 자동 위치 탐색을 추가한 변경은 아니다. |
| C / @pikachu515 | 공통 검사 자세가 `Ry(tilt) @ R_tool_axis(angle)`로 바뀌었다. 가림 마스크·사과 좌표계 회전·같은 IK/FK/TCP 정의를 공유해야 한다. 촬영은 목표 자세 도착→안정화→새 영상 순서이며 고정 Hz가 아니다. 현재 실기 기본 `inspection.policy`는 NEVER다. ADAPTIVE/ALWAYS, 검사 중심·각도·대기시간은 현장에서 확정해야 한다. |
| C / @pikachu515 | 링 구멍으로 드러난 면도 실제 깊이·핑거 제외 마스크·구면 누적으로 판정한다. 속도 시험의 운반 성공을 착색 정확도로 해석하면 안 된다. 꼭지 가중치·색 임계·2차 판정 정책을 임의로 바꾸지 않았다. 실제 관측률과 색 기준을 다시 검증한다. |
| D / @uhdahmrok | 수제 핑거 장착 후 TCP, 접촉면 개폐 폭, 패드 압축/안정화, 보수적 외형·충돌 구, 힘·손상·슬립 한계를 실측해야 한다. 외경 78.664 mm·내경 48.92 mm와 사진 점 거리 48.144/48.464/71.76/79.76 mm를 TCP 오프셋으로 자동 해석하지 않는다. |
| D / @uhdahmrok | 기본 fixed_effort의 `grasp_force_set_n`은 NaN이다. Studio 설정 토크와 과실 접촉력 N은 다른 값이다. 손상 지표·CSV 소비 코드는 유한 값만 사용하고 NaN을 0 N으로 대체하지 않는다. `planned_drop_m`은 계획값이며 `settle_error_m`은 후속 영상 기반 검증값이다. |
| B / @dbrua3311 및 별도 실행 어댑터 사용자 | `PiperStudioMotionTransport` 이동에는 `MotionLimits`가 필요하다. 기존 `max_joint_step_rad`만 넘기는 생성은 이동을 거부한다. `holding_verifier` 없이 명령 승인을 파지 증거로 사용하지 않는다. 기본 실기는 `build_live_runtime`로 조립하는 것을 기준으로 한다. 별도 HAL도 리프트 후 파지 증거와 속도/중단 계약을 구현해야 한다. |
| 공통 M4 | 직접 `SceneRunner`를 조립하는 경로는 `capture_for_verification`, `capture_for_retry`, `before_execute`를 연결해야 새 실기 검증 루프를 사용한다. `tools/run_cycle.py --live --execute`는 이 조립을 제공한다. 실기 실패 뒤 다음 사이클에서 자동 재시작하지 않으며 원인을 확인한 후 새 실행을 시작한다. |
| 기록/발표 | 최초 캡처 시간을 포함하고 장면 공통 시간을 사과 행에 나눠 기록한다. 단계별 측정과 전체 `t_total_s`를 구별한다. 외부 CSV scene_id는 최초 장면이고 내부 실행 항목은 재촬영된 최신 장면을 사용할 수 있다. 복구/재시도 비용은 전체 시간으로 비교한다. |

공유 `aple/schema.py`와 자동 생성 `docs/interfaces.md`는 변경하지 않았다. 기존 입력 데이터 클래스 자체를 팀원이 일괄 교체할 필요는 없다.

## 열려 있는 팀원 PR과 병합 주의사항

2026-10-09 main `7217ca8` 기준이다. 아래 PR은 조회 당시 아직 병합되지 않았다. 팀원 PR 코드를 이번 브랜치에 몰래 복사하거나 병합하지 않았다.

- [#32](https://github.com/pikachu515/aple-pac2026/pull/32): 드라이런 예약/기록 테스트와 겹친다. 이번 변경은 `plan_scene`의 복사본에서 임시 예약하고 실제 실행 전에 확정한다. records 단계에서 해제하는 방식과 역할을 중복시키지 않는다. #32의 force 전달 테스트는 **fixed_effort에서 NaN**, calibrated_force에서 계획 N 전달로 나누어야 한다. 인위적으로 profile의 N만 바꿔 fixed_effort에서도 실측값처럼 기록하면 안 된다. `planned_drop_m` 전달 검증은 유지한다.
- [#35](https://github.com/pikachu515/aple-pac2026/pull/35): `simulation.py`, `scene_runner.py`, `live_camera.yaml`, `configs/simulation/piper.yaml`, `run_cycle.py` 등이 겹친다. 실물 사진 시뮬레이션 옵션·깊이 보정과 이번 실기 조립/최신 프레임/파지 확인 기능을 모두 보존해 병합하고 재검증한다. 특히 #35는 M1 진입점에서 `DepthCorrection.snapshot()`을 각각 적용한다. **비항등 깊이 보정을 쓸 경우, M2 탑뷰·InHandCaptureProvider의 마스크·VisualHoldingVerifier·슬립/재등급에도 같은 보정 깊이를 사용하도록 A/C/B가 통합해야 한다.** 원본 저장은 보정 전으로 유지하며, M1에서 이미 보정하는 경로에 또 적용하면 이중 보정된다. 모듈별 1회 적용 또는 공통 보정 스냅샷 경계를 합의하고 왕복/파지/검사/상자 검증을 다시 돌린다.
- [#36](https://github.com/pikachu515/aple-pac2026/pull/36): #35 위에 쌓인 카메라 재보정 PR이다. Tare 후 새 스트림·Piper 공유 메모리에서 거리를 다시 확인하고 소프트웨어 보정과 중복 적용하지 않는다. 제공 카메라가 바뀌면 10/8 보정 결과를 그대로 쓰지 않는다. 노출/해상도/깊이-컬러 정렬과 가장자리 오차를 다시 확인한다.

이 목록은 수정 필요사항을 알리는 인수인계이며 #32/#35/#36을 해결/완료로 표시하는 것은 아니다.

## 대회 현장에서 필수로 입력·확인할 값

단위는 SI(m, s, rad, N), `_mm` 키만 mm다. 값을 사진·제품 기본 스트로크·시뮬레이션 가정에서 임의로 채우지 않는다.

| 파일·키 | 입력/확인 내용 | 담당 |
|---|---|---|
| `workcell.T_base_camera` | 실제 설치의 4×4 강체 변환. 카메라를 옮기거나 기울이면 재보정 | A |
| `workcell.tcp_offset_m` | 장착 후 플랜지 좌표계 TCP 이동. IK/FK가 같은 변환 사용 | A·D·B |
| `workcell.tray` | 실제 중심/크기/벽 높이. 축 정렬 가정 확인 | A |
| `workcell.boxes.*` | 실제 grid, 원점, pitch, 벽 높이. UPPER/MIDDLE/REJECT 식별 확인 | A |
| `workcell.poses.home_joint_rad`, `scan_joint_rad` | 충돌 없이 도달하고 과실/상자를 가리지 않는 6관절 자세 | B·A |
| `safety.joint_limits_rad`, `piper_web.joint_limits_deg` | 같은 실제 관절 범위를 rad/deg로 일치시킴. URDF·설치된 Studio와 대조 | B |
| `safety.workspace_aabb_m` | 실제 작업영역 3축 상하한 | B·A |
| `safety.segment_validation.*` | 샘플 간격(rad), 바닥 z(m), 여유(m), 벽 두께(m), 장애물 AABB, 비인접 링크 쌍, payload 검사 링크, 리프트 초기 접촉 오차(m) | B·D·A |
| `execution.motion.velocity_limits_rad_s` | 6관절 속도 한계 | B |
| `execution.motion.acceleration_limits_rad_s2` | 6관절 가속도 한계 | B |
| `execution.motion.command_period_s` | Studio 실제 처리/추종 주기 | B |
| `execution.motion.tracking_tolerance_rad` | 추종 오차 허용치. 충돌 여유와 함께 검증 | B |
| `execution.motion.velocity_tolerance_rad_s` | 측정 속도 오차·지연 허용치 | B |
| `execution.motion.capture_joint_tolerance_rad` | 촬영 전후 정지 확인 허용치 | B·A |
| `execution.gripper.closing_compression_m` | 실제 과실/패드 접촉의 안쪽 닫힘 목표량 | D·B |
| `execution.gripper.contact_settle_s`, `contact_width_tolerance_m`, `contact_stability_m` | 접촉 폭·안정화 시간·안정 판정 변동량 | D·B |
| `piper_web.gripper_width_range_m` | 장착 후 normalized 0/100에 대응하는 실제 접촉면 간격 | D·B |
| `gripper_attachment.installed`, `assembled.*` | 장착했다면 true와 전체 장착 기하. `tcp_axes_match_flange`는 현재 이동-only TCP 변환의 전제. false면 회전 변환을 구현해야 하며 실행 거부를 없애면 안 됨 | D·B |
| `execution.gripper.fixed_effort` | Studio 저장 토크와 일치. 1000은 1 N·m 설정값, 과실 접촉력 1 N이 아님 | D·B |
| `execution.placement_verify.height_tol_m` | 빈 칸 반복 촬영의 깊이 흔들림과 과실 크기 오차를 고려한 안착 허용치 | A·B |
| `piper_web.base_url`, `iface`, `enabled`, `robotd_safety_confirmed` | 실제 follower/주소, Studio 보호·데드맨 확인 후 활성화 | B |
| `piper_web.request_timeout_s`, `poll_interval_s`, `arrival_tolerance_rad`, `gripper_arrival_tolerance` | HTTP 지연과 실제 추종에 맞춤. 빠른 요청만으로 물리 속도를 높이지 않음 | B |
| `live_camera.serial`, `depth_calib`, `near_mm/far_mm`, `frames`, `max_skew_ms`, `timeout_s`, `max_frame_age_s` | 실제 D435/컬러·깊이 정렬/인코딩 범위/시각 확인. serial null은 자동 선택이며 필수 누락이 아님 | A |
| `color.card_json`, 품종별 hue/chroma/delta, `inspection.top_view_bias/top_view_error` | 실제 조명·품종·사과 방향으로 확정. color.patch_json은 대회 기본 미사용 | C |
| `inspection.policy`, `box_boundary`, `views.*` | 미션의 등급 경계, 2차 검사 대상, 실제 가능한 중심·틸트·각도·촬영 대기. center_m null은 집어 올린 위치 사용 | C·B |
| `inspection.slip_check.*`, `in_hand_mask.*` | 정지 사과/미끄러지지 않은 회전의 깊이·자세 오차, 링과 과실 분리 허용치 | C·A·B |
| `grasp_planner.*`, `size_bins.*`, `workcell.placement.*`, `execution.soft_landing.*`, `execution.recovery.*`, `safety.speed_scale_max` | 실제 외형·대상 크기·낙하 높이·마찰/손상·운반/복구 속도와 마지막 하강 확인 | A·B·D |
| #35/#36 병합 후 `live_camera.depth_correction.*`, `depth_check.card_mm/max_error` | 해당 카메라의 재보정 뒤 검증. 모든 인식/판정/파지 확인에서 동일 깊이 기준 | A·C·B |

`execution.default_backend: scripted`, `gripper.control_mode: fixed_effort`가 이번 실제 Studio 조립 경로다. ACT·calibrated_force는 설정값만 바꿔 연결되는 기능이 아니다. FSR 모드를 사용하려면 별도 피드백 전송과 effort→force 보정표, 마찰·손상 한계를 추가한다.

`self_collision_pairs`, `payload_self_links`, `obstacle_aabbs_m`의 빈 목록은 해당 보호 범위를 사람이 확인한 경우에만 명시한다. null을 자동으로 빈 목록으로 바꾸지 않는다. 유한 샘플링·구형 근사는 연속 충돌 안전 인증이 아니며 Studio 보호는 계속 필요하다.

### 값 목록·점검 명령

```bash
python3 tools/field_values.py list --all
python3 tools/field_values.py list --owner B
# 로봇/카메라를 연결하거나 이동하지 않는 실기 조립 설정 점검
python3 -c 'from aple.config import load_config; from aple.hal.live_runtime import missing_for_live_execution; print("\n".join(missing_for_live_execution(load_config())))'
```

`field_values`는 주석/null 표시를 찾아주는 도구다. 아래 자동 목록의 '필수'는 표시 분류이며, 장착 여부·백엔드·카메라 사용 조건은 별도로 판단한다. 목록 개수 0만으로 실기 준비가 끝났다고 선언하지 않는다. 실제 실행은 `missing_for_live_execution`과 Studio/영상/추종/안착 검증을 모두 통과해야 한다. 설정은 `field_values set 키 값 --note 설명`으로 측정 근거를 남기고 대회 중에도 브랜치→PR로 공유한다.

## 결과와 재현 범위

PR 준비 최종 pytest: **636 passed, 1 skipped, 1 warning in 241.89s (0:04:01)**. `tests/`와 보관 실험의 테스트를 포함하며 인터페이스 생성 검사와 `git diff --check`도 통과했다. 로컬 검증은 Python 3.10 + 설치된 Piper/MuJoCo이며 CI는 Python 3.12다. 경고는 구형 pytest의 `pythonpath` 설정 인식 문제 1개다.

| 시험 | 결과 | 해석 |
|---|---|---|
| 기존 속도 배율 16, seed 42/7/13/99/2026 | 5개 장면·15개 사과 운반 성공, slip 0; 장면 53.382/53.098/58.050/55.824/54.100 s | 시뮬레이션의 시험 범위. 모두 불확실성에 따른 REJECT이며 착색 정확도 100%가 아님 |
| 기존 배율 32, seed 42 | 53.382 s, 배율 16과 동일 | 명령 보간값이 포화한 것. 제조사 최고속도/실기 승인 속도가 아님 |
| 개선 후 짧은 물리 재확인, seed 42 | 1/1 운반 성공, 물리 약 17.9 s, 관측률 0.258, REJECT | 새 파지/충돌 연결의 한 사과 회귀 확인. 전체 장면/실물 성공률 증명이 아님 |
| 자유 XYZ 보관 실험 | 11뷰·79.8% 관측, 1개 운반 성공, 영상 실행 131.5 s | 폐기한 위아래/자유 관측 비교 기록. 현재 기본 옆 회전의 결과가 아님 |
| 대각선 카메라 보관 실험 | 0/30/45/60° 위아래 검사 당시 평균 관측 35.9/35.2/34.8/34.4% | 카메라만 바꾼 전체 실기 성공률이 아님. 정상 깊이·재보정·가림/결측 검증 필요 |

현재 기본은 **옆 회전**이며 보관 실험을 대회 실행에 연결하지 않는다. 시뮬레이션 빈손/운반/검사 보간값 28.8/8.8/5.6 rad/s, 촬영 전 물리 대기 0.06 s, 압축/마찰/질량 등은 실기 설정으로 복사하지 않는다. 실기 촬영은 현재 settle 기본 0.3 s이고 이후 새 노출을 기다리므로 실제 촬영 시간은 이를 초과할 수 있다. 같은 검사 자세에서 J6만으로 360° 연속 회전을 보장하지 않는다.

### 재실행과 영상

Piper 수학/물리 의존성이 있는 Python 환경에서 실행한다. 로컬 작성자의 환경은 `/home/seo/.venvs/piper-daemons/`이며 이 가상환경은 저장소에 포함하지 않는다.

```bash
python3 tools/run_cycle.py --simulate --trace logs/simulation/trace.json
MUJOCO_GL=egl python3 tools/run_in_hand_simulation.py --quick --optimize --seed 42 --output logs/review-replay
MUJOCO_GL=egl python3 tools/run_in_hand_simulation.py --optimize --seed 42 --video --output logs/review-video
# 원래 빠른 후보/전체 바닥 장면
python3 tools/run_in_hand_simulation.py --optimize --seed 42 --optimization-config configs/simulation/in_hand_fast.yaml --scene-config configs/simulation/in_hand_full_floor.yaml
# 실기: ① 드라이런 ② 실측/단계별 검증 후 명시적 실행
python3 tools/run_cycle.py --live --cycles 1 --save data/field/1009_a
python3 tools/run_cycle.py --live --execute --cycles 1
```

영상·실험 로그·실물 사진·가중치는 AGENTS 규칙상 커밋하지 않는다. 기존 결과는 작성자의 `/mnt/extra/aple-pac2026/logs/` 아래에 보존돼 있으며, PR에 보고 수치를 옮겼다. 이 PR은 GitHub에 동영상 파일을 업로드한 첨부가 아니다. 다른 컴퓨터에서는 코드/설정/seed로 재현하거나 팀 공유 드라이브로 매체를 따로 공유한다. 보관 실험의 과거 수치는 당시 소스/그리퍼 조건의 결과이며 현재 코드 재실행에서 같은 수치가 보장되지 않는다.

## 현재 표시된 현장 값 전체 목록

아래는 `tools.field_values.scan()`이 configs 최상위 파일에서 찾은 null/PLACEHOLDER/[임시]/[선택] 항목이다. 링크한 변경 요약 및 위 필수 점검 표와 함께 사용한다. 런타임 기본 상수 전부를 임의로 현장 실측값으로 바꾸라는 뜻은 아니다.

| 파일 | 담당 | 키 | 표시 | 현재 값 | 설정 설명 |
|---|---|---|---|---|---|
| camera_top.yaml | A | `camera_top.serial` | 선택 | `null` | [선택] 기록용 — 실제 장치 선택은 live_camera.serial |
| color.yaml | C | `color.card_json` | null | `null` | tools/color_session.py card 결과 경로 (현장 세션 폴더의 card.json). CardMonitor 가 읽는다 |
| execution.yaml | B | `execution.placement_verify` | PLACEHOLDER | `{height_tol_m: 0.010}` | PLACEHOLDER: 깊이 노이즈 실측 후 교체 |
| execution.yaml | B | `execution.trajectory` | 선택 | `{max_joint_step_rad: 0.05, max_cartesian_step_m: null}` | [선택] null: 기존 관절 보간; m 값: TCP 중간 목표 추가 |
| execution.yaml | B | `execution.motion.velocity_limits_rad_s` | null | `null` | 관절별 6개 |
| execution.yaml | B | `execution.motion.acceleration_limits_rad_s2` | null | `null` |  |
| execution.yaml | B | `execution.motion.command_period_s` | null | `null` |  |
| execution.yaml | B | `execution.motion.tracking_tolerance_rad` | null | `null` |  |
| execution.yaml | B | `execution.motion.velocity_tolerance_rad_s` | null | `null` |  |
| execution.yaml | B | `execution.motion.capture_joint_tolerance_rad` | null | `null` | 촬영 전후 관절 변화 허용치 |
| execution.yaml | B | `execution.recovery.speed_scale` | 임시 | `0.2` | [임시] 복구 이동 속도. safety.speed_scale_max 이하여야 한다 |
| execution.yaml | B | `execution.recovery.lost_grip_width_m` | 임시 | `0.01` | [임시] 측정 폭이 이 값 이하(완전히 닫힘)면 사과를 놓친 것 → 정지. 역할 D 실측(S8-6)과 맞출 것 |
| execution.yaml | B | `execution.gripper.close_speed` | 선택 | `null` | [선택] 코드에서 쓰지 않음 |
| execution.yaml | B | `execution.gripper.closing_compression_m` | null | `null` | 실측 패드 변형/접촉을 반영한 안쪽 닫힘 목표. 임의 값 복사 금지 |
| execution.yaml | B | `execution.gripper.contact_settle_s` | null | `null` |  |
| execution.yaml | B | `execution.gripper.contact_width_tolerance_m` | null | `null` |  |
| execution.yaml | B | `execution.gripper.contact_stability_m` | null | `null` |  |
| execution.yaml | B | `piper_web.iface` | PLACEHOLDER | `"PLACEHOLDER_follower"` | PLACEHOLDER: 시뮬/실행 전 Piper Web 등록 follower 이름으로 교체 |
| execution.yaml | B | `piper_web.camera_id` | PLACEHOLDER | `"PLACEHOLDER_d435"` | PLACEHOLDER: 시뮬/실행 전 실제 D435 camera id로 교체 |
| execution.yaml | B | `piper_web.gripper_width_range_m` | PLACEHOLDER | `[0.0, 0.07]` | PLACEHOLDER: 시뮬/실행 전 실측 [닫힘, 열림] 폭으로 교체 |
| grasp_planner.yaml | A | `grasp_planner.open_margin_m` | 임시 | `0.01` | [임시] 핑거 안쪽 면 ↔ 사과 표면 여유 (사전 개방 폭 = 직경 + 2 × 이 값) |
| grasp_planner.yaml | A | `grasp_planner.finger_thickness_m` | 임시 | `0.012` | [임시] 핑거 + 어댑터 + 패드 두께 (닫힘 방향) |
| grasp_planner.yaml | A | `grasp_planner.finger_width_m` | 임시 | `0.02` | [임시] 핑거 폭 (닫힘 방향에 수직) |
| grasp_planner.yaml | A | `grasp_planner.tool_down_rpy_deg` | 임시 | `[180, 0, 0]` | [임시] TCP 가 아래를 보는 자세 (base_link RPY). 역할 B IK 규약과 맞춘다 |
| gripper_attachment.yaml | D | `gripper_attachment.assembled.tcp_axes_match_flange` | null | `null` | true: 현재 TCP 보정은 이동만 지원. 축이 다르면 회전 보정 필요 |
| gripper_attachment.yaml | D | `gripper_attachment.assembled.tcp_offset_mm` | null | `null` | 플랜지 원점 → 고정 파지 기준점, 플랜지 좌표계 [x,y,z] |
| gripper_attachment.yaml | D | `gripper_attachment.assembled.contact_width_range_mm` | null | `null` |  |
| gripper_attachment.yaml | D | `gripper_attachment.assembled.finger_thickness_mm` | null | `null` | 닫힘 방향의 보수적 외형 두께 (패드 포함) |
| gripper_attachment.yaml | D | `gripper_attachment.assembled.finger_width_mm` | null | `null` | 닫힘 방향에 수직인 보수적 외형 폭 |
| gripper_attachment.yaml | D | `gripper_attachment.assembled.grasp_height_offset_mm` | null | `null` | 사과 중심 기준 파지 높이, 부호 있는 mm |
| gripper_attachment.yaml | D | `gripper_attachment.assembled.collision_spheres_flange_mm` | null | `null` |  |
| live_camera.yaml | A | `live_camera.serial` | 선택 | `null` | [선택] null = 공유 메모리에서 첫 RealSense 자동 선택 |
| perception.yaml | A | `perception.sphere_fit` | 선택 | `{ransac_threshold_m: 0.003, max_fit_residual_m: null, max_radius_disagreement: 0.25}` | [선택] max_fit_residual_m null = 잔차 검사 안 함 |
| perception.yaml | A | `perception.detect.max_height_ratio` | 임시 | `1.3` | [임시] 꼭대기 높이 / 직경 상한 — 넘으면 기둥·팔·상자 벽 (구는 ≈ 1) |
| perception.yaml | A | `perception.detect.min_circularity` | 임시 | `0.5` | [임시] 둥근 정도 4πA/P² 하한 — 가늘고 긴 물체 제외 (반으로 가른 사과 ≈ 0.7) |
| safety.yaml | B | `safety.joint_limits_rad` | null | `null` | 실측 전에는 scripted motion 금지 |
| safety.yaml | B | `safety.workspace_aabb_m` | null | `null` | measured [[x_min,x_max], [y_min,y_max], [z_min,z_max]] |
| safety.yaml | B | `safety.segment_validation.sample_step_rad` | null | `null` |  |
| safety.yaml | B | `safety.segment_validation.floor_z_m` | null | `null` |  |
| safety.yaml | B | `safety.segment_validation.clearance_m` | null | `null` |  |
| safety.yaml | B | `safety.segment_validation.wall_thickness_m` | null | `null` |  |
| safety.yaml | B | `safety.segment_validation.support_contact_tolerance_m` | null | `null` | 리프트 시작 접촉에서만 허용할 바닥/과실 추정 오차 (실측) |
| safety.yaml | B | `safety.segment_validation.obstacle_aabbs_m` | null | `null` | 실측 장애물. 장애물이 없다고 확인한 경우 [] |
| safety.yaml | B | `safety.segment_validation.self_collision_pairs` | null | `null` | 검증할 비인접 링크 이름 쌍 [[link_a, link_b], ...] |
| safety.yaml | B | `safety.segment_validation.payload_self_links` | null | `null` | 그리퍼/사과와 충돌 검증할 링크 이름 목록 |
| size_bins.yaml | D | `size_bins.bins[0]` | 선택 | `{range_mm: [30, 50],  open_width_mm: null, grasp_height_offset_mm: 0}` | [선택] open_width_mm null = 직경 + 여유로 계산 |
| size_bins.yaml | D | `size_bins.bins[1]` | 선택 | `{range_mm: [50, 70],  open_width_mm: null, grasp_height_offset_mm: 0}` | [선택] open_width_mm null = 직경 + 여유로 계산 |
| size_bins.yaml | D | `size_bins.bins[2]` | 선택 | `{range_mm: [70, 85],  open_width_mm: null, grasp_height_offset_mm: 0}` | [선택] open_width_mm null = 직경 + 여유로 계산 |
| size_bins.yaml | D | `size_bins.bins[3]` | 선택 | `{range_mm: [85, 100], open_width_mm: null, grasp_height_offset_mm: 0}` | [선택] open_width_mm null = 직경 + 여유로 계산 |
| size_bins.yaml | D | `size_bins.force` | 선택 | `{safety_factor: 2.0, a_max_m_s2: 4.9, mu: null, f_bruise_n: null}` | [선택] execution.gripper.control_mode: calibrated_force 일 때만 필요 |
| size_bins.yaml | D | `size_bins.gripper_calib` | 선택 | `{effort_to_force_table: null}` | [선택] calibrated_force 일 때만. FSR 보정표 [[effort, force_n], ...] |
| varieties.yaml | C | `varieties.fuji.delta_up` | 임시 | `0.05` | [임시] '상' 판정 여유 — 리젝트 곡선으로 조정 |
| varieties.yaml | C | `varieties.fuji.delta_mid` | 임시 | `0.0` | [임시] |
| varieties.yaml | C | `varieties.fuji.hue_range_deg` | 임시 | `[-40, 55]` | [임시] 착색 색상각 범위 (0 = 적색, 90 = 황색). suggest 로 확정 |
| varieties.yaml | C | `varieties.fuji.min_chroma` | 임시 | `15` | [임시] 무채색(그림자·반사) 제외 |
| varieties.yaml | C | `inspection.shadow` | 임시 | `{max_L: 15, max_fraction: 0.15}` | [임시] L* 15 미만 면적이 15% 넘으면 판정 보류·재촬영 (비율은 고치지 않는다) |
| varieties.yaml | C | `inspection.top_view_error` | 임시 | `0.10` | [임시] 판정 구간 반폭 = 치우침을 뺀 뒤 남는 탑뷰 오차. truth 로 확정 |
| varieties.yaml | C | `inspection.top_view_bias` | 임시 | `0.0` | [임시] 탑뷰 − 전표면 평균 치우침 (+ 면 탑뷰가 높게 봄). 10/7 실측: 꼭지가 위면 평균 +0.12. truth 로 확정 |
| varieties.yaml | C | `inspection.max_view_disagreement` | 임시 | `0.25` | [임시] 모드 B 뷰 간 편차 한계 |
| varieties.yaml | C | `inspection.slip_check.sigma_lateral_m` | 임시 | `0.0015` | [임시] 깊이 관측의 옆 방향 오차 |
| varieties.yaml | C | `inspection.slip_check.sigma_axial_m` | 임시 | `0.003` | [임시] 깊이 방향 오차 (D435 편향 + Piper 8bit 양자화 4mm) |
| varieties.yaml | C | `inspection.slip_check.sigma_pose_m` | 임시 | `0.003` | [임시] 카메라–로봇 보정 + 순기구학 + 파지 측정 오차 |
| varieties.yaml | C | `inspection.in_hand_mask.surface_tol_m` | 임시 | `0.006` | [임시] 2차: 맞춘 구 표면 뒤 6mm·앞 3mm 만 사과. 링·핑거 앞면이 사과에서 8mm 이상 떨어져야 한다 |
| varieties.yaml | C | `inspection.views.center_m` | 선택 | `null` | [선택] base_link 검사 중심. null이면 집어 올린 위치 사용 |
| workcell.yaml | A | `workcell.T_base_camera` | null | `null` | 4x4, eye-to-hand 캘리브레이션 결과 (m) |
| workcell.yaml | A | `workcell.tcp_offset_m` | null | `null` | 핑거 어댑터 반영 |
| workcell.yaml | A | `workcell.tray.center_m` | null | `null` | [x, y, z] 트레이 바닥 윗면 중심 (m), 현장 실측. null 이면 영상 전체에서 검출 |
| workcell.yaml | A | `workcell.boxes.UPPER` | null | `{grid: [2, 3], cell_pitch_m: 0.09, wall_height_m: 0.08, origin_m: null}` |  |
| workcell.yaml | A | `workcell.boxes.MIDDLE` | null | `{grid: [2, 3], cell_pitch_m: 0.09, wall_height_m: 0.08, origin_m: null}` |  |
| workcell.yaml | A | `workcell.boxes.REJECT` | null | `{grid: [2, 3], cell_pitch_m: 0.09, wall_height_m: 0.08, origin_m: null}` |  |
| workcell.yaml | A | `workcell.poses` | null | `{home_joint_rad: null, scan_joint_rad: null}` |  |
| yolo.yaml | A | `yolo` | 선택 | `{weights: null, task: detect, imgsz: 640, conf: 0.4, iou: 0.5, device: 'cuda:0', mask_erode_px: 0, classes: null}` | [선택] weights null = 깊이·모양 검출 (COCO yolo11n + classes [47] 은 #29) |
