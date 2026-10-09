# 현장 보정 절차 — 시뮬레이션은 참고만, 값은 현장에서 정한다

## 원칙 (10/7 결정)

합성 장면·시뮬레이션 테스트가 보증하는 것은 **"코드가 의도대로 이어지고, 정해진 규칙대로 판단한다"** 까지다.
임계값·오차·오프셋·파지력 같은 **수치는 하나도 믿지 않는다.** 카메라 노이즈, 조명, 사과 표면, 핑거, 로봇 영점은
현장에서만 알 수 있다. 그래서 현장에서는

1. 남은 값을 한눈에 보고 (`field_values list`)
2. 실제 장면을 찍어 저장하고 (`run_cycle --save`)
3. 값을 바꿔 **같은 실제 장면으로** 바로 다시 돌려 보고 (`run_cycle --replay --set`)
4. 정해지면 한 줄로 설정 파일에 써 넣고 (`field_values set`), 틀리면 되돌린다 (`field_values undo`)

설계를 바꿔야 할 때도 같은 순서다. 코드 상수가 아니라 `configs/*.yaml` 값으로 바뀌도록 짜 두었으므로,
대부분의 조정은 코드 수정 없이 끝난다 (AGENTS.md: 현장 측정 값은 configs 에 둔다).

## 도구

| 도구 | 하는 일 |
|---|---|
| `python tools/field_values.py list [--owner A~D]` | 아직 현장 값이 아닌 것 (`[임시]` · `PLACEHOLDER` · `null`), 역할별 |
| `python tools/field_values.py set 키 값 --note "무엇으로 쟀나"` | 그 줄만 바꾸고 주석 유지, 표시를 `[실측 MM/DD]` 로. 설정이 깨지면 원상복구 |
| `python tools/field_values.py undo` / `history` | 마지막 변경 되돌리기 / 현장 변경 이력 (`logs/field_changes.csv`, 발표 근거) |
| `python tools/run_cycle.py --live --save data/field/이름` | Piper 카메라로 판정·계획 (로봇 안 움직임) + 원본 장면 저장 |
| `python tools/run_cycle.py --replay data/field/이름 --set 키=값` | 저장한 실제 장면을 값만 바꿔 다시 판정·계획 (파일은 그대로) |
| `python tools/color_session.py card · suggest · truth · curve --apply` | 색 기준 실측 → 설정에 바로 반영 (역할 C, `docs/inspection_bench.md`) |
| `python tools/judge_live.py` | Piper Studio 를 켠 채로 실시간 판정 화면 |
| `python tools/rs_check.py --save-calib` | D435 깊이→컬러 보정값 (장치당 1회) |

## 대회 전에 미리 준비할 수 있는 값 (사전값)

필수 값 대부분은 현장 배치·조명을 봐야 정해진다. 아래만 로봇 사양·설계 파일로 **미리 알 수 있다** — 값을 미리 넣지는 않고,
담당자가 출처를 챙겨 와서 현장 1~5단계에서 `field_values set` 으로 바로 넣고 실물로 확인한다.

| 값 | 담당 | 출처 | 현장 확인 |
|---|---|---|---|
| `safety.joint_limits_rad` | B | PiPER 공식 URDF (Piper Studio `piper_robot/kinematics.py` `joint_limits`): J1 ±2.618, J2 0~3.142, J3 −2.967~0, J4 ±1.745, J5 ±1.222, J6 ±2.094 rad. 안쪽으로 여유를 둔다 | Piper 조그로 양 끝 접근 |
| `gripper_attachment.*` 7개, `workcell.tcp_offset_m`, `grasp_planner.finger_thickness_m`·`finger_width_m` | D (A) | 링형 핑거 Fusion 설계 파일 — 플랜지→링 중심 거리, 핑거 두께·폭, 접촉면 간격 | 장착 후 캘리퍼스 |
| `piper_web.camera_id` | B | D435i 시리얼 243122071248 (Piper 표기 예: `rs_243122071248_color`) | Piper 카메라 화면의 이름과 대조 |
| `piper_web.gripper_width_range_m` | B·D | PiPER 그리퍼 사양 0~70 mm | 링 장착 후 닫힘·열림 폭 실측 |

현장에서만 정해지는 값: 카메라–로봇 캘리브레이션, 트레이·상자 위치, home/scan 자세, 작업 공간 범위, `piper_web.iface`(로봇 등록 이름),
색 기준(조명), 검출·복구 기준. 이 값들은 아래 순서대로 정한다.

## 1일차 현장 보정 순서 (14:30 ~)

앞 단계가 뒤 단계의 전제다. 표의 "통과 기준"을 넘기 전에는 다음 단계로 가지 않는다.

| 순서 | 값 | 담당 | 도구 | 통과 기준 |
|---|---|---|---|---|
| 1 | `piper_web.iface`, `piper_web.camera_id` | B | Piper Studio 화면 확인 → `field_values set` | `run_cycle --live` 가 장면을 받는다 |
| 2 | D435 깊이→컬러 보정, `live_camera.*` | A | `rs_check --save-calib` | `judge_live` 에서 사과 마스크가 사과에 겹친다 |
| 3 | 회색 카드 `color.card_json` | C | `color_session card` | 조명 ΔE ≤ 2.5 (OK) |
| 4 | `workcell.T_base_camera` | A | `touch_calib` (#13) | 점 터치 잔차 RMS ≤ 4 mm |
| 5 | `workcell.tcp_offset_m`, `piper_web.gripper_width_range_m` | D | 캘리퍼스 → `field_values set` | FK(IK(T)) ≈ T, 실제 핑거 끝이 목표점에 온다 (이슈 #24) |
| 6 | `safety.joint_limits_rad`, `safety.workspace_aabb_m` | B | Piper Studio 조그로 실측 | 실측 전에는 스크립트 실행 금지 (fail-closed) |
| 7 | 트레이·상자 위치 `workcell.tray`, `workcell.boxes.*.origin_m` | A | `m1_live` (#13) | 상자 칸 높이가 빈 상자에서 0 근처 |
| 8 | 사과 직경 (깊이 vs 캘리퍼스) | A·D | `run_cycle --live --save` + 캘리퍼스 | 직경 오차 ≤ 3 mm |
| 9 | 착색 기준 `varieties.*.hue_range_deg`, `min_chroma`, `top_view_bias`·`top_view_error`, `delta_up` | C | `color_session suggest · truth --poses first|all · curve --apply` | 육안 판정 일치율 ≥ 90% (`compare`) |
| 10 | 기본 fixed_effort: Studio 토크·장착 후 접촉 폭·패드 압축·접촉 안정 시간; calibrated_force 선택 시 `size_bins.force.*`, `size_bins.gripper_calib` 및 피드백 백엔드 | D·B | 실측 + `field_values set`; FSR는 선택 경로 | 리프트 후 새 깊이 영상으로 파지 확인, 운반에서 슬립/손상 없음 |
| 11 | 모드 B `inspection.slip_check.*`, `inspection.in_hand_mask.surface_tol_m` | C | 미끄러지지 않게 잡고 검사 자세로 돌리며 `--save` → 거리 분포 | 미끄러지지 않은 회전의 99% 가 통과 |
| 12 | `execution.placement_verify.height_tol_m` | B·A | 빈 칸 반복 촬영의 높이 흔들림 | 흔들림의 3배 |

각 단계가 끝나면 `field_values list`의 미측정 항목이 줄어든다. 목록 개수는 주석 기준이며 장착 여부·백엔드 조건을 구분해야 한다.
10/9 실기 조립에는 시간 궤적·접촉 안정·충돌 링크 등 추가 값이 필요하다. [전체 인수인계와 추가 설정](pr_execution_handoff_20261009.md)을 확인하고
`missing_for_live_execution` 점검 및 실제 추종/파지/안착 확인을 모두 통과한 뒤 `run_cycle --live --execute`를 연다.
비워 두는 게 정상인 값(자동 선택·자동 계산·다른 모드에서만 필요)은 설정 주석에 `[선택]` 을 달아 필수에서 뺀다 (`list --all` 로 보기).

**탑뷰 판정 구간 (9단계)**: 탑뷰 오차는 한쪽으로 쏠린다(10/7 실측: 꼭지가 위면 +4~+18 %p). `truth` 가 치우침(`top_view_bias`)과
남는 흔들림(`top_view_error`)을 함께 제안한다. 현장 사과가 꼭지 위로 놓이면 `--poses first`, 방향이 무작위면 `--poses all`.
**truth 를 `--apply` 하기 전 기본값(치우침 0, ±0.10)은 꼭지 위 사과를 '중→상' 으로 보낼 수 있다.**

## 현장 반복 루프 (판정·계획을 고칠 때)

```bash
python tools/run_cycle.py --live --cycles 5 --save data/field/1009_a          # ① 실제 장면 5개 저장 + 지금 값으로 판정
python tools/run_cycle.py --replay data/field/1009_a --set varieties.fuji.min_chroma=18   # ② 값만 바꿔 같은 장면 재판정
python tools/field_values.py set varieties.fuji.min_chroma 18 --note "1009_a 재판정"      # ③ 낫다면 확정
python tools/field_values.py undo                                              # ④ 아니면 되돌림
```

- 저장한 장면은 색 보정 전 **원본**이라, 회색 카드를 다시 잡아도 그대로 다시 쓸 수 있다
- 판정이 이상하면 `color_session analyze data/field/1009_a` 로 오버레이를 눈으로 본다
- 설계 자체를 바꿔야 하면 (예: 판정 카메라·검사 각도) 바꾼 근거를 `docs/strategy.md` 에 적는다 (AGENTS.md)

## 기록과 공유

- `logs/field_changes.csv` 는 git 에 올라가지 않는다 (`logs/` 제외). 현장 실측 이력은 발표 자료(기술적 근거 15점)로 쓰므로 마감 전에 공유 드라이브에 복사한다
- 설정 파일 변경은 **대회 중에도 브랜치 → PR → CI 초록 → 머지**로 올린다 (10/8 결정). 여러 값을 한 PR에 모아도 되지만, 커밋은 한 값씩(예: `현장 실측: varieties.fuji.min_chroma 18`) — 되돌리기 쉽다. 급하면 `field_values` 로 먼저 바꿔 시험하고 PR 은 직후에 올린다
- 시뮬레이션·합성 결과로 정한 값은 모두 `[임시]` 또는 `null` 로 남겨 두었다. 현장에서 하나씩 `[실측]` 으로 바뀌는 것이 정상이다
