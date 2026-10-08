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
| `python tools/depth_check.py --live` · `fit` | 회색 카드 크기로 깊이 거리 확인·보정 (#29 직경 문제의 원인, 아래 2-1단계) |
| `python tools/rs_calibrate.py check · occ · tare · restore` | D435 자체 재보정 (rsd 를 멈추고). IMU 기울기 + 수직 높이로 기준 거리 계산 |

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
| 2-1 | 깊이 거리 `live_camera.depth_correction`, 카드 크기 `live_camera.depth_check.card_mm` | A | 자로 카드 실측 → `depth_check --live` (필요하면 `fit`) | 카드 거리 오차 ≤ 2% |
| 3 | 회색 카드 `color.card_json` | C | `color_session card` | 조명 ΔE ≤ 2.5 (OK) |
| 4 | `workcell.T_base_camera` | A | `touch_calib` (#13) | 점 터치 잔차 RMS ≤ 4 mm |
| 5 | `workcell.tcp_offset_m`, `piper_web.gripper_width_range_m` | D | 캘리퍼스 → `field_values set` | FK(IK(T)) ≈ T, 실제 핑거 끝이 목표점에 온다 (이슈 #24) |
| 6 | `safety.joint_limits_rad`, `safety.workspace_aabb_m` | B | Piper Studio 조그로 실측 | 실측 전에는 스크립트 실행 금지 (fail-closed) |
| 7 | 트레이·상자 위치 `workcell.tray`, `workcell.boxes.*.origin_m` | A | `m1_live` (#13) | 상자 칸 높이가 빈 상자에서 0 근처 |
| 8 | 사과 직경 (깊이 vs 캘리퍼스) | A·D | `run_cycle --live --save` + 캘리퍼스 | 직경 오차 ≤ 3 mm |
| 9 | 착색 기준 `varieties.*.hue_range_deg`, `min_chroma`, `top_view_bias`·`top_view_error`, `delta_up` | C | `color_session suggest · truth --poses first|all · curve --apply` | 육안 판정 일치율 ≥ 90% (`compare`) |
| 10 | 파지력 `size_bins.force.*`, `size_bins.gripper_calib` | D·B | FSR + `field_values set` | 사과 무게 + 가속에서 미끄러짐 없음, 멍 기준 아래 |
| 11 | 모드 B `inspection.slip_check.*`, `inspection.in_hand_mask.surface_tol_m` | C | 미끄러지지 않게 잡고 검사 자세로 돌리며 `--save` → 거리 분포 | 미끄러지지 않은 회전의 99% 가 통과 |
| 12 | `execution.placement_verify.height_tol_m` | B·A | 빈 칸 반복 촬영의 높이 흔들림 | 흔들림의 3배 |

각 단계가 끝나면 `field_values list` 의 **필수** 값 개수가 줄어든다. 0 이 되어야 실기 실행(`run_cycle --execute`)을 연다.
비워 두는 게 정상인 값(자동 선택·자동 계산·다른 모드에서만 필요)은 설정 주석에 `[선택]` 을 달아 필수에서 뺀다 (`list --all` 로 보기).

**깊이 거리 (2-1단계, #29)**: 10/7 실물 D435i 는 실제 0.30 / 0.43 / 0.74 m 에서 깊이가 26 / 33 / 45% 짧았다
(회색 카드 130×100 mm 기준). 그래서 약 85 mm 사과가 45~50 mm 로 나오고, 사과 중심이 테이블 아래로 들어가 파지가 바닥에 부딪혔다.
원인은 오른쪽 IR 카메라의 주점 어긋남(충격 등)이었고 **10/8 Tare 로 재보정해 카메라에 저장했다** — 카드 오차 −45% → +0.8~1.7%,
Piper Studio 경로에서 M1 직경 87.0 mm (자 86 mm). 저장 전 보정표는 `hardware/rs_calib/` 에 있다.

현장 순서 (카메라를 마운트에 단 뒤, `touch_calib` 전):
1. 카드를 트레이 높이에 눕혀 화면 가운데에 두고 `depth_check --live` — 2% 이내면 끝
2. 넘으면 `systemctl --user stop piper-rsd` 후 렌즈–트레이 면 **수직 높이**를 줄자로 재고
   `rs_calibrate.py check --height-mm 높이` → `tare --height-mm 높이` (임시 적용 결과 확인) → 같은 명령에 `--write`.
   카메라가 기울어도 된다 — IMU 로 광축 거리를 계산한다. 카드 거리와 'IMU+높이로 본 카드 위치 깊이'가 다르면 줄자 높이를 의심한다
   (10/8: 78 cm 로 쟀으나 카드·IMU·보정 후 깊이 기울기가 모두 73 cm 를 가리켰다). Tare 는 화면 가운데에 사람·물체가 없게 한다
3. `systemctl --user start piper-rsd` → Piper 카메라 화면에서 D435 컬러·깊이 연결 → **컬러·깊이가 848×480 인지** 확인
   (10/8: rsd 재시작 뒤 프로필 `pac-top` 이 카메라와 매칭되지 않아(`unmatched`) 컬러가 1280×720 으로 나왔고, 깊이 정렬이 틀어져
   `depth_check` 가 엉뚱한 값을 냈다. `POST /api/cameras/resolution {"id": "rs:<시리얼>:color", "capture": [848, 480], "output": [848, 480]}`)
4. 다시 `depth_check --live` 로 통과 확인. 재보정을 못 하면 카드를 두 높이에서 찍어 `depth_check fit` 값을 `field_values set` 으로 넣는다

**IMU 권한 (노트북마다 1회)**: `rs_calibrate --height-mm` 가 `Permission denied` 면
`sudo cp hardware/udev/99-realsense-imu.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules` 후 **카메라 USB 를 다시 꽂는다**
(`udevadm trigger` 만으로는 `/dev/iio:device*` 권한이 바뀌지 않았다). `ls -l /dev/iio*` 가 `crw-rw-rw-` 면 된다.

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
