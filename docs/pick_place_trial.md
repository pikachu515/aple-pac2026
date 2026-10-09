# 사과 1개 집기 → 상자 적재 → 복귀 시간 시험

`tools/run_pick_place_trial.py`는 사과 1개를 집어 상자에 놓고 시작 자세로 돌아오는
한 사이클을 MuJoCo 접촉 물리로 실행하고 시간을 잰다. CAN·실제 카메라·Studio 서비스를 열지 않는다.
조건(속도·이송 높이·착지 속도 등)을 바꿔 가며 사이클 시간을 빠르게 비교하는 것이 목적이다.

- **사과 좌표**: 원래는 M1(인식)이 주는 값이다. 지금은 설정 파일의 `apple.center_m` 을 받았다고 가정한다.
  인식·색 판정·회전 검사는 하지 않는다.
- **상자 좌표**: `--seed` 로 무작위 배치한다(같은 seed = 같은 위치). 고정하려면 `--box X Y`.
- **실행 경로**: 실제 코드와 같은 `MotionExecutor.pick_for_inspection` → `place_held_apple` →
  `move_named('home')` 을 Piper 궤적 어댑터·구간 검사·연착지(soft landing)로 실행한다.
  사과는 가정 링 그리퍼와의 접촉·중력으로만 움직인다([gripper_physics.md](gripper_physics.md)).

## 실행

```bash
cd /mnt/extra/aple-pac2026
PY=~/.venvs/piper-daemons/bin/python3

$PY tools/run_pick_place_trial.py --seed 1                 # 1회 (계산 약 4초)
$PY tools/run_pick_place_trial.py --seed 1 --repeat 5      # 상자 위치 5가지 (seed 1~5)
$PY tools/run_pick_place_trial.py --apple 0.42 -0.10 --box 0.30 0.30 --target REJECT

# 조건 비교: 같은 seed 들을 값마다 다시 실행하고 평균을 비교
$PY tools/run_pick_place_trial.py --seed 1 --repeat 3 \
    --compare execution.soft_landing.slow_speed_scale=0.008,0.03,0.1

# 옆에서 실제 시간 속도로 보기 (창). --live-rate 0.5 는 절반 속도
$PY tools/run_pick_place_trial.py --seed 1 --live

# 영상 파일 (side.mp4, 30 fps)
MUJOCO_GL=egl $PY tools/run_pick_place_trial.py --seed 1 --video
```

`--live` 는 물리 시간을 벽시계에 맞춰 재생한다. 경로 계산 중에는 물리 시간이 멈추므로
창에서도 멈춰 보인다(아래 "계산 멈춤").

출력 예:

```
  seed      1 상자(+0.329,+0.286) 사이클 8.00s | 집기 2.76 빈손이송 0.54 상자이송 1.77 착지·놓기 2.91 복귀 0.57 | 계산 5.6s
요약 (사이클 = 집기 시작 ~ 시작 자세 복귀, 물리 시간)
  기본: 성공 5/5 · 평균 7.60s · 최소 7.29s · 최대 8.00s
```

- **사이클**: 집기 시작부터 시작 자세 복귀까지의 물리 시간. 실제 로봇 시간의 실측이 아니다.
  같은 PC·seed·조건끼리 비교한다.
- 단계 시간: `집기`(그리퍼 열기·빈손 이송·하강·파지·들기, `빈손이송` 포함), `상자이송`, `착지·놓기`(감속 하강·놓기·후퇴), `복귀`.
- **계산**: 프로그램 실제 경과 시간. 실패하면 이유를 출력하고 종료 코드 1.

결과는 `logs/pick-place-trial/<시각>/` 에 남는다(커밋하지 않음). `runs.csv` 에 실행별 시간,
실행 폴더마다 `report.json`·`trace.json`·`virtual-config.yaml`·`assumptions.yaml`(`--video` 면 `side.mp4`).

## 조건 바꾸기

자주 바꾸는 값은 `configs/simulation/pick_place_trial.yaml` 에 있다. 파일을 고치거나
`--set` 으로 한 번만 바꾼다. `--compare` 는 같은 형식의 키 하나에 값 여러 개를 준다.

| 바꿀 것 | `--set` / `--compare` 키 |
| --- | --- |
| 빈손 관절 속도 (rad/s) | `simulation.in_hand_scene.empty_joint_velocity_rad_s` |
| 사과 든 관절 속도 (rad/s) | `simulation.in_hand_scene.loaded_joint_velocity_rad_s` |
| 마지막 감속 하강 속도 배율 | `execution.soft_landing.slow_speed_scale` |
| 감속 하강 거리 (m) | `execution.soft_landing.final_descent_m` |
| 이송 높이 (m) | `trial.transfer_height_m` |
| 이송 중 툴 기울기 (rad, π=수직) | `trial.transfer_tilt_rad` |
| 상자 위 hover 여유 (m) | `workcell.placement.margin_m` |
| 그리퍼 개폐 대기 (s) | `assumed_gripper.gripper_settle_s` |
| 목표 도착 판정 (rad) | `assumed_gripper.joint_arrival_tolerance_rad` |
| 나머지 상자를 장애물로 배치 | `trial.other_boxes=random` |
| 인식 좌표 오차 (m) | `trial.apple.perception_error_m=[0.005,0,0]` |

`trial.` 은 시험 설정 파일, `assumed_gripper.` 은 그리퍼 물리 가정, 나머지는 가상 설정 전체(`virtual-config.yaml` 과 같은 구조)를 바꾼다.
없는 키는 오류로 거부한다. 속도 기본값은 in-hand 속도 탐색 선택값(`in_hand_speed_selected.yaml`)이다.

## 기본값의 이유와 확인 결과 (10/9)

- 파지 자세(툴 160°)로는 J5 한계(±65°) 때문에 TCP 0.14 m 정도까지만 닿아 상자 위를 지나기 어렵다.
  수치 IK 는 관절 한계를 보지 않으므로 한계 여유 0.03 rad 로 다시 확인했다. 140° 는 X 0.25~0.48 m 에서
  TCP 0.20~0.24 m 까지 닿아 기본값을 **140° / 0.20 m** 로 두었다. 시작·복귀 자세는 이 이송면 위
  `start_xy_m` 에서 IK 로 구한다.
- seed 1~5, 기본값: 5/5 성공, 사이클 평균 7.60 s (7.29~8.00 s).
  같은 seed 를 기존 in-hand 처럼 툴 수평·0.24 m 로 옮기면(`--set simulation.in_hand_scene.tilt_rad=1.5708`) 평균 11.35 s.
- 착지·놓기 단계가 가장 길다(약 2.9 s). 감속 하강 배율 0.008 → 0.03 → 0.1 에서 seed 1~2 평균 7.81 → 6.37 → 6.04 s.
  사과 손상 모델이 없으므로 빠른 착지가 실기에서 안전하다는 근거는 아니다.
- `trial.other_boxes=random` (seed 7~9) 도 3/3 성공, 평균 7.48 s.

물리 시간은 가상 모터 추종·목표 보간 모델의 결과다. 실기 속도·가속도 한도, 통신 지연, 카메라는 반영하지 않는다.

## Piper Studio 웹에서 보기

성공한 실행 폴더는 기존 재생 도구로 Studio 가상 팔(`sim_follower1`)에서 다시 볼 수 있다.
가정 그리퍼 simd 를 그 실행의 가정값으로 띄운 뒤([gripper_physics.md](gripper_physics.md) 의 임시 서비스 절차,
`studio_assumed_simd.py` 에 `--assumptions <실행 폴더>/assumptions.yaml` 추가) 재생한다.

```bash
$PY tools/replay_in_hand_studio.py logs/pick-place-trial/<시각>/01-seed-1
```

재생은 실시간이라 느리다. 조건 탐색은 오프라인 실행으로 하고, 확인할 결과만 재생한다.

## 빠른 설정 (10/9 추가)

시험 파일의 `set:` 목록이 기본으로 적용된다. seed 1~5 사이클 7.60 → 4.22 s, 5/5 성공.

| 바꾼 값 | 기존 → 현재 | seed 1~3 단독 효과 |
| --- | --- | --- |
| 마지막 감속 하강 배율 | 0.008 → 0.05 | 7.65 → 6.05 s |
| 직선 경유점 간격 `max_cartesian_step_m` | 0.01 → 0.06 m | 7.65 → 6.63 s (1 cm 마다 멈춰 도착 확인하던 것) |
| 그리퍼 개폐 대기 | 0.4 → 0.2 s | 7.65 → 6.85 s |
| 이송 경로 후보 | 직선·원호 모두 계산 → 첫 안전 경로 | 사이클 동일, 계산 멈춤 감소 |

도착 판정 `assumed_gripper.joint_arrival_tolerance_rad` 0.02 → 0.05 를 더하면 4.22 → 3.40 s 지만
도착 확인을 느슨하게 하는 것이라 기본값에는 넣지 않았다. 팔의 실제 최고 관절 속도는 약 90~130°/s
(가상 모터 추종 한계)로, 명령 보간 속도를 더 올려도 빨라지지 않는다.

### 계산 멈춤

사과를 든 직후 상자까지의 경로를 계획(IK + 0.01 rad 간격 구간 안전 검사)하는 동안 팔이 멈춘다.
직선·원호 후보를 모두 계산하면 약 1 s, 첫 안전 경로만 쓰면 약 0.35 s.
실기도 같은 Python 계획 코드를 쓰므로 이 계산 시간은 실기 사이클에도 더해진다고 봐야 한다.
없애려면 다음 구간 경로를 이동 중(또는 파지 전)에 미리 계산해야 한다 — 아직 구현하지 않았다.

## 사과·상자 배치 탐색

```bash
$PY tools/search_pick_place_layout.py          # 사과 20곳 × 상자 20곳, 8코어 약 2분
```

베이스 기준 거리 0.26/0.33/0.40/0.47 m × 방향 −60/−30/0/30/60° 격자에서, 15 cm 이상 떨어진
모든 (사과, 상자) 조합 322개를 실행했다(빠른 설정, 시작 자세 (0.30, 0)). 190개 성공.
결과: `logs/pick-place-layout/20261009-210854/layouts.csv`.

- **가장 빠른 배치**: 사과 정면(0°) 0.33 m, 상자 같은 거리 ±30° 옆 → 3.66 s.
- **베이스 회전 각도가 가장 큰 비용**: 사과↔상자 방향 차이 30° → 60° → 90° → 120° 에서
  상자이송 평균 0.98 → 1.37 → 1.93 → 2.68 s. 거리 차이(0 → 0.21 m)는 1.32 → 1.39 s 로 거의 영향 없음.
- **같은 16 cm 를 옮길 때** (사과 (0.36, 0) 고정): 옆 0.80 s, 앞 0.96 s, 뒤 0.98 s (상자이송).
  사이클은 옆 3.73 s, 뒤 3.79 s, 앞 4.06 s. 짧은 거리면 옆(J1 회전)이 앞뒤(J2·J3·J5 동시)보다 빠르다.
- 거리 0.33~0.40 m 가 좋고, ±60° 와 0.26/0.47 m 끝은 느리다.
- 실패 132개는 대부분 배치 자체가 불가능한 경우: 사과 ±60°(0.26/0.40/0.47 m)는 이 가상 셀의
  작업 영역·핑거 간격 검사에서 파지 후보가 없음(98), 상자가 사과와 겹침(8), 상자 접근 구간 안전 검사 실패(24) 등.
- 사이클에 시작 자세 왕복이 들어가므로 시작 자세 (0.30, 0) 아래가 유리하게 나온다.
  실제 운영에서는 시작(촬영) 자세를 사과 영역 위에 두는 것과 함께 정해야 한다.

권장: 사과 영역은 시작 자세 바로 아래(정면, 0.33~0.40 m), 상자는 그 옆으로 방향 차이 30° 이내
(같은 거리, 중심 간 15~20 cm). 상자를 사과 영역 반대편이나 ±60° 이상 돌아가는 곳에 두지 않는다.

## 사과 3개 색 분류 (빨강 → 왼쪽 상자, 노랑 → 오른쪽 상자)

위 배치 탐색 결과대로 사과는 시작 자세 아래 중앙 영역에, 상자는 그 좌우(같은 거리)에 둔다.

```bash
$PY tools/run_color_sort.py              # 새 무작위 배치·색, 끝나면 옆 카메라 창에서 실제 속도로 재생
$PY tools/run_color_sort.py --seed 3     # 같은 seed = 같은 배치·색
$PY tools/run_color_sort.py --seed 3 --rate 0.5    # 느리게 재생
$PY tools/run_color_sort.py --seed 3 --no-window   # 숫자만
```

- 설정: `configs/simulation/color_sort_trial.yaml`. 사과 중심 영역 X 0.26~0.47, Y ±0.06 m, 중심 간 0.10 m 이상.
  색은 seed 로 빨강/노랑 무작위. 왼쪽/오른쪽은 로봇 기준(base_link +Y 가 왼쪽).
- 상자: 칸 0.12 m × 2행 3열 = 6칸(0.36 × 0.24 m), 중심 (0.33, ±0.29). 3행으로 키우면 바깥 칸이
  베이스 옆으로 너무 멀어 경로 안전 검사에서 막혔다(10/9 시험).
- 순서: 시작 → 가까운 사과부터 집어 색 상자에 놓고 바로 다음 사과 → 마지막에 시작 자세 복귀.
  칸은 실제 `PlacementPlanner` 가 먼 칸부터 채운다.
- **시간**: 이 PC 에서 경로 계산(IK·구간 검사)에 걸린 실제 시간을 팔이 제자리에서 기다린 시간으로 물리에 넣는다.
  그래서 출력 시간 = 같은 코드로 움직이는 실제 팔이 걸릴 시간(물리 모델 기준). `--fast` 는 계산 대기를 빼고 동작 시간만.
  창 재생은 기록한 움직임을 벽시계에 맞춰 보여 준다(시뮬레이션 16.18 s 를 실제 16.20 s 동안 표시 확인).
  화면 그리기를 계산과 동시에 하면 계산이 크게 느려져(대기 2.1 s → 16.8 s) 재생을 분리했다.
- seed 1~8 (`--fast`): 8/8 성공, 12.3~13.1 s. 계산 대기 포함 seed 1·4: 14.9 / 15.3 s (그중 대기 약 2.1 s).
  계산 대기는 PC 성능에 따라 달라진다.

## 실기 코드 반영 (10/9)

색 분류 시뮬레이션의 경로·움직임 방식과 속도를 실기 실행 코드에 넣었다. 좌표는 기존대로 인식·적재 계획이 준 값을 쓴다.

- `aple/hal/piper_motion.py` (실기 어댑터): 빈손으로 사과에 갈 때와 사과를 든 뒤 상자로 갈 때 모두
  이송면(`execution.transfer.height_m`)으로 올라가 → 그 높이에서 목표 위까지 이동 → 목표로 내려간다.
  이송 중 툴은 목표 자세에서 바깥으로 `outward_tilt_rad` 더 기울인다(베이스 좌표 기준이라 툴 RPY 규약과 무관).
  모든 경유점은 기존 IK·구간 안전 검사를 그대로 거친다. `height_m: null` 이면 이전 경로.
- `aple/execution/goal_builder.py`, `motion_executor.py`: 놓을 때 툴을 상자 칸 방위로 돌리고(`face_place_bearing`),
  `far_place_radius_m` 보다 먼 칸은 `far_place_tilt_rad` 만큼 바깥으로 기울인다(사과 중심 기준 회전, 구형 사과).
- `configs/execution.yaml`: 이송면 0.20 m·기울기 20°/10°, 직선 경유점 간격 0.06 m,
  관절 속도·가속도 한계 = 시뮬레이션에서 관절이 실제로 낸 최고 속도 / 상위 1% 가속도
  ([1.82, 1.62, 2.14, 1.49, 2.32, 1.86] rad/s, [51.2, 39.9, 77.7, 58.9, 117.3, 75.9] rad/s²).
  `configs/safety.yaml` `speed_scale_max` 0.5 → 1.0. **실제 팔에서 잰 값이 아니다.**
- 검증: `run_color_sort.py` 는 이제 실기 어댑터와 모드 A `pick_place` 경로를 그대로 실행한다
  (`report.json` 의 `motion_path: hardware_adapter_pick_place`). seed 1~4 성공, 15.5~16.0 s(계산 대기 포함).
- 기존 가상 셀(`configs/simulation/piper.yaml`)은 이송면 없이 이전 경로를 유지한다.
