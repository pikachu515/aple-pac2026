# 중앙 회전·색상 재판정 접촉 시뮬레이션

`tools/run_in_hand_simulation.py`는 실물과 분리된 MuJoCo 실행기다. CAN·실제 카메라·로봇
서비스를 열지 않는다. 이 기능은 아래 새 명령으로 실행한다. 시뮬레이션 배치·속도·물리
가정은 별도 설정에만 있다. 10/9 코드 검토 개선에서 `piper_motion.py`의 검사 자세를
공통 툴 로컬 축 회전으로 수정했고, 실기 기본 설정에 미측정 보호 항목을 추가했다.
실기 연결·필수 실측값은 [인수인계](pr_execution_handoff_20261009.md)를 참고한다.

## 실행

현재 `--optimize`는 속도 탐색에서 검증한
`configs/simulation/in_hand_speed_selected.yaml`과
`configs/simulation/in_hand_speed_search_scene.yaml`을 기본으로 사용한다.
5개 배치의 사과 15개를 모두 집기·4뷰 촬영·재판정·적재했고 미끄러짐은 없었다.
세 사과 전체 물리 시간은 53.1~58.1초다. 시험 표본의 성공이며 모든 배치/실기의
성공률 100% 보장은 아니다. [탐색 결과·촬영 주기·복귀 명령](simulation_speed_search.md)을 참고한다.
원래 장면과 이전 속도 프로파일은 보존하며, `--scene-config`를 명시하면 그 장면을 사용한다.

```bash
cd /mnt/extra/aple-pac2026
~/.venvs/piper-daemons/bin/python3 tools/run_in_hand_simulation.py --optimize --seed 42
```

### 빠른 개발 시험과 시간 측정

매번 새 시뮬레이션을 만들지 않고 같은 실행기의 옵션을 바꿔 사용한다.

```bash
# 사과 1개: 영상·촬영 파일 저장 없이 실행, 웹 재생 없음
~/.venvs/piper-daemons/bin/python3 tools/run_in_hand_simulation.py --quick --seed 42

# 같은 배치의 사과 3개 전체 시험, 영상 없이 실행
~/.venvs/piper-daemons/bin/python3 tools/run_in_hand_simulation.py --seed 42

# 최종 확인용 영상·촬영 파일 저장
MUJOCO_GL=egl ~/.venvs/piper-daemons/bin/python3 tools/run_in_hand_simulation.py --seed 42 --video --save-captures
```

`--quick`은 설정 목록의 첫 번째 물리 사과만 처리한다. 나머지 두 사과와 모든 상자를
장애물로 유지하므로 같은 seed의 전체 시험과 배치를 비교할 수 있다. 인식 ID는 물리
사과 번호와 다를 수 있다. `--apple-count 2`는 첫 두 사과를 처리한다. 기본은 3개다.
빠른 모드에 `--video`, `--save-captures`, 2~3개 실행을 함께 지정하면 오류로 거부한다.
영상과 촬영 파일은 기본으로 저장하지 않지만 판정용 RGB-D 합성과 물리·미끄러짐 검사는
생략하지 않는다. 부분 시험 성공은 선택한 사과에 대한 결과이며 전체 장면 성공이 아니다.
부분 시험 폴더의 웹 재생은 기본으로 거부한다. 명시적인 `--allow-partial`은 검증한
사과 ID만 재생·평가하며 전체 장면 성공으로 표시하지 않는다.

`report.json`의 `wall_time_s`는 프로그램 실제 경과 시간, `simulated_time_s`는 MuJoCo
물리 시간이다. 물리 시간은 실제 로봇 처리시간의 실측값이 아니다. 같은 기기·seed·사과 수·
저장 옵션으로 변경 전후를 비교한다. 영상 없는 실행도 물리 계산과 IK 때문에 시간이 든다.

`timings.csv`와 `report.json`의 `timing`에는 집기, 빈손 이송, 중앙 이송, 회전·검사,
색상 촬영, 재판정, 상자 이송, 착지·놓기·후퇴, 최종 안정화 시간을 기록한다.
관절 실행(`joint_motion`)과 자세 경로 실행(`pose_motion`)도 기록해, 후자의 자체 시간으로
IK·경로 검증 등 관절 실행 바깥 비용을 살펴볼 수 있다. 영상·촬영 파일 저장 비용도
옵션을 켜면 따로 기록한다. 생성 전 설정·라이브러리 초기화와 결과 저장 등 일부 비용은
전체 경과 시간에만 포함된다.

단계는 중첩된다. 예를 들어 `inspection`에 `center_transfer`와 `rotation_inspection`이
포함되고, 회전 검사에는 `color_capture`가 포함된다. `wall_time_s`와
`simulated_time_s` 열을 모든 행에서 합산하면 중복된다. `exclusive_*` 열은 하위 단계
시간을 뺀 값이다. JSON의 개별 기록에는 물리 사과 번호와 예외 여부도 남긴다.
하위 실행기가 실패를 반환하는 경우에는 시간 기록의 예외 여부와 별도로 보고서의
`success`, `errors`, 사과별 실행 결과를 확인한다. 촬영 전 정지 대기는 현재 실행기의
실제 시간 대기이므로 물리 시간과 다르게 나타날 수 있다.

빠른 모드 seed 42 실측 결과(`logs/in-hand-benchmark/seed-42/report.json`):
실제 실행 56.2초, 물리 시간 67.9초. 첫 물리 사과가 4장 검사 후 MIDDLE 상자에 안착했고
미끄러짐은 0회였다. 물리 시간 기준 집기 7.0초, 중앙 이송 10.2초,
회전·검사 28.5초, 상자 이송 14.1초, 착지·놓기·후퇴 6.8초였다.
색상 RGB-D 합성은 실제 시간 총 0.62초였다. 이번 한 사과 시험에서는 중앙 이송 이후
회전·검사 동작이 가장 길었으며, 전체 3개 시험이나 실기의 병목을 확정하는 수치는 아니다.
동일 seed의 기존 전체 시험과 사과·상자 설정이 같은 것과 영상·촬영 파일 미생성을 확인했다.
시간 기록/부분 웹 실행 거부를 포함한 관련 테스트 49개가 통과했다.

이 PC의 Piper Studio 데몬용 가상환경에 MuJoCo/Piper 수학 모델/OpenCV가 설치되어 있다.

```bash
cd /mnt/extra/aple-pac2026
MUJOCO_GL=egl ~/.venvs/piper-daemons/bin/python3 tools/run_in_hand_simulation.py --video
```

`--seed 42`를 추가하면 같은 배치가 재현된다. seed가 없으면 매번 새 배치를 생성한다.
`--output logs/다른이름`으로 출력 위치를 바꿀 수 있다. 같은 seed를 다시 실행해도
기존 결과를 덮어쓰지 않고 `seed-42-2`처럼 새 폴더를 만든다.

결과 폴더에는 실제 생성한 설정 `virtual-config.yaml`, 관절·그리퍼·촬영 이벤트
`trace.json`, 판정/낙하/안착 결과 `report.json`, `--video`의 측면 영상 `side.mp4`,
`--save-captures`의 합성 RGB PNG와 깊이 NPY `captures/`, 단계별 시간 `timings.csv`가 생긴다. 데이터·영상은 logs에만 두며
커밋하지 않는다. 성공은 종료 코드 0, 실패는 1이고 오류 원인을 report에 남긴다.

## 사과 크기와 질량의 근거

[한국농촌경제연구원 보고서](https://www.krei.re.kr/attach/pdf/91374p.pdf)의 표 3-10
(본문 28쪽)은 농촌진흥청 원예연구소 사과시험장 자료를 인용한다. 평균 과중 297.1g,
횡경 87.3mm, 최저 252g/82.6mm, 최고 337g/91.9mm를 보고했다.
1997~1998년 조사에 기반한 참고값이며 현재 전국 사과의 평균이나 품종별 분포를 뜻하지 않는다.
[농촌진흥청 후지 출하 자료](https://www.rda.go.kr/board/board.do?boardId=farmprmninfo&dataNo=100000061967&mode=updateCnt&prgId=day_farmprmninfoEntry)는
최고 품질 후지의 목표 과중을 320g ±10%로 제시한다. 이는 평균값이 아닌 품질 목표다.

| 가상 사과 | 횡경에 맞춘 지름 | 실제 물리 모델 질량 |
| --- | --- | --- |
| 작은 표본 | 0.0826 m | 0.252 kg |
| 대표 표본 | 0.0873 m | 0.2971 kg |
| 큰 표본 | 0.0919 m | 0.337 kg |

충돌 형상은 구 근사다. 실제 사과의 종경·꼭지·변형과 과피의 손상은 모델링하지 않는다.
착색 무늬는 판정 시험용 합성 데이터이며 통계적 한국 사과 색 분포가 아니다.
질량은 사과별 MuJoCo body에 각각 적용된다. 기존 실행 프로파일의 질량 추정 모델을
바꾸거나 실물 파지력으로 환산하지 않는다.

## 배치와 중앙 검사

가상 설정은 `configs/simulation/in_hand_scene.yaml`에 있다. 실측 workcell에 합치지 않는다.

- 사과와 UPPER/MIDDLE/REJECT 상자 모두 같은 X 0.10~0.55m, Y -0.38~0.38m 영역에서 샘플링한다.
- 물체 중심 사이 거리는 최소 0.17m이고, 외곽의 겹침도 거부한다. 상자 셀 폭은 0.12m다.
- 높은 이송 자세의 도달성을 위해 로봇 베이스에서 반경 0.30~0.49m를 사용한다.
- 검사 사과 중심은 맵 XY의 정확한 중점 `[0.325, 0, 0.24]` m다. 중앙 바닥 원은 검사 공간 표시다.
- 중앙 지점 주위 0.08m는 다른 물체 외곽이 들어오지 않도록 비운다.
- 들어 올린 뒤 높은 위치에서 원호를 따라 이동한다. 베이스 가까이의 손목 특이 자세와
  낮은 직선 이동 중 옆 사과를 미는 문제를 피하기 위한 가상 경로다.
  반경 0.48m보다 먼 상자는 적재 자세를 150도로 기울여 손목 관절 범위를 확보한다.
- 배치 샘플링은 겹침/공간 조건만 검사한다. 실행 실패를 숨기려고 성공할 때까지
  재실행하거나 다른 seed로 자동 바꾸지 않는다. IK/관절 범위/물리 접촉 실패는 그대로 기록한다.

## 색상 재판정과 미끄러짐

M1 합성 RGB-D 인식 → 집기 → 중앙 이동 → 툴 수평 기울임 → 툴 축 기준
-100/-35/+35/+100도 회전·촬영 → 기존 C의 다면 누적·재판정 → 최종 상자 적재 순서다.
시뮬레이션 전용 어댑터가 이 각도·이송 경로를 설정하며, 회전 자세 계산은 실기와 같은
`Ry(tilt) @ R_tool_axis(angle)` 함수를 사용한다. 실기의 각도·검사 중심은 별도 현장 설정이다.
±120도 J6 범위를 넘어 360도 연속 회전을 명령하지 않는다.

사과는 실제 접촉과 중력으로만 이동한다. 붙이는 weld나 위치 강제 이동은 없다.
과피의 적색 영역은 사과 body 좌표계에 고정되고, 촬영은 실제 물리 사과 위치/회전에서
합성한다. MuJoCo 광선 검사로 핑거·로봇·상자가 가린 픽셀을 제외한다. 기존 M2 모듈이
합성 RGB-D와 구면 지도를 판정하며, 관측 부족이나 경계값의 REJECT를 임의로 바꾸지 않는다.

물리 이동 중에는 예상 파지 중심과 실제 사과 중심의 차이가 0.012m를 넘으면 즉시
실행을 중단하고 `slips`에 시간/사과/변위를 남긴다. 기존 실행기의 촬영별 미끄러짐 검사도
함께 사용한다. 이 검사는 시뮬레이션 정답 위치에 접근하므로 실물 센서보다 유리하다.

빈손 이송 1.0rad/s, 들고 이송 0.35rad/s, 검사 회전 0.20rad/s를 목표 관절 보간에 적용한다.
실제 속도는 추종 지연과 기존 단계별 speed_scale에 따라 더 낮아질 수 있다. 빈손/파지/검사
상태와 적용 속도를 trace에 기록한다. 실물 속도나 Studio 웹의 속도 계약을 바꾸지 않는다.

큰 사과용 가상 컵 곡률 반경은 0.046m로 따로 적용한다. TCP/각 핑거 5N 힘 제한/마찰계수
등은 기존 가정이며 사진의 D1~D4에서 계산한 실제 부품 형상이 아니다. 설정과 가정을
report에 함께 기록한다. 가상 시험의 무낙하 결과는 실제 수제 그리퍼의 안전 보장이 아니다.

## 선택적 속도·계산 개선 후보

기본 실행은 비교용으로 유지한다. 후보를 사용하려면 `--optimize`를 추가한다.

```bash
# 이전 회전 방식의 속도 후보 비교 (불투명 면 추가 이전)
~/.venvs/piper-daemons/bin/python3 tools/run_in_hand_simulation.py --quick --seed 42 --optimize --optimization-config configs/simulation/in_hand_optimization.yaml

# 전체 3개 확인
~/.venvs/piper-daemons/bin/python3 tools/run_in_hand_simulation.py --seed 42 --optimize --optimization-config configs/simulation/in_hand_optimization.yaml
```

후보 값은 `configs/simulation/in_hand_optimization.yaml`에만 둔다. 다음을 적용한다.

- 높은 이송면에서 직선과 기존 원호 경로를 각각 IK/관절 범위/기존 segment validator로
  검사하고, 별도의 MuJoCo 상태에서 로봇과 상자·들고 있지 않은 사과의 접촉도 검사한다.
  검사를 통과한 후보 중 관절 보간의 예상 시간이 짧은 경로를 실행한다. 두 후보에 대한
  선택이며 전역 최적 경로라는 뜻은 아니다. 중간 목표 도착 대기는 유지한다.
  `route_choices`에 선택, 후보 예상 시간, 거부 이유를 기록한다. probe는 실제 실행
  `MjData`나 사과 위치를 수정하지 않는다. 파지된 사과의 동적 움직임은 실제 접촉으로
  시험하며 후보 접촉 검사가 모든 동적 충돌을 증명하지는 않는다.
- 매 물리 스텝의 미끄러짐 검사는 유지하되, Python에서 공식 FK를 반복 계산하는 대신
  MuJoCo의 같은 link6 자세와 같은 TCP 오프셋을 사용한다. 초기 FK 불일치는 거부한다.
  20개 관절 자세의 공식 FK 일치와 미끄러짐 주입 시 정지를 테스트했다.
- 검사 전 실제 시간 sleep을 없애고 촬영 전 물리를 0.06초 진행한다. 합성 촬영에는
  실제 카메라의 프레임 지연이 없다. 실제 카메라의 안정화 대기를 0으로 만드는 변경이 아니다.
  네 방향, 촬영 해상도, 색상 판정 기준, 물리 timestep과 미끄러짐 검사 주기는 유지한다.
- 빈손 목표 관절 속도만 1.0 → 1.5rad/s로 시험한다. 이는 제조사 최고속도나 실기 허용값이
  아니다. 운반 0.35rad/s, 회전 0.20rad/s는 유지한다.
- 착지 속도 배율 0.2를 시험하고 마지막 0.03m만 감속한다. 기존 공용 soft landing helper는
  `hover_z - final_descent_m`에서 감속을 시작하므로 이 장면의 hover 간격 0.07m에서는
  실제 감속 구간이 0.04m였다. 시뮬레이션 후보에서만 release 높이 기준으로 보정한다.
  공용 helper와 실기 설정은 수정하지 않았다. 사과 손상 모델이 없으므로 0.2가 실제로
  손상 없는 최적 배율이라는 결론은 내리지 않는다.

seed 42 첫 물리 사과 비교: 기본 실제 실행 56.2초/물리 67.9초,
후보 실제 실행 18.7초/물리 61.4초. 물리 동작은 약 9.6% 감소했고,
실제 실행시간의 큰 감소는 계산 비용 절감도 포함한다. 후보의 전체 3개 시험은
실제 실행 55.0초/물리 188.5초로 기존 물리 215.9초보다 약 12.7% 짧았다.
각 사과는 4장 검사 후 기존과 같은 UPPER/MIDDLE/REJECT 상자에 안착했고 미끄러짐 0회였다.
보고서는 `logs/in-hand-optimization-final/seed-42/report.json`과
`logs/in-hand-optimization-full/seed-42/report.json`에 있다. 영상 없는 오프라인 실행이며,
Studio 웹 전송/도착 대기에 같은 계산 시간 단축이 그대로 적용되는 결과는 아니다.

다른 배치 seed 7의 후보 전체 시험도 성공했다. 실제 실행 47.9초/물리 183.6초였고
기존 물리 211.3초 대비 약 13.1% 감소했다. 각 사과 4장 검사·동일 등급 적재,
미끄러짐 0회였다(`logs/in-hand-optimization-full/seed-7/report.json`).
관련 테스트 56개가 통과했다. 두 seed의 통과는 실기 손상 안전성이나 모든 무작위 배치의
성공률을 보장하는 검증은 아니다.

## 현재 계획: 기존 옆 회전으로 복귀

이전 빠른 후보는 `configs/simulation/in_hand_fast.yaml`이다.
현재 기본 후보는 위 속도 탐색 결과로 갱신했으며, 아래 수치는 이전 후보의 기록이다.
이전 속도 개선·옆 회전 후보인 `in_hand_optimization.yaml`도 보존한다.
빈손 1.5→1.8rad/s, 사과를 든 이동 0.35→0.55rad/s, 검사 회전
0.2→0.35rad/s로 올렸다. 마지막 3cm 감속 비율은 0.2→0.12727로
조정해 절대 관절 속도 상한 0.07rad/s를 유지한다. 이는 TCP 선속도나
실기 손상 안전성의 보장이 아니다. 촬영 수·물리 정착·파지력·미끄러짐
판정·불투명 곡면 그리퍼·경로 검사 기준은 유지한다.

웹 가상 재생의 도착 확인 간격은 0.1→0.02초다. 도착 확인과 가상 전용
보호를 없애지 않는다. 이 값은 시뮬레이션 복사 설정에만 반영한다.
개방 폭도 0.110→0.108m로 조정해 0.112m 가상 최대 개방 끝점에서
여유를 둔다. 사과 직경보다 큰지 검증하고, 파지 프로파일의 개방 폭에도
같은 값을 반영한다. SDK 인코더 범위를 완화하거나 잘라 맞추지 않는다.
실기 `configs/execution.yaml`, `configs/varieties.yaml`은 변경하지 않는다.

seed 42의 같은 곡면 그리퍼 비교: 기존 물리 동작 63.0초, 빠른 후보
43.5초로 약 31% 감소했다. 검사 구간은 37.8→23.8초다.
`logs/in-hand-fast/seed-42-2/report.json`, `seed-7/report.json`에서 각 배치의
사과 1개·4뷰·REJECT 적재 성공과 물리 미끄러짐 0회를 확인했다.
나머지 사과와 상자는 장애물로 유지하지만 3개 전체 처리 시험은 아니다.
첫 과속 후보(이동 0.75/회전 0.5rad/s)는 검사 후 상자 이동에서 IK가
실패해 채택하지 않았다(`logs/in-hand-fast/seed-42/`).

최종 개방 폭을 적용한 오프라인 실행은
`logs/in-hand-fast/seed-42-4/report.json`에서 성공, 물리 동작 43.5초를 확인했다.
최종 웹 재생도 `logs/in-hand-fast/seed-42-4/studio-4/report.json`에서
396개 목표 도착, 검사 4회, 목표 상자 안착·빈손 복귀까지 성공했다.
이는 사과 1개 부분 시험이다. 웹의 색상 결과는 오프라인 판정 기록을 재생한다.
개방 직후 가상 인코더가 일시적으로 범위를 벗어나는 경우에만 최대 1초 동안
정상 실측값을 다시 읽는다. 값 보정·클램프 없이 도착과 제어권 검사를 유지하며,
계속 범위를 벗어나거나 다른 오류가 발생하면 실패한다. 중간 개방 목표를 나눠
보내는 시도는 접촉 상태에서 도착하지 못해 채택하지 않았다.
관련 테스트 33개가 통과했다.

```bash
# 현재 빠른 후보
~/.venvs/piper-daemons/bin/python3 tools/run_in_hand_simulation.py --quick --seed 42 --optimize --scene-config configs/simulation/in_hand_full_floor.yaml --output logs/in-hand-fast
# 이전 속도로 비교 / 복귀
~/.venvs/piper-daemons/bin/python3 tools/run_in_hand_simulation.py --quick --seed 42 --optimize --optimization-config configs/simulation/in_hand_optimization.yaml --scene-config configs/simulation/in_hand_full_floor.yaml
```

후속 형상 수정: 흰색 링을 사과를 감싸는 구면 오목 형상으로 바꾸었다.
외경 78.664mm·내경 48.92mm는 유지한다. 구면 곡률 반경 0.046m는 기존
시뮬레이션의 가정이며 실측 오목 깊이가 아니다. 이 가정에서 내측 테두리와
외측 테두리의 깊이 차는 약 15mm다. 중앙 패드를 제거하고 접촉 패드를
구멍 바깥 둘레에 재배치했으며, 중앙으로 뻗던 지지대는 외측 테두리에서
끝나도록 줄였다. 새 접촉 패드는 기존 총 패드 질량을 나눠 사용한다.
이는 기존 접촉 배치를 바꾸는 물리 모델 변경이며 실제 파지 성공률을
보장하지 않는다. 속도·힘·TCP·미끄러짐 판정 기준은 유지한다.

초기 평면 링 기록(오목 형상 수정 전): 외경 78.664mm, 내경 48.92mm를
`configs/simulation/gripper_physics.yaml`에 기록했다. 모델은 외반경
0.039332m·내반경 0.02446m인 불투명 링을 양 핑거에 사용한다.
각 링을 96개 볼록 조각으로 구성해 MuJoCo의 광선 검사에서도 중앙
구멍이 열리도록 했다. 당시에는 기존 접촉 패드·샤프트·힘·TCP·속도
설정을 유지했다. 당시 평면 링의 두께 3mm는 가정이었다.
구멍 안에 남아 있던 접촉 패드와 샤프트는 위 후속 수정에서 제거·재배치했다.

현재 오목 링 시험은 `logs/in-hand-concave-ring/seed-42-3/report.json`에 있다.
사과 1개 집기·4뷰·REJECT 적재 성공, 물리 미끄러짐 0회, 관측률 0.257이다.
평면 링보다 테두리의 가림이 늘어 관측률이 낮아졌다. 새 형상으로 전체
무작위 배치나 실기 파지 안정성을 검증한 것은 아니다.

링 모델의 첫 옆 회전 시험(`logs/in-hand-ring/seed-42/report.json`)은
사과 1개·4뷰·재판정·REJECT 적재 성공, 물리 미끄러짐 0회였다.
관측률은 0.358이다. 아래의 solid-disk 실패 기록은 중앙 구멍 적용 전
결과로 보존하며, 현재 링 모델의 실패 결과가 아니다.

사용자 결정으로 위아래 회전 계획을 폐지했다. 현재 후보와 이전 속도
후보 모두 `inspection_rotation_mode:
tool_roll`로 기존 local-Z 회전을 사용한다. 불투명 원판의 가림은 유지한다.
실기 `inspect_skill.py`와 `configs/varieties.yaml`은 원래 회전 방식이므로
변경하지 않았다. 아래의 위아래·자유 XYZ·대각선 카메라 결과는 과거 비교
기록이며 현재 실행 계획으로 사용하지 않는다.

```bash
~/.venvs/piper-daemons/bin/python3 tools/run_in_hand_simulation.py --quick --seed 42 --optimize --scene-config configs/simulation/in_hand_full_floor.yaml
```

복귀 후 `logs/in-hand-side-restored/seed-42/report.json`으로 시험했다.
불투명 원판을 유지하면 첫 옆 회전 자세에서 가시 깊이 픽셀이 130개로,
기존 미끄러짐 검사 최소 200점에 못 미쳤다. 같은 자세에서 재촬영해도
확인되지 않아 `SLIP` 실패로 검사·적재를 중단했다. 물리 미끄러짐 기록은
0회지만, 색 재판정·적재가 성공한 시험은 아니다. 가림을 숨기거나 검증
임계값을 낮추지 않았다. 과거 불투명 면 없는 모델의 성공률·관측률을
현재 모델에 적용하지 않는다.

Piper Studio 웹 장면도 기존 카메라·불투명 모델을 유지한 채 초기 준비
상태로 다시 적용했다(`studio-ready.json`). 실패한 검사 경로를 성공한
실행처럼 웹에 재생하지 않았다. 위아래 실험 영상과 설정은 보관용이다.

## 불투명 그리퍼와 위아래 회전 (폐지된 실험 기록)

사용자 확인: 실제 수제 그리퍼는 불투명하고 옆면을 가리므로 검사에는 위아래 회전이
필요하다는 당시 가정으로 `configs/simulation/in_hand_vertical.yaml`을 시험했다.
현재 계획에서는 사용하지 않으며, 당시 결과만 재현하려면 이 설정을 명시한다.

```bash
~/.venvs/piper-daemons/bin/python3 tools/run_in_hand_simulation.py --quick --seed 42 --optimize --optimization-config configs/simulation/in_hand_vertical.yaml
# 같은 위아래 회전의 측면 영상
MUJOCO_GL=egl ~/.venvs/piper-daemons/bin/python3 tools/run_in_hand_simulation.py --apple-count 1 --seed 42 --optimize --optimization-config configs/simulation/in_hand_vertical.yaml --video
```

두 핑거를 잇는 local-Y 축을 기준으로 TCP pitch 140 → 123.3 → 106.7 → 90도로 기울인다.
사과 중심은 `[0.325,0,0.24]`m로 보정하고, 검사 중 핑거를 잇는 축은 수평으로 유지한다.
기존 local-Z roll 검사로 핑거를 번갈아 위아래에 두는 동작과 구분된다. 현재 위치에서는
더 넓은 pitch 목표가 IK/관절 범위를 벗어나므로 범위를 억지로 늘리지 않는다.

지름 0.08m의 불투명 원판을 양 핑거에 추가해 화면과 RGB-D 광선 검사에 반영한다.
이 치수는 가림 시험용 가정이고 사진 D1~D4의 점간 거리에서 추론한 지름이 아니다.
사진의 48.144/48.464/71.76/79.76mm 정보는 변경하지 않았다. 원판은 광학 표현이며
질량·접촉을 추가하지 않는다. 실제 오목한 컵의 접촉 역학은 기존 패드/샤프트 근사이고,
실제 CAD나 전체 물리 형상을 구현했다는 의미가 아니다. 불투명 면은 광선 검사에서
사과 픽셀을 가리며 사과에 붙이는 제약이나 위치 강제 이동은 추가하지 않는다.

첫 물리 사과의 오프라인 시험은 집기·4장 촬영·적재를 통과했다
(`logs/in-hand-vertical/seed-42/report.json`): 실제 실행 17.2초, 물리 45.8초,
미끄러짐 0회. 다만 관측률은 약 0.37이고 색상 구간이 넓어 REJECT 판정이다.
이전 불투명 면 없는 모델의 약 0.69와 동일 조건 비교는 아니며, 이전 관측 결과를
실제 불투명 그리퍼 성능으로 해석하면 안 된다. 가림을 반영한 위아래 회전 범위에서는
착색 등급 판정을 충분히 확정하지 못했다. 판정 기준을 완화하거나 다른 등급을 강제하지 않는다.
이는 전체 3개 분류·적재 성공 검증이 아닌 한 사과 시연이다.

Studio에는 이 시험의 `assumptions.yaml`로 가상 서비스를 시작한 뒤 다음처럼 시연했다.

```bash
~/.venvs/piper-daemons/bin/python3 tools/replay_in_hand_studio.py logs/in-hand-vertical/seed-42 --allow-partial
```

사과 1개만 실제로 들고 기울여 선택된 REJECT 상자에 놓았고 웹 결과도 성공했다.
`studio/report.json`의 `tested_apples:[1]`, `full_scene_complete:false`로 범위를 명시한다.
새 가정값은 모델 hash가 달라 이전 후보를 웹 재생할 때 해당 assumptions로 서비스를
다시 시작해야 한다. 실기 코드/실측 설정에는 이 회전·불투명 원판 가정을 반영하지 않는다.
측면 영상은 `logs/in-hand-vertical-video/seed-42/side.mp4`에 저장했다.
약 20~33초 구간에서 위아래 회전을 확인할 수 있다. 영상 실행도 4장 검사·적재 성공,
미끄러짐 0회였다. 관련 회전·가림·부분 재생 보호·기존 시뮬레이션 테스트 60개가 통과했다.

## 바닥 전체를 보는 카메라

`configs/simulation/in_hand_full_floor.yaml`은 같은 사과/상자 배치를 유지하고 카메라만
높이 1.2m, 수직 시야각 55도, 작업면에 수직인 자세로 설정한다. Studio의 유한 테이블
X -0.25~0.95m, Y -0.50~0.50m의 네 모서리가 640×480 화면에 모두 들어오는지 시험한다.
무한한 바닥 전체를 뜻하는 설정은 아니다. 바닥면 기준 화면 범위는 약 1.67×1.25m다.
Native MuJoCo 카메라와 합성 판정 카메라의 높이·영상 축·초점거리를 맞췄다.

```bash
~/.venvs/piper-daemons/bin/python3 tools/run_in_hand_simulation.py --quick --seed 42 --optimize --scene-config configs/simulation/in_hand_full_floor.yaml
```

`logs/in-hand-full-floor/seed-42/report.json`의 한 사과 시험은 위아래 회전 4장·적재 성공,
미끄러짐 0회였다. 관측률은 약 0.355로 이전 약 0.370보다 증가하지 않았고 REJECT였다.
넓은 화면은 작업 영역을 더 담지만 카메라 위치의 방향이나 그리퍼 가림을 없애지 않는다.
같은 해상도에서 사과 픽셀 수가 줄어드는 것도 기록해야 한다. 나머지 사과·상자도 장면에
존재하지만 이번 시험의 실행 대상은 첫 물리 사과 1개다.

이 폴더의 assumptions로 가상 simd를 시작하고 아래 명령으로 웹에서 재생한다.

```bash
~/.venvs/piper-daemons/bin/python3 tools/replay_in_hand_studio.py logs/in-hand-full-floor/seed-42 --allow-partial
```

전체 바닥 화면은 `http://127.0.0.1/api/cameras/sim:top/stream`, 측면은
`http://127.0.0.1/api/cameras/sim:side/stream`이다. 가상 초기화 때만 잠깐 범위를 벗어나는
그리퍼 원시 값은 첫 열기 명령에서 최대 1초 유효한 값이 올 때까지 기다린다.
값을 보정/클램프하거나 실기 판독을 완화하지 않고, 정상 명령의 판독 규칙은 유지한다.

웹 시험도 사과 1개·4장 검사 자세·적재를 통과했다(`studio/report.json`).
재생 중 실제 탑뷰 프레임은 `top-live.jpg`에 저장했다. 관련 테스트 64개가 통과했다.

## 실기와 시뮬레이션의 경계 (공용 코드 유지)

무작위 좌표·사과 질량/형상·합성 카메라·맵 중앙·컵 곡률·마찰·핑거 힘·가상 TCP·
임시 속도는 시뮬레이션 설정과 메모리 복사본에만 적용한다. 원본 설정은 덮어쓰지 않는다.
실기 진입점 `tools/run_cycle.py` 및 공용 `aple/execution`, `aple/planning`,
`aple/orchestration`, `piper_motion.py`, `piper_studio.py`는 이번 최적화에서 수정하지 않는다.
실기는 기존 M1이 제공한 좌표와 기존 작업 계획을 사용한다. 시뮬레이션 처리 순서를
실기 계획에 넣거나 실기 입력 변수를 없애지 않았다.

실기에 옮길 때는 경로 알고리즘과 로봇 전송 기능을 따로 검증하고, 실제 TCP/개폐 폭,
캘리브레이션/충돌 공간, 파지력/마찰과 손상, 속도/가감속, 카메라 프레임 시점을 확인해야 한다.
가상 값을 그대로 현장 YAML에 복사하지 않는다. 이번 후보는 실기 활성 설정을 입력받으면
거부한다. 기존 시뮬레이션의 이상적 `KinematicWorld`는 초기 합성 인식 장면 생성에만
사용하며, 사과 이동 성공은 접촉 물리 상태로 별도 판정한다.

## Piper Studio 웹에서 재생

오프라인 물리 시험이 `success: true`인 폴더만 웹에서 재생할 수 있다. 색상 판정은
오프라인 촬영 결과를 사용하고, 웹 재생은 그 결과로 선택한 경로를 실제 Studio 접촉 물리로
다시 수행한다. 웹 카메라로 색상 재판정을 새로 수행하는 기능은 아니다.

먼저 실행 폴더의 `assumptions.yaml`로 가상 simd를 시작해야 한다. 기존 64mm 사과용 컵과
가정값이 다르면 재생 도구가 움직이기 전에 거부한다. 아래 `seed-N`은 실제 완료 폴더로
바꾼다. 진행 중인 웹 조작이 끝난 상태에서만 가상 서비스를 재시작한다.

```bash
systemctl --user stop aple-assumed-simd.service
systemctl --user stop piper-simd.service
systemd-run --user --unit=aple-assumed-simd --collect \
  --property=Restart=on-failure \
  --setenv=MUJOCO_GL=egl \
  --setenv=PIPER_REDIS_URL=redis://127.0.0.1:6379/0 \
  --setenv=PIPER_CONFIG_DIR=/srv/piper-data/config \
  /home/seo/.venvs/piper-daemons/bin/python3 \
  /mnt/extra/aple-pac2026/tools/studio_assumed_simd.py \
  --assumptions /mnt/extra/aple-pac2026/logs/in-hand-simulation/seed-N/assumptions.yaml

~/.venvs/piper-daemons/bin/python3 tools/replay_in_hand_studio.py logs/in-hand-simulation/seed-N
```

`http://127.0.0.1/scene`, 측면 카메라 `http://127.0.0.1/api/cameras/sim:side/stream`에서 본다.
재생은 `sim_follower1`/서버 transport=`sim`만 사용하며 다른 팔이나 기존 jog 세션을
가로채지 않는다. 사과를 붙이는 제약 없이 접촉으로 움직이고 최종 안착을 별도
`studio/report.json`에 기록한다. 재시험은 studio-2처럼 새 폴더로 분리하고, 기존 장면은 각 폴더의 `previous-scene.json`으로 보관한다.
빈손/파지 목표 속도에 맞춰 관절 목표를 보간하되 웹의 필터·도달 대기로 실제 시간은
더 길어질 수 있다. 오프라인 성공이 웹 재생 성공을 보장하지 않으므로 보고서를 확인한다.

## 이번 확인 결과 (seed 42)

`logs/in-hand-simulation/seed-42-17/report.json`의 오프라인 접촉 시험은 성공했다.
세 사과 모두 중앙에서 4장씩 촬영했고 UPPER/MIDDLE/REJECT로 재판정되어 해당 상자에
안착했다. 미끄러짐 중단 0회, 최종 상자 중심 오차는 약 1.0/1.6/1.9mm였다.
모든 사과는 약 0.234~0.236m 높이까지 실제 접촉으로 들렸다.
영상은 `logs/in-hand-simulation/seed-42-17/side.mp4`, 촬영 데이터는 그 폴더의 captures다.

RGB-D/색상은 합성이며 한 seed의 물리 통과 결과다. 여러 배치의 성공률이나 실물 그리퍼의
파지·손상 성능을 입증하지 않는다. 관련 테스트는 레이아웃 재현·겹침 방지·개별 질량,
정확한 중앙 카메라 위치, 사과 ID와 물리 body 매칭, body 회전에 따른 색상 변화,
미끄러짐 주입 시 중단, 실패한 오프라인 시험/다른 모델의 웹 실행 거부를 확인한다.

같은 seed 42의 Studio 웹 접촉 재생도 성공했다. 1,481개 이벤트와 검사 자세 12개를
완료했고 사과 3개 모두 안착했다. 중심 오차는 약 1.9/2.0/2.3mm였다. 웹 보고서는
`logs/in-hand-simulation/seed-42-17/studio/report.json`에 있다. 첫 웹 시험은 더 엄격한
0.005rad 도달 기준에서 멈췄고, 현재는 오프라인과 같은 0.02rad 기준을 사용한다.
이후 별도로 실제 사과의 최종 위치를 확인하므로 목표 명령 승인만으로 성공 처리하지 않는다.

다른 배치 seed 7도 먼 상자용 자세를 보완한 뒤 오프라인 접촉 시험을 통과했다
(`logs/in-hand-simulation/seed-7-3/report.json`). 3개 모두 4장씩 촬영·재판정·안착했고
미끄러짐 0회였다. 처음에는 반경 약 0.486m의 상자에서 160도 적재 자세가 joint5 한계를
넘어 멈췄다. 현재 먼 상자만 150도 자세를 사용하는 가상 분기로 해결했고, 실기 코드는
바꾸지 않았다. 최종 관련 테스트 42개가 통과했다.
