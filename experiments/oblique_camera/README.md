# 대각선 카메라 설치 검토 — 기존 코드 유지

**보관용 비교:** 사용자 결정으로 위아래 회전 계획을 폐지하고 기존
옆 회전으로 복귀했다. 아래 수치는 당시 위아래 회전 4개 자세의 결과이며,
현재 옆 회전 경로의 관측률로 사용하면 안 된다.

결론: **정상 깊이와 재캘리브레이션이 있으면 대각선 카메라에서도 M1의
3D 계산을 사용할 수 있다. 그러나 현재 코드를 그대로 실기에 적용해도
문제없다고 보장할 수는 없다. 카메라만 기울여서는 현재의 좁은 위아래
회전 경로의 관측 범위가 늘어나지 않았다.**

이 실험은 새 폴더에만 추가했다. 기존 자유 XYZ 실험, 고정 중심 실험,
Piper Studio 서비스, 실기 소스·설정은 수정하지 않았다. 새로운 장치에
연결하지 않고 기존 독립 가상환경의 Python을 사용한다.

## GitHub에서 확인한 코드

`git pull --ff-only origin main`으로 최신 main을 확인했다. 기준 커밋은
`7217ca8`이며 가져올 추가 변경은 없었다. 역할 A 이수빈(`lsb1017`)의
[PR #13](https://github.com/pikachu515/aple-pac2026/pull/13)은 병합됐고,
병합 커밋은 `0ae9f144ce5c17b7f43d804672277885d2e288ca`다. 검토는 이 PR만의
옛 스냅샷이 아니라 이후 main에 반영된 수정까지 포함한 현재 코드로 했다.

| 코드 | 대각선 설치에 대한 판단 |
| --- | --- |
| `aple/hal/calibration.py` | 픽셀·깊이를 카메라 3D로 역투영하고, 회전+이동을 가진 `T_base_camera`로 base_link에 변환한다. 수직 설치를 강제하지 않는다. |
| `aple/perception/geometry.py` | 평면 RANSAC과 실루엣 원뿔+깊이 구 피팅은 일반 카메라 방향을 계산할 수 있다. |
| `aple/perception/pose_estimator.py: _measure` | 정상 깊이로 구 중심을 계산한 뒤 base_link로 변환한다. |
| `aple/perception/pose_estimator.py: _center_from_support` | 깊이가 부족하면 주변 받침면의 카메라 z를 이용한다. `D=z_s/(c_z+sinα)`는 받침면의 기울기를 반영하지 않아 대각선에서는 오차가 커진다. |
| `aple/perception/detector.py` | ROI를 캘리브레이션으로 투영하고 평면 위 높이를 구한다. 다만 모폴로지·원형도·화면 경계·가림이 실제 마스크와 중심 오차에 영향을 준다. |
| `aple/perception/box_occupancy.py` | 깊이 점을 base_link로 변환한 후 칸의 XY와 z 높이를 사용하므로 계산 자체는 방향에 독립적이다. 상자 벽·사과의 가림과 관측하지 못한 칸은 별도 문제다. |
| `aple/perception/pose_estimator.py: top_z` | 보이는 표면 점의 base z 95분위수를 사용한다. 대각선에서는 꼭대기 관측이 달라져 높이를 실제 꼭대기보다 낮게 잡을 수 있다. |
| `aple/inspection/inhand.py` | 카메라–TCP 회전과 이동을 일반 4×4 변환으로 계산한다. 카메라를 옮기면 동일한 새 변환을 M1·M2에 공유해야 한다. |

`Source.TOP`과 `judge_camera: top`은 현재 판정용 D435 슬롯 이름으로 쓰인다.
카메라를 비스듬히 놓았다고 RGB 전용 웹캠 슬롯으로 교체할 수 있는 것은
아니다. D435의 정렬된 컬러·깊이·내참수를 유지해야 한다.

## 측정 조건

- 카메라는 로봇 base_link의 +X 쪽, 맵 중심 `[0.325,0,0]`을 향한다.
- 표의 각도는 **수직 아래 방향에서 기울인 각도**다. 30도는 바닥과
  60도, 45도는 바닥과 45도, 60도는 바닥과 30도로 바라본다.
- 맵 중심까지 거리 1.2m, 수직 화각 55도, 해상도 640×480을 유지했다.
  카메라 높이는 각각 1.200 / 1.039 / 0.849 / 0.600m이다.
- 무작위 배치 seed 42와 7, 사과 3개·상자 3개, 기존 불투명 핑거 모델을
  사용했다. 인식 감사는 MuJoCo 렌더링 RGB-D로 수행해 로봇·상자 가림을
  포함했다. 깊이는 이상적인 시뮬레이션 깊이이며 실제 D435의 결측·잡음·광택은
  재현하지 않는다.
- 학습된 YOLO 가중치는 설정돼 있지 않아, 기존 깊이·모양 검출을 실행했다.
  YOLO 모델의 시점 변화에 대한 성능 검증을 한 것은 아니다.
- 정상 검출 결과와 별도로, 렌더러에서 얻은 정답 가시 마스크를 사용해
  구 피팅만 진단했다. 이 정답 마스크를 실제 검출기 입력에 섞지 않았다.

## M1 위치 인식 결과

두 배치에 사과 3개씩이 있다. 중심 오차는 카메라 좌표가 아닌 base_link에서
가상 실제 중심과의 3D 거리다. 아래 표의 오차는 가독성을 위해 mm로 표시한다.

| 수직에서 기울임 | 검출 수 (seed 42 / 7) | 실제 검출→3D 최대 오차 | 정답 마스크→3D 최대 오차 | 사과 깊이 제거 시 대체 계산 최대 오차 |
| --- | --- | --- | --- | --- |
| 0도 | 3 / 2 | 1.8mm | 1.8mm | 1.9mm |
| 30도 | 3 / 3 | 4.6mm | 1.9mm | 11.0mm |
| 45도 | 3 / 3 | 9.7mm | 2.1mm | 26.4mm |
| 60도 | 3 / 3 | 18.9mm | 2.1mm | 60.4mm |

정답 마스크로는 모두 약 2mm 이내이므로 일반적인 좌표 변환·구 피팅이
대각선에서 막히는 구조는 아니다. 그러나 검출 마스크가 달라지면 실제 중심
오차가 커질 수 있다. 또한 수직 배치에서도 seed 7의 사과 1개는 검출되지
않았다. 어느 배치가 항상 우월하다는 통계적 결론을 내릴 표본 수는 아니다.

깊이가 없는 사과에는 현재 대체 계산이 confidence 0.5를 부여할 수 있다.
대각선에서 오차가 커져도 신뢰도만으로 충분히 차단된다고 가정하면 안 된다.
이 경로를 기울어진 받침면을 고려하도록 수정하거나, 대각선 운영에서는
정상 깊이를 확보하고 깊이 부족 시 실행을 거부하는 조건이 필요하다.

이번 **빈 상자** 24개 관측에서는 모두 `valid=True`, 잘못 점유 판정한 칸은
0개였다. 쌓인 사과·높은 벽·다층 적재·팔에 가려진 상자의 검증은 아니다.

결과: `logs/oblique-camera-perception/report.json`.
카메라 영상: 같은 폴더의 `seed-42-tilt-30.png`, `seed-42-tilt-45.png` 등.

## 손에 든 사과의 관측 비율

기존 고정 중심 검사에서 물리 접촉으로 사과 1개를 집어 4개 위아래 자세로
돌렸다. 각 촬영 순간의 **동일한 실제 물리 상태**를 0/30/45/60도 카메라로
다시 관측했다. 비교 카메라 사이에서는 물리 시간이 진행되지 않으며,
사과·로봇의 자세나 사과 표면 색 패턴을 변경하지 않는다. 다른 각도의
카메라를 동시에 합산하지 않고, 설치 각도 하나씩 독립 평가한다.

가시성 계산에는 실제 물리 사과의 중심·반지름을 사용하고, 기존 M2
`accumulate_views`에 동일한 마스크 침식·경사 기준·구면 칸 설정을 적용했다.
불투명 핑거·팔·상자에 가린 픽셀은 광선 검사로 제외했다. 이 표는
**관측 가능한 표면의 비율**이며 색 판정 정확도나 실기 성공률이 아니다.

| 수직에서 기울임 | seed 42 | seed 7 | 두 실행 평균 |
| --- | --- | --- | --- |
| 0도 | 36.2% | 35.7% | 35.9% |
| 30도 | 35.2% | 35.2% | 35.2% |
| 45도 | 35.2% | 34.5% | 34.8% |
| 60도 | 34.8% | 34.0% | 34.4% |

기존 실행 파이프라인의 수직 카메라 재판정은 seed 42에서 35.5%였다.
표는 카메라 비교에 실제 구 중심을 사용하므로 기존 FK 예측 중심으로
계산한 파이프라인 수치와 조금 다르다. 비교 표 안에서는 같은 계산 조건이다.

현재 4개 자세는 약 50도의 좁은 위아래 회전 범위를 사용한다. 카메라를
기울이면 주로 보이는 면의 위치가 바뀌며, 관측한 면의 합집합이 크게
늘어나지는 않았다. 별도 자유 XYZ 실험의 79.8%는 11개 다른 자세를
사용한 결과이므로 이 표와 카메라만 바꾼 효과로 비교하면 안 된다.

결과: `logs/oblique-camera-contact/seed-42/report.json`과 `seed-7/report.json`의
`camera_comparison`. 두 실행 모두 수직 카메라로 진행한 기존 집기·4뷰·리젝트
적재가 성공했다. **대각선 카메라로 전체 인식→집기→재판정→적재를 연결한
성공률 시험은 아니다.**

## 적용 전에 필요한 것

30도 정도를 첫 실측 후보로 삼을 수 있지만, 이번 결과만으로 실기 기본
배치를 바꾸지는 않는다. 팀원이 확인해야 할 사항은 다음과 같다.

- 역할 A: 장착 후 `workcell.T_base_camera` 재측정, D435 내참수·RGB-D 정렬,
  전 작업 영역 중심·직경 오차와 가림, 깊이 결측 대체 계산, 빈/찬 상자 점유·높이 확인.
- 역할 B: 새 좌표에서 집기·팔 가림·상자 이동 검증, 측정 오차와 맞는 접근
  여유·실행 거부 조건 확인. 카메라를 옮겼는데 예전 변환으로 집으면 안 된다.
- 역할 C: `top_view_bias`, `top_view_error`, 색/조명 보정과 마스크 설정 재검증.
  기존 탑뷰 실측 오차·치우침을 대각선 영상에 그대로 적용할 수 없다.
- 역할 D: 실제 불투명 그리퍼 기하·TCP·파지 중 안정성 확인.

기존 전략은 D435 탑뷰를 기본으로 한다. 대각선 설치는 아직 비교 실험이며
팀 전략이나 실기 설정을 변경한 것이 아니다.

## 재실행

```bash
# 실제 렌더링 깊이로 M1 코드 검토 (결과 폴더가 있으면 -2 등을 붙임)
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  experiments/free_inspection/.venv/bin/python3 \
  experiments/oblique_camera/audit_perception.py

# 같은 접촉 물리 검사 상태에서 카메라 각도별 관측 비교
experiments/free_inspection/.venv/bin/python3 \
  experiments/oblique_camera/run.py --quick --seed 42 --optimize \
  --scene-config configs/simulation/in_hand_full_floor.yaml

# 카메라 변환·기존 대체 계산 특성·물리 상태 보존 검증
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 \
  experiments/free_inspection/.venv/bin/python3 -m pytest \
  experiments/oblique_camera/test_camera.py -q
```

공용 실기 코드나 설정을 수정하지 않았으며, 원래 시뮬레이션으로 돌아가기
위해 서비스를 재시작하거나 파일을 복구할 필요가 없다.

## 각도별 동영상

동일한 seed 42의 기존 고정 중심 집기·4뷰·적재를 다시 실행하면서
한 물리 상태를 네 카메라 각도에서 동시에 녹화한다. 영상마다 별도로
로봇 동작을 계산하거나 사과를 강제로 고정하지 않는다. 녹화용 카메라
자세는 렌더링 직후 복원하므로 원래 M1/M2 카메라 설정은 유지된다.

```bash
# 각도별 영상을 다시 생성 (기존 폴더가 있으면 seed-42-2 등에 저장)
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  experiments/free_inspection/.venv/bin/python3 \
  experiments/oblique_camera/record_videos.py --seed 42

# 첫 녹화 결과를 재생
xdg-open /mnt/extra/aple-pac2026/logs/oblique-camera-videos/seed-42/camera-00deg.mp4
xdg-open /mnt/extra/aple-pac2026/logs/oblique-camera-videos/seed-42/camera-30deg.mp4
xdg-open /mnt/extra/aple-pac2026/logs/oblique-camera-videos/seed-42/camera-45deg.mp4
xdg-open /mnt/extra/aple-pac2026/logs/oblique-camera-videos/seed-42/camera-60deg.mp4

# 동영상 폴더 열기
xdg-open /mnt/extra/aple-pac2026/logs/oblique-camera-videos/seed-42
```

전체 바닥 시야 비교를 위해 640×480, 3fps로 저장한다. `video-report.json`에
프레임 수·영상 길이·실행 성공 여부를 기록하고, 상세 물리 실행 결과는
같은 폴더의 `physical-run/seed-42/report.json`에 저장한다.
