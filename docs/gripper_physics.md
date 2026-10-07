# 수제 핑거를 가정한 Piper Studio 접촉 시뮬레이션

기존 좌우 이동 핑거에 연장부와 오목한 접촉 패드를 붙인 **가정 모델**이다.
Studio 0.5.12의 MuJoCo에서 마찰·접촉·중력으로 사과를 집고 옮긴다.
사과를 TCP에 고정하거나 파지/놓기 이벤트로 순간이동시키지 않는다.
계획 단계의 이상적인 기구학 결과와 실제 MuJoCo 사과 위치를 구분한다.

사진의 48.144 / 48.464 / 71.76 / 79.76은 모두 **mm 단위의 표시점 사이 거리**로
`configs/gripper_attachment.yaml`에 보존했다. 이 거리만으로 장착 좌표계·파지 중심·
컵 곡률을 확정할 수 없어, 현재 물리 모델은 사진의 정확한 CAD 재현이 아니다.

## 이 PC에서 다시 보기

현재 `aple-assumed-simd.service`가 가정 모델을 제공한다.
브라우저에서 http://127.0.0.1/scene 을 열고, 저장소 루트에서 실행한다.
재생 중에는 별도 웹 조종 세션을 시작하지 않는다.

```bash
cd /mnt/extra/aple-pac2026
~/.venvs/piper-daemons/bin/python3 tools/run_studio_gripper_physics.py
```

`sim:front`·`sim:top`·`sim:side` 카메라를 자동 연결하고 별도 장면을 만든다. 기존 장면은
삭제하지 않으며, 적용 전 명세는 `logs/studio-gripper-physics/previous-scene*.json`에 저장한다.
프리뷰를 직접 열려면 http://127.0.0.1/api/cameras/sim:front/stream 을 사용한다.
측면은 http://127.0.0.1/api/cameras/sim:side/stream 에서 볼 수 있다.
가정 모델은 `sim:side`를 추가하며 실행 도구가 자동 연결한다.
`report.json`에는 실제 물리 좌표·최대 상승 높이·상자 안착·중심 오차를 따로 기록한다.
중심 12 mm 정확도 기준을 넘더라도 상자에 안착했다면 이송 성공이며,
`center_accurate: false`를 숨기지 않는다. 상자 안착은 벽 높이에서 사과 단면이
상자 내부에 들어오는지와 최종 높이로 검사한다.

전체 동작 영상은 `logs/studio-gripper-physics/complete-transfer.mp4`에 저장했다.
영상·사진·로그 데이터는 `.gitignore` 대상이며 커밋하지 않는다.

## 다른 세션에서 가정 모델 시작하기

Studio 설치의 simd·MuJoCo 패키지를 가진 Python 환경을 사용한다.
현재 PC에서는 아래 명령을 쓴다. 조그·텔레옵·학습 등 Studio 작업이 끝난 상태에서 실행한다.

```bash
systemctl --user stop piper-simd.service
systemd-run --user --unit=aple-assumed-simd --collect \
  --property=Restart=on-failure \
  --setenv=MUJOCO_GL=egl \
  --setenv=PIPER_REDIS_URL=redis://127.0.0.1:6379/0 \
  --setenv=PIPER_CONFIG_DIR=/srv/piper-data/config \
  /home/seo/.venvs/piper-daemons/bin/python3 \
  /mnt/extra/aple-pac2026/tools/studio_assumed_simd.py
```

`Unit aple-assumed-simd.service already exists`가 나오면 새로 만들지 않고
`systemctl --user restart aple-assumed-simd.service`를 실행한 뒤 시험 도구를 실행한다.
서비스 재시작은 진행 중인 조종·시험이 끝난 상태에서만 한다.

설치된 Studio 파일은 수정하지 않는다. 모델 생성과 0.5.12의 빈 action mmap 재시도는
임시 simd 프로세스 안에만 적용한다. 모델의 가정값 해시를 Studio 실제 물체 응답에서
확인한 뒤 움직이므로, 기본 모델이 떠 있으면 물리 시험을 거부한다.
조종 대상은 `sim_follower1`, 서버 transport는 `sim`으로 고정·검증한다.

기본 시뮬레이터로 복원하려면:

```bash
systemctl --user stop aple-assumed-simd.service
systemctl --user start piper-simd.service
```

그 뒤 Studio 장면 목록에서 이전 장면을 적용하고 가상 카메라를 다시 연결한다.

## 가정값과 변경 이유

모든 가정값은 `configs/simulation/gripper_physics.yaml`에 둔다. 실기 설정에 저장하지 않는다.

| 값 | 현재 가정 |
| --- | --- |
| 플랜지 → TCP | +Z 0.16 m |
| 패드 접촉면 간격 | 0.044–0.112 m |
| 기존 핑거 이동량 | 각 0.034 m (설치된 Studio 모델 기준) |
| 오목한 컵 곡률 반경 | 0.035 m |
| 사과 | 반경 0.032 m, 질량 0.15 kg, 구형 |
| 패드 미끄럼 마찰계수 | 0.8 |
| 핑거 힘 제한 | 각 5 N, 선형 액추에이터 힘 (실물 토크 Nm와 다름) |
| 들어올리기 / 적재 여유 | 0.08 m / 0.07 m |
| 개방 여유 | 사과 직경 +0.020 m |

기존 낮은 들어올리기와 관절 직선 보간은 옆 사과와 부딪혔다. 이 가상 셀에서는
TCP 경로를 0.01 m 간격으로 나누어 각 구간에 IK·관절 경로 검사를 적용한다.
`execution.trajectory.max_cartesian_step_m`은 기본 null이며 기존 실기 동작을 바꾸지 않는다.
joint 5 한계 때문에 가상 파지 방향은 수직에서 20도 기울였다. 좁게 열면 오목한 패드가
사과를 끌고 가므로 개방 여유도 늘렸다. 가상 트레이·핑거 외곽 치수는 계획용 가정이다.

## 확인 결과와 해석

첫 Studio 시험은 관절 227회·그리퍼 9회, 총 236개 목표에 도달했다.
사과 3개 모두 물리적으로 들렸고 MIDDLE·REJECT·UPPER의 해당 상자 안에 안착했다.
상자 중심 오차는 약 7 / 5 / 16 mm였고 마지막 사과는 중심 12 mm 기준을 넘었다.
처음 반복할 때 접촉으로 닫기 목표 폭에 도달하지 못해 중단됐다. 이후 가상 시험에서는
힘 제한 상태에서 닫기는 접촉을 허용하고, 개방은 목표 도달을 확인하도록 분리했다.
팔 도달 허용치도 0.005 rad로 줄였다. 수정 후 반복 시험은 3개 모두 안착했고
중심 오차는 약 8 / 5 / 4 mm로 12 mm 기준도 모두 통과했다.
부품을 제거한 기본 핑거로 같은 오프라인 경로를 실행한 비교 시험은 사과를 들지 못했다.
전체 녹화 시험에서도 236개 목표와 3개 안착을 확인했으며 중심 오차는 약 6 / 5 / 7 mm였다.
이 반복 결과만으로 통계적인 성공률을 주장하지 않는다.

동일한 모델의 오프라인 MuJoCo 시험에서도 3개 모두 안착했으며 중심 오차는 약 14 mm였다.
Studio는 실제 시간의 조그 필터·도달 대기를 거치므로 오프라인 결과와 수치가 다르다.

이 결과는 이 가정·배치에서 이송이 가능함을 보인다. 실제 성공률·사과 손상·수제 부품의
정확한 충돌 형상을 검증한 결과는 아니다. 실측 TCP, 장착 방향, CAD/STL, 개폐 간격,
질량과 마찰·힘을 넣어 모델을 교체한 뒤 실기와 비교해야 한다.
색 판정은 합성 RGB-D를 사용한다. Studio 카메라를 통한 실물 판정이나 모드 B 시험은 아니다.

웹 없이 접촉 시험만 실행할 수도 있다.

```bash
~/.venvs/piper-daemons/bin/python3 tools/test_gripper_physics.py \
  --trace logs/studio-gripper-physics/planned-trace.json
```
