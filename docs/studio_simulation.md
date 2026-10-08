# Piper Studio 웹에서 APLE 궤적 보기

Studio 0.5.12의 MuJoCo 가상 팔 `sim_follower1`에서 APLE 통합 실행기가 계산한 관절·그리퍼 명령을 재생한다.
이 도구는 **계획된 궤적의 웹 시각화**다. 기본 Studio 핑거와 가상 TCP는 형상이 다르므로 물리 파지 성공을 판정하지 않는다.

수제 핑거를 가정해 사과가 실제 접촉으로 옮겨지는 시험은 [gripper_physics.md](gripper_physics.md)를 사용한다.
아래 기본 핑거 궤적 재생을 사용할 때는 기본 `piper-simd.service`로 복원한다.
가정 그리퍼 simd가 켜져 있으면 해당 문서의 `run_studio_gripper_physics.py`를 사용한다.

## 실행

1. Piper Studio가 실행 중인지 확인한다. 웹 **설정 → 서비스 → 시뮬레이터(simd)** 를 켠다. 이 PC에서는 이미 켜 두었다.
2. 저장소 루트의 터미널에서 실행한다. `PYTHONPATH`는 Piper Studio 저장소의 `robot/` 또는 설치된 `piper_robot` 패키지 경로다.

```bash
cd /mnt/extra/aple-pac2026
PYTHONPATH=/tmp/aple-piper-math python3 tools/run_studio_simulation.py
```

이 PC의 `/tmp/aple-piper-math`는 임시 수학 모듈 경로다. 다른 PC에서는 자기 설치 경로로 바꾼다.
Studio가 다른 주소에 있으면 `--url http://해당주소`를 추가한다. 인증을 쓰면 `PIPER_WEB_PASSWORD` 환경변수를 사용한다. 비밀번호를 명령 인수나 YAML에 넣지 않는다.

3. 브라우저에서 **http://127.0.0.1/scene** 을 연다. 별도 `APLE planned trajectory` 장면에 사과 3개와 등급별 상자 3개를 준비하며, `sim:top`·`sim:front` 가상 카메라를 연결한다. 페이지를 이미 열어 두었다면 새로고침한다.
4. 가상 카메라 프리뷰에서 팔 움직임을 본다. 명령은 터미널 도구가 보내므로 재생 중 웹 조종 세션을 추가로 시작하지 않는다. 조종 창을 이미 사용 중이면 닫은 후 실행한다.

프리뷰만 직접 열 수도 있다.

- 탑뷰: http://127.0.0.1/api/cameras/sim:top/stream
- 정면: http://127.0.0.1/api/cameras/sim:front/stream

장면만 준비하고 팔을 움직이지 않으려면 `--prepare-only`를 쓴다. `--step-delay 0.3`은 명령 사이 최소 표시 시간을 늘린다. 실기 속도 제어와는 다르다.

## 동작·로그

APLE의 장치 없는 전체 사이클을 먼저 검증하고, 계산된 관절·그리퍼 명령을 Studio 가상 팔에서 재생한다. 조그 세션의 소유권·목표 도달을 확인하고 E-stop heartbeat를 유지하며, 완료·오류·Ctrl+C에서 소유한 세션을 종료한다. 대상은 `sim_follower1`로 고정하고 서버의 `transport=sim`을 확인한다. 실기용 토크 확인/안전 조건은 변경하지 않았다. 가상 팔에는 Studio가 제공하지 않는 설정 토크 측정을 적용하지 않는다.

`logs/studio-simulation/`에 다음을 저장한다.

- `planned-trace.json`: 이상적인 기구학 가상 실행의 계획·결과
- `studio-report.json`: Studio에서 도달한 명령 수·재생 상태. `physical_grasp_verified`는 항상 false다.
- `previous-scene*.json`: 적용 전 장면 ID·명세. 기존 장면을 삭제/덮어쓰지 않고 새 장면을 import한다. 이전 환경은 Studio 장면 목록에서 다시 적용할 수 있다.

MuJoCo의 관절 제한·접촉 때문에 목표에 도달하지 못하면 재생을 실패로 기록한다. 기구학 가상 실행의 성공과 Studio 목표 도달은 서로 다른 결과다.

Studio 0.5.12에서 첫 조그 시작 시 명령 스레드가 `ValueError: cannot mmap an empty file`로
멈추는 경우를 이 PC에서 확인했다. 조그를 종료한 상태에서 웹 설정의 simd를 재시작하거나
`systemctl --user restart piper-simd.service`로 **시뮬레이터만** 재시작한 뒤 다시 실행한다.
서비스 재시작 후 연결 상태를 다시 확인하도록 도구에 반영했다. Studio 내부 코드는 수정하지 않았다.

## 범위

- Studio 기본 팔과 기본 핑거에서 **팔의 접근·이송·적재 궤적 및 그리퍼 개폐 명령**을 볼 수 있다.
- 계획된 파지/놓기 이벤트로 Studio 사과를 강제로 이동시키지 않는다. 물리적으로 집히지 않으면 사과는 원래 자리에 남는다.
- 기본 TCP는 가상 0.16 m지만 Studio 기본 핑거의 기준점은 다르다. 따라서 사과 옆에서 핑거가 떠 보일 수 있다. 수제 어댑터의 실측값·MuJoCo 핑거 형상·접촉 힘/마찰을 연결해야 파지 성공률 시험이 된다.
- 합성 RGB-D의 착색비율을 판정한다. Studio 렌더 영상으로 색 판정을 하거나 모드 B·ACT를 실행하지 않는다.

원래 기구학 시뮬레이션 설명은 [simulation.md](simulation.md), 수제 어댑터의 미확정 값은 [gripper_attachment.md](gripper_attachment.md) 참조.

## 이 PC에서 확인한 결과

Studio 0.5.12의 `sim_follower1`에서 팔 이동 80회·그리퍼 명령 9회, 총 89개 목표 도달과
조그 종료를 확인했다. 탑뷰·정면 프리뷰 HTTP 200을 확인했으며,
`logs/studio-simulation/studio-report.json`에 `status: completed`, `goals_reached: 89`가 기록됐다.
이 결과는 궤적 재생 확인이며 사과의 물리 파지·적재 성공 결과가 아니다.
