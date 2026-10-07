# 실행 환경 기록 (S14-1)

팀 전원이 같은 버전을 쓴다. 바꾸면 여기 먼저 기록한다.

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04 (24.04.5 LTS) |
| 커널 | 7.0.0-31-generic |
| Piper Studio 버전 / 커밋 | **v0.5.12** (10/7 사용자 업그레이드 확인). HAL 조그는 기존 API, 토크 확인은 0.5.12 API 기준; 실제 장비 호환성 확인은 별도 |
| librealsense / pyrealsense2 | pyrealsense2 2.58.4 |
| NVIDIA 드라이버 / CUDA | 580.178.04 / CUDA 13.0 |
| Python | 3.12 (3.12.3) |
| ultralytics | 8.4.173 (torch 2.14.1+cu130) |

값은 2026-10-06 역할 C 노트북(RTX 4050 Laptop 6GB)에서 확인했다. `requirements.txt` 는 아직 버전을 고정하지 않았으므로 `pip install` 시점에 따라 달라질 수 있다.

## Piper Studio 연결

- 웹·API 주소는 `http://127.0.0.1` (80번 포트). 8000번은 호스트에 열려 있지 않다 → `configs/execution.yaml` `piper_web.base_url`
- v0.5.8 부터 게이트웨이 로그인이 있다(설정 → 보안). 로그인이 꺼져 있으면 `aple/hal` 은 로봇 제어를 거부한다. 실기 전에 비밀번호를 정하고 실행 환경에 `PIPER_WEB_PASSWORD` 로 넣는다 (커밋 금지)
- 업데이트: `~/piper-web-deploy/<현재 버전>/piper-install.sh v0.5.12 --pull-only` → `~/piper-web-deploy/v0.5.12/apply.sh`
- 백엔드 컨테이너가 GPU 를 잃는 경우(대시보드 "GPU 없음", 컨테이너 안 `Failed to initialize NVML`): `docker restart piper-web-backend`. Docker cgroup 드라이버가 systemd 일 때 생기는 문제다

## 그리퍼 토크 확인 (v0.5.12)

`python3 tools/check_gripper_torque.py`는 등록된 `piper_web.iface`의 저장된 토크를 GET으로만 읽고
`execution.gripper.fixed_effort`와 비교한다. 장비를 움직이거나 토크를 변경하지 않는다.
실기 제어 opt-in과 별도로 읽기만 수행하며, 설정 파일의 enabled는 바꾸지 않는다.
설정의 1000은 1.0 N·m 명령 단위이며 사과 접촉 힘 1000 N이라는 뜻이 아니다.

HAL은 조그 시작 전, 각 목표 전송 전과 도달 대기 중 토크를 다시 조회한다. 불일치·API 실패·
잘못된 응답이면 명령을 거부하고 이미 시작한 조그는 종료한다. 구버전/지원하지 않는 팔의
404에도 고정 1000으로 추정하지 않는다. 확인은 폴링이므로 조회 직후 외부 UI에서 바꾸는 경쟁을
원자적으로 막지는 못한다. 실행 중에는 Studio 토크 슬라이더를 고정해서 운용한다.
사과별 자동 변경과 N 단위 힘 보정은 구현하지 않았다.
