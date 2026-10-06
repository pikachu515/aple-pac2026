# 실행 환경 기록 (S14-1)

팀 전원이 같은 버전을 쓴다. 바꾸면 여기 먼저 기록한다.

| 항목 | 값 |
|---|---|
| OS | Ubuntu 24.04 (24.04.5 LTS) |
| 커널 | 7.0.0-31-generic |
| Piper Studio 버전 / 커밋 | **v0.5.9** (`aple/hal` 이 이 버전 API 기준). v0.5.10 이 나와 있으나 대회 전에는 올리지 않는다 |
| librealsense / pyrealsense2 | pyrealsense2 2.58.4 |
| NVIDIA 드라이버 / CUDA | 580.178.04 / CUDA 13.0 |
| Python | 3.12 (3.12.3) |
| ultralytics | 8.4.173 (torch 2.14.1+cu130) |

값은 2026-10-06 역할 C 노트북(RTX 4050 Laptop 6GB)에서 확인했다. `requirements.txt` 는 아직 버전을 고정하지 않았으므로 `pip install` 시점에 따라 달라질 수 있다.

## Piper Studio 연결

- 웹·API 주소는 `http://127.0.0.1` (80번 포트). 8000번은 호스트에 열려 있지 않다 → `configs/execution.yaml` `piper_web.base_url`
- v0.5.8 부터 게이트웨이 로그인이 있다(설정 → 보안). 로그인이 꺼져 있으면 `aple/hal` 은 로봇 제어를 거부한다. 실기 전에 비밀번호를 정하고 실행 환경에 `PIPER_WEB_PASSWORD` 로 넣는다 (커밋 금지)
- 업데이트: `~/piper-web-deploy/<현재 버전>/piper-install.sh v0.5.9 --pull-only` → `~/piper-web-deploy/v0.5.9/apply.sh`
- 백엔드 컨테이너가 GPU 를 잃는 경우(대시보드 "GPU 없음", 컨테이너 안 `Failed to initialize NVML`): `docker restart piper-web-backend`. Docker cgroup 드라이버가 systemd 일 때 생기는 문제다
