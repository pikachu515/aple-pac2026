# AI 코딩 도구 작업 규칙

이 저장소에서 작업하는 AI 코딩 도구(Codex, Claude Code 등)는 아래 규칙을 따른다.
사람용 상세 안내는 `docs/git_guide.md`, 구조 설명은 `README.md` 와 `docs/architecture.md` 에 있다.

## 프로젝트

PAC 2026 사과 착색 등급 판정·적재 시스템. Python 3.12, Ubuntu 24.04, 로봇·카메라는 Piper Studio를 통해 접근한다.
작업 전 `README.md` 와 `docs/architecture.md` 를 읽는다.

## 반드시 지킬 것

- 작업은 브랜치에서 한다. 대회 전(10/8까지)에는 main 에 직접 push 하지 않고 PR 로 올린다
- 커밋 전에 `pytest` 를 실행하고 통과를 확인한다
- 커밋에 담을 파일 목록을 사용자에게 먼저 보여 주고 확인받는다. `git add .` / `git add -A` 를 쓰지 않고 파일을 지정한다
- 사용자의 담당 폴더 밖을 고쳐야 하면 고치기 전에 사용자에게 알린다 (담당표: `docs/git_guide.md` 2절)
- `aple/schema.py` 는 팀 합의 없이 바꾸지 않는다. 바꿨다면 `python tools/gen_interfaces.py` 를 실행하고 `docs/interfaces.md` 를 함께 커밋한다
- `docs/interfaces.md` 는 자동 생성 파일이므로 직접 고치지 않는다
- 현장에서 측정하는 값은 코드 상수로 넣지 않고 `configs/*.yaml` 에 둔다
- 단위는 SI (m, N, s, rad). mm 는 설정 파일 키(`_mm`)에만 쓴다. 위치는 base_link 기준

## 하지 않을 것

- `git push --force`, `git reset --hard`, 브랜치 삭제, 커밋 기록 수정
- 사진·영상·데이터셋·`*.pt`·체크포인트·`.env`·API 키를 커밋하는 것
- 로봇 CAN 이나 카메라 장치를 직접 여는 코드 (Piper Studio 가 점유한다 — `aple/hal` 을 통한다)
