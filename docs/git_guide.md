# 협업 방법

GitHub가 처음인 팀원도 따라 할 수 있도록 매일 쓰는 명령만 정리했다.

## 0. 처음 한 번

```bash
# 초대 수락: https://github.com/notifications 또는 초대 메일
gh auth login                     # 또는 git 첫 push 때 GitHub 로그인
git clone https://github.com/pikachu515/aple-pac2026.git
cd aple-pac2026
git config user.name  "본인 GitHub 아이디"
git config user.email "본인 GitHub 이메일"

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest                            # 전부 통과하면 준비 완료
```

## 1. 매일 작업 순서

```bash
git switch main
git pull                                   # ① 시작 전에 항상 최신 받기

git switch -c inspection/hue-angle         # ② 작업 브랜치 만들기 (모듈/주제)

# ... 코드 수정 ...

pytest                                     # ③ 테스트 통과 확인
git add aple/inspection/coloration.py      # ④ 바꾼 파일만 골라 담기
git commit -m "착색 픽셀 판정 구현 (색상각 + 채도 하한)"
git push -u origin inspection/hue-angle    # ⑤ 올리기
```

⑥ GitHub 저장소 화면에 뜨는 **Compare & pull request** 버튼 → 설명 작성 → **Create pull request**
⑦ 다른 팀원 1명이 확인하고 **Merge** → 브랜치 삭제

PR을 올리면 GitHub가 자동으로 테스트를 돌린다. PR 화면 아래에 초록 체크(✓)가 뜨면 통과, 빨간 X면 **Details** 를 눌러 실패 내용을 확인하고 고친 뒤 다시 push 한다.

## 2. 약속

**자기 폴더 위주로 고친다.** 다른 모듈 폴더를 고쳐야 하면 담당자에게 먼저 말한다. 같은 파일을 두 사람이 동시에 고치면 충돌이 난다.

| 폴더 | 담당 |
|---|---|
| `aple/perception/`, `aple/hal/calibration.py` | 역할 A |
| `aple/inspection/` | 역할 C |
| `aple/execution/`, `aple/planning/`, `aple/hal/piper_studio.py` | 역할 B |
| `hardware/`, `configs/size_bins.yaml` | 역할 D |
| `aple/orchestration/` | 통합 담당 |
| `aple/schema.py` | **전원 합의 후에만** |

**`aple/schema.py` 를 고쳤으면** 다이어그램을 다시 만들어 같이 올린다. (main 에 합쳐지면 GitHub가 자동으로 다시 만들어 주지만, PR 단계에서는 직접 해야 테스트가 통과한다.)

```bash
python tools/gen_interfaces.py
git add aple/schema.py docs/interfaces.md
```

**main 에 직접 push 하지 않는다.** 무료 계정이라 GitHub가 막아 주지 않으므로 서로 지켜야 한다. (대회 중에는 시간상 예외 허용 — `pytest` 통과 후)

**올리지 않는 것:** 사진·영상·데이터셋·`*.pt` 가중치(공유 드라이브에), API 키·비밀번호(`.env` 에만). `.gitignore` 가 대부분 막아 주지만 `git add .` 대신 파일을 골라 담는 습관을 들인다.

## 3. 로봇 없이 개발하기

로봇은 대회 당일 지급되므로, 그 전에는 D435로 찍어 둔 스냅샷으로 개발한다.

```bash
# D435로 찍기 (Piper Studio 카메라 데몬을 멈춘 상태에서)
python tools/capture_snapshot.py --out data/snapshots/1005_desk --note "형광등, 사과 5개"
```

```python
# 코드에서 읽기 — 현장의 PiperStudio.snapshot() 과 같은 형태(SceneSnapshot)로 나온다
from aple.hal.replay import iter_snapshots

for snap in iter_snapshots("data/snapshots/1005_desk"):
    dets = detect(snap, cfg)          # 역할 A
    ...
```

찍은 폴더는 공유 드라이브에 올려 모두가 같은 데이터로 개발한다. `data/` 폴더는 git에 올라가지 않는다.

## 4. 자주 생기는 문제

| 증상 | 해결 |
|---|---|
| `git pull` 했더니 conflict | 충돌 파일을 열어 `<<<<<<<` ~ `>>>>>>>` 사이를 정리 → `git add 파일` → `git commit` |
| push 가 거절됨 (rejected) | 그 사이 다른 사람이 올림 → `git pull` 후 다시 `git push` |
| 실수로 커밋한 파일 빼기 (push 전) | `git reset HEAD~1` (수정 내용은 남는다) |
| 지금 브랜치 확인 | `git status` 첫 줄 |
