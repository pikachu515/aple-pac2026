# 협업 방법

GitHub가 처음인 팀원도 따라 할 수 있도록 정리했다. 명령어를 외울 필요는 없다. **AI 코딩 도구에게 시키거나(3-A), VS Code 버튼으로 하거나(3-B), 명령어를 직접 쓰는(3-C)** 세 방법 중 편한 것을 쓴다.

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

AI 도구를 쓸 거라면 위 과정도 "docs/git_guide.md 0번 따라서 환경 세팅해 줘"라고 시키면 된다.

## 1. 용어 세 개

| 용어 | 뜻 |
|---|---|
| **main** | 팀 공용 완성본. 항상 돌아가야 하는 코드 |
| **브랜치** | main을 복사한 내 작업 공간. 여기서 망가뜨려도 main은 그대로 |
| **PR (Pull Request)** | "내 브랜치 변경을 main에 합쳐 주세요" 요청. 올리면 GitHub가 자동 테스트를 돌리고, 바뀐 줄만 모아 보여 준다 |

## 2. 규칙 — 가볍게 쓰는 PR

PR은 main이 망가져서 네 명이 동시에 멈추는 일을 막기 위한 것이다. 무료 계정이라 GitHub가 main을 막아 주지 않으므로 이 규칙이 유일한 안전장치다. 대신 절차는 최소로 한다.

| 시기 | 방식 |
|---|---|
| **대회 전** (10/5~10/8) | 브랜치 → PR. **초록 체크(✓)가 뜨면 본인이 바로 Merge 해도 된다** |
| **대회 중** (21시간) | 자기 담당 폴더는 main에 바로 push 해도 된다. 단 `pytest` 통과 후 |
| **항상** | `aple/schema.py` 를 바꿀 때는 팀원 1명 확인 (대회 중에는 말로 합의) |

**자기 폴더 위주로 고친다.** 다른 폴더를 고쳐야 하면 담당자에게 먼저 말한다. 같은 파일을 두 사람이 동시에 고치면 충돌이 난다.

| 폴더 | 담당 | GitHub |
|---|---|---|
| `aple/perception/`, `aple/hal/calibration.py`, `configs/` 카메라·인식·작업셀 | 역할 A (이수빈) | lsb1017 |
| `aple/execution/`, `aple/planning/`, `aple/hal/piper_studio.py`, `configs/` 실행·안전 | 역할 B (서유겸) | dbrua3311 |
| `aple/inspection/`, `configs/varieties.yaml` | 역할 C (변승호) | pikachu515 |
| `hardware/`, `configs/size_bins.yaml` | 역할 D (강우담) | uhdahmrok |
| `aple/orchestration/` | 통합 (전원) | — |
| `aple/schema.py` | **전원 합의 후에만** | — |

PR을 올리면 **바꾼 폴더의 담당자가 자동으로 리뷰어로 지정된다** (본인 제외). 담당이 바뀌면 `tools/request_reviewers.py` 의 표를 고친다.

**`aple/schema.py` 를 고쳤으면** `python tools/gen_interfaces.py` 로 다이어그램을 다시 만들어 같이 올린다. (main 에 합쳐지면 GitHub가 자동으로 다시 만들어 주지만, PR 단계에서는 직접 해야 테스트가 통과한다.)

**올리지 않는 것:** 사진·영상·데이터셋·`*.pt` 가중치(→ 공유 드라이브), API 키·비밀번호(→ `.env`). `.gitignore` 가 대부분 막아 주지만, 올리기 전에 담긴 파일 목록을 한 번 확인한다.

## 3. 매일 작업하는 방법

### 3-A. AI 코딩 도구에게 시키기 (추천)

팀원마다 쓰는 AI가 달라도 된다. 저장소에 `AGENTS.md`·`CLAUDE.md` 가 있어서 **어떤 도구를 쓰든 위 규칙을 자동으로 읽고 따른다.**

| 구독 | 도구 | 설치 · 실행 (Ubuntu) |
|---|---|---|
| Claude Pro / Max | **Claude Code** | `curl -fsSL https://claude.ai/install.sh \| bash` → 저장소 폴더에서 `claude` → Claude 계정 로그인 |
| ChatGPT Plus / Pro | **Codex** | VS Code 확장(Ubuntu VS Code에서 사용 가능) 또는 CLI `npm install -g @openai/codex` → 저장소 폴더에서 `codex` → ChatGPT 계정 로그인 |
| 대화창만 쓸 때 | ChatGPT · Claude 웹 | 아래 말투로 물어보고, 알려 준 명령을 직접 실행 |

설치 명령은 바뀔 수 있으므로 안 되면 각 도구의 공식 안내를 확인한다. 두 도구 모두 VS Code 확장으로도 쓸 수 있다.

**그대로 따라 말하면 되는 문장**

| 상황 | 말하기 |
|---|---|
| 작업 시작 | "main 최신으로 받고 `inspection/hue-angle` 브랜치 만들어 줘" |
| 다 고쳤을 때 | "pytest 돌리고, 통과하면 내가 바꾼 파일만 커밋해서 PR 올려 줘. 올리기 전에 담길 파일 목록 보여 줘" |
| PR 테스트 실패 | "PR 테스트가 실패했어. 원인 찾아서 고쳐 줘" |
| 충돌 났을 때 | "git 충돌 났어. 어떤 파일이 왜 충돌했는지 설명하고, 내 변경이랑 상대 변경 둘 다 살리는 방향으로 정리해 줘" |
| 남의 코드 이해 | "aple/perception/pose_estimator.py 가 뭐 하는 건지 설명해 줘" |

**AI에게 맡길 때 지킬 것**

- **담길 파일 목록을 눈으로 확인**하고 나서 올리게 한다. 사진·가중치·`.env` 가 섞이면 안 된다
- **API 키·비밀번호를 대화창에 붙여 넣지 않는다**
- `git push --force`, `git reset --hard`, 브랜치 삭제는 **시키지 않는다**. AI가 하려고 하면 멈추고 팀에 물어본다
- 다른 사람 폴더나 `aple/schema.py` 를 고치려 하면 멈추게 한다
- AI가 만든 코드도 **내가 설명할 수 있어야** 한다. 발표 질의응답은 담당자가 답한다

### 3-B. VS Code 화면으로

1. 왼쪽 아래 브랜치 이름 클릭 → **새 분기 만들기** → 이름 입력 (예: `inspection/hue-angle`)
2. 코드 수정 → 왼쪽 **소스 제어** 아이콘(가지 모양)
3. 바뀐 파일 옆 **+** 로 담기 → 위 칸에 설명 입력 → **커밋**
4. **변경 내용 게시(Publish Branch)** → 브라우저 GitHub 화면의 **Compare & pull request** → **Create pull request**
5. 초록 체크(✓) 확인 → **Merge pull request**

### 3-C. 명령어로

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

⑥ GitHub 저장소 화면의 **Compare & pull request** → **Create pull request**
⑦ 초록 체크(✓) 확인 → **Merge pull request** → 브랜치 삭제

PR 화면에 빨간 X가 뜨면 **Details** 를 눌러 실패 내용을 확인하고, 고친 뒤 같은 브랜치에 다시 push 한다.

## 4. 로봇 없이 개발하기

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

## 5. 자주 생기는 문제

| 증상 | 해결 |
|---|---|
| `git pull` 했더니 conflict | AI에게 3-A의 "충돌" 문장으로 요청. 직접 할 때는 충돌 파일의 `<<<<<<<` ~ `>>>>>>>` 사이를 정리 → `git add 파일` → `git commit` |
| push 가 거절됨 (rejected) | 그 사이 다른 사람이 올림 → `git pull` 후 다시 `git push` |
| 실수로 커밋한 파일 빼기 (push 전) | `git reset HEAD~1` (수정 내용은 남는다) |
| 지금 브랜치 확인 | `git status` 첫 줄, 또는 VS Code 왼쪽 아래 |
