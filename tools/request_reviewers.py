"""PR이 바꾼 폴더의 담당자를 리뷰어로 자동 지정한다 (.github/workflows/reviewers.yml 에서 실행).

무료 계정 비공개 저장소에서는 GitHub CODEOWNERS 가 동작하지 않아 이 스크립트로 대신한다.
담당이 바뀌면 OWNERS 표만 고친다. PR 작성자 본인은 제외된다.

    python tools/request_reviewers.py                       # Actions: 환경변수 PR, AUTHOR 사용
    python tools/request_reviewers.py --dry-run a.py b.py   # 로컬 확인: 파일 경로를 직접 넣는다
"""

from __future__ import annotations

import os
import subprocess
import sys

A, B, C, D = "lsb1017", "dbrua3311", "pikachu515", "uhdahmrok"
ALL = (A, B, C, D)

# (경로 접두어, 담당자) — 위에서부터 모두 검사한다
OWNERS: list[tuple[str, tuple[str, ...]]] = [
    ("aple/schema.py", ALL),                 # 데이터 계약은 전원
    ("aple/orchestration/", ALL),            # 통합 영역
    ("aple/perception/", (A,)),
    ("aple/hal/calibration.py", (A,)),
    ("configs/camera_top.yaml", (A,)),
    ("configs/perception.yaml", (A,)),
    ("configs/yolo.yaml", (A,)),
    ("configs/workcell.yaml", (A,)),
    ("aple/inspection/", (C,)),
    ("configs/varieties.yaml", (C,)),
    ("aple/execution/", (B,)),
    ("aple/planning/", (B,)),
    ("aple/hal/piper_studio.py", (B,)),
    ("configs/execution.yaml", (B,)),
    ("configs/safety.yaml", (B,)),
    ("configs/orchestration.yaml", (B,)),
    ("hardware/", (D,)),
    ("configs/size_bins.yaml", (D,)),
]


def owners_for(paths: list[str], author: str) -> list[str]:
    found: list[str] = []
    for path in paths:
        for prefix, people in OWNERS:
            if path == prefix or path.startswith(prefix):
                found += [p for p in people if p not in found]
    return [p for p in found if p.lower() != author.lower()]


def main() -> int:
    if "--dry-run" in sys.argv:
        paths = [a for a in sys.argv[1:] if a != "--dry-run"]
        print(owners_for(paths, os.environ.get("AUTHOR", "")))
        return 0

    pr, author = os.environ["PR"], os.environ["AUTHOR"]
    out = subprocess.run(["gh", "pr", "view", pr, "--json", "files", "--jq", ".files[].path"],
                         check=True, capture_output=True, text=True).stdout
    reviewers = owners_for(out.split(), author)
    if not reviewers:
        print("담당자 지정 대상 없음")
        return 0
    subprocess.run(["gh", "pr", "edit", pr, "--add-reviewer", ",".join(reviewers)], check=True)
    print("리뷰 요청:", ", ".join(reviewers))
    return 0


if __name__ == "__main__":
    sys.exit(main())
