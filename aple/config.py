"""configs/*.yaml 을 하나의 dict 로 읽는다. 최상위 키(workcell, varieties, inspection, ...)는 파일끼리 겹치지 않는다.

set_config_value 는 실측으로 정한 값을 설정 파일에 써 넣는다. YAML 을 다시 쓰면 주석이 사라지므로
해당 줄만 바꾸고, 줄 끝 주석의 `[임시]` 표시는 `[실측 날짜]` 로 바꾼다.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
_KEY_LINE = re.compile(r"^(\s*)([^\s#:][^:#]*?):(.*)$")


def load_config(config_dir: str | Path = CONFIG_DIR) -> dict:
    cfg: dict = {}
    for p in sorted(Path(config_dir).glob("*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        dup = set(cfg) & set(data)
        if dup:
            raise ValueError(f"{p.name}: 최상위 키 중복 {sorted(dup)}")
        cfg.update(data)
    return cfg


def _fmt(value) -> str:
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _split_comment(rest: str) -> tuple[str, str]:
    """'값   # 주석' → ('값', '   # 주석'). 괄호 안의 # 은 주석으로 보지 않는다."""
    depth = 0
    for i, ch in enumerate(rest):
        if ch in "[{":
            depth += 1
        elif ch in "]}":
            depth -= 1
        elif ch == "#" and depth == 0 and (i == 0 or rest[i - 1].isspace()):
            j = i
            while j > 0 and rest[j - 1].isspace():
                j -= 1
            return rest[:j], rest[j:]
    return rest.rstrip(), ""


def set_config_value(path: str | Path, dotted_key: str, value, note: str | None = None) -> str:
    """dotted_key (예: 'varieties.fuji.hue_range_deg') 의 값을 바꾼다. 바뀐 줄을 돌려준다."""
    path = Path(path)
    lines = path.read_text(encoding="utf-8").split("\n")
    want = dotted_key.split(".")
    stack: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m = _KEY_LINE.match(line)
        if not m:
            continue
        indent, key, rest = len(m.group(1)), m.group(2).strip(), m.group(3)
        while stack and stack[-1][0] >= indent:
            stack.pop()
        if [k for _, k in stack] + [key] == want:
            old_val, comment = _split_comment(rest)
            if not old_val.strip():
                raise ValueError(f"{dotted_key} 는 하위 항목을 가진 키다 — 값 하나로 바꿀 수 없다")
            tag = f"[실측 {note or date.today().strftime('%m/%d')}]"
            comment = comment.replace("[임시]", tag) if "[임시]" in comment else comment
            head = f"{m.group(1)}{key}: {_fmt(value)}"
            if comment:
                col = len(m.group(1)) + len(key) + 1 + len(old_val)          # 원래 주석 위치를 되도록 유지
                gap = max(1, col - len(head) + (len(comment) - len(comment.lstrip())))
                head = head + " " * gap + comment.lstrip()
            lines[i] = head
            path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
            return head
        stack.append((indent, key))
    raise KeyError(f"{path.name} 에 {dotted_key} 가 없다")
