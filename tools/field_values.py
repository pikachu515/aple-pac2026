"""현장 실측 값 관리 — 아직 안 정해진 값 목록, 한 줄 변경, 되돌리기. 공통.

시뮬레이션·합성 장면 결과는 "코드가 의도대로 이어졌는가"만 보증한다. 임계값·오차·오프셋 같은 수치는
전부 현장에서 다시 잰다. 이 도구로 남은 값을 보고, 잰 값을 설정 파일에 바로 써 넣고, 틀리면 되돌린다.

    python tools/field_values.py list                      # 아직 현장 값이 아닌 것 ([임시] · PLACEHOLDER · null)
    python tools/field_values.py list --owner C            # 역할별
    python tools/field_values.py set workcell.tcp_offset_m "[0, 0, 0.135]" --note "캘리퍼스"
    python tools/field_values.py undo                      # 마지막 변경 되돌리기
    python tools/field_values.py history                   # 현장 변경 이력 (발표 근거)

set 은 그 줄만 바꾸고(주석 유지) 표시를 [실측 MM/DD] 로 바꾼다. 바꾼 뒤 설정 전체를 다시 읽어 깨지면 원상복구한다.
모든 변경은 logs/field_changes.csv 에 남는다 (시각, 키, 이전 값, 새 값, 메모).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from functools import reduce
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from aple.config import CONFIG_DIR, _KEY_LINE, _split_comment, load_config, set_config_value  # noqa: E402

LOG = ROOT / "logs" / "field_changes.csv"
_OWNER = re.compile(r"역할\s*([A-D](?:\s*[·,]\s*[A-D])*)|(통합 담당|공통)")


@dataclass(frozen=True)
class Pending:
    file: str
    key: str
    value: str
    reason: str          # "임시" | "PLACEHOLDER" | "null"
    comment: str
    owner: str


def _owner(path: Path) -> str:
    head = path.read_text(encoding="utf-8").split("\n", 1)[0]
    m = _OWNER.search(head)
    if not m:
        return "?"
    return m.group(1).replace(" ", "") if m.group(1) else "공통"


def _reason(value: str, comment: str) -> str | None:
    if "[임시]" in comment:
        return "임시"
    if "PLACEHOLDER" in comment or "PLACEHOLDER" in value:
        return "PLACEHOLDER"
    if re.search(r"(^|[\s{,:\[])null\b", value):
        return "null"
    return None


def scan(config_dir: Path = CONFIG_DIR) -> list[Pending]:
    """아직 현장 값이 아닌 줄. 들여쓰기로 키 경로를 따라간다 (set_config_value 와 같은 방식)."""
    out = []
    for path in sorted(Path(config_dir).glob("*.yaml")):
        owner = _owner(path)
        stack: list[tuple[int, str]] = []
        items: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").split("\n"):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            if line.lstrip().startswith("- "):                       # 리스트 항목: 부모 키[번호]
                parent = ".".join(k for _, k in stack)
                idx = items[parent] = items.get(parent, -1) + 1
                value, comment = _split_comment(line.lstrip()[2:])
                reason = _reason(value, comment)
                if reason:
                    out.append(Pending(path.name, f"{parent}[{idx}]", value.strip(), reason,
                                       comment.strip().lstrip("#").strip(), owner))
                continue
            m = _KEY_LINE.match(line)
            if not m:
                continue
            indent, key, rest = len(m.group(1)), m.group(2).strip(), m.group(3)
            while stack and stack[-1][0] >= indent:
                stack.pop()
            value, comment = _split_comment(rest)
            value = value.strip()
            dotted = ".".join([k for _, k in stack] + [key])
            stack.append((indent, key))
            if not value:
                continue
            reason = _reason(value, comment)
            if reason:
                out.append(Pending(path.name, dotted, value, reason, comment.strip().lstrip("#").strip(), owner))
    return out


def _file_for(key: str, config_dir: Path) -> Path:
    top = key.split(".")[0]
    for path in sorted(Path(config_dir).glob("*.yaml")):
        if top in (yaml.safe_load(path.read_text(encoding="utf-8")) or {}):
            return path
    raise KeyError(f"설정에 {top} 가 없다")


def _get(cfg: dict, key: str):
    try:
        return reduce(lambda d, k: d[k], key.split("."), cfg)
    except (KeyError, TypeError):
        raise KeyError(f"설정에 {key} 가 없다 (오타 확인)") from None


def _set_in_flow(path: Path, key: str, value, tag: str) -> str:
    """'parent: {a: 1, b: 2}' 처럼 한 줄 { } 안에 있는 값을 바꾼다."""
    *parents, leaf = key.split(".")
    lines = path.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(lines):
        m = _KEY_LINE.match(line)
        if m and m.group(2).strip() == parents[-1] and "{" in m.group(3):
            new, n = re.subn(rf"(\b{re.escape(leaf)}:\s*)([^,}}]+)", lambda mm: mm.group(1) + _fmt(value), line, count=1)
            if n:
                if "PLACEHOLDER" in new:
                    new = re.sub(r"PLACEHOLDER:?", tag, new, count=1)
                lines[i] = new
                path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
                return new
    raise KeyError(f"{path.name} 에서 {key} 줄을 찾지 못했다")


def _fmt(value) -> str:
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return "null" if value is None else f"{value:g}" if isinstance(value, float) else str(value)


def set_value(key: str, raw: str, note: str = "", *, config_dir: Path = CONFIG_DIR, log: Path = LOG) -> str:
    """현장 값 쓰기 → 바뀐 줄. 설정이 깨지면 원상복구하고 오류."""
    old = _get(load_config(config_dir), key)
    value = yaml.safe_load(raw)
    path = _file_for(key, config_dir)
    backup = path.read_text(encoding="utf-8")
    tag = f"[실측 {datetime.now().strftime('%m/%d')}]"
    try:
        try:
            line = set_config_value(path, key, value, note=None)
            if "PLACEHOLDER" in line.split("#", 1)[-1]:
                text = path.read_text(encoding="utf-8").replace(line, re.sub(r"PLACEHOLDER:?", tag, line, count=1))
                path.write_text(text, encoding="utf-8", newline="\n")
                line = re.sub(r"PLACEHOLDER:?", tag, line, count=1)
        except (KeyError, ValueError):
            line = _set_in_flow(path, key, value, tag)
        if _get(load_config(config_dir), key) != value:
            raise ValueError(f"{key} 를 {value!r} 로 쓰지 못했다")
    except Exception:
        path.write_text(backup, encoding="utf-8", newline="\n")
        raise
    log.parent.mkdir(parents=True, exist_ok=True)
    new_file = not log.exists()
    with log.open("a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["time", "file", "key", "old", "new", "note"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), path.name, key,
                    yaml.safe_dump(old, default_flow_style=True, allow_unicode=True).strip().removesuffix("...").strip(),
                    raw, note])
    return line


def history(log: Path = LOG) -> list[dict]:
    if not log.exists():
        return []
    with log.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def undo(*, config_dir: Path = CONFIG_DIR, log: Path = LOG) -> str:
    """아직 되돌리지 않은 마지막 변경을 되돌린다 (되돌린 것도 이력에 '되돌림 #번호' 로 남는다)."""
    rows = history(log)
    done = {int(r["note"].split("#", 1)[1]) for r in rows if r["note"].startswith("되돌림 #")}
    todo = [(i, r) for i, r in enumerate(rows) if not r["note"].startswith("되돌림") and i not in done]
    if not todo:
        raise SystemExit("되돌릴 변경이 없다")
    i, last = todo[-1]
    return set_value(last["key"], last["old"], f"되돌림 #{i}", config_dir=config_dir, log=log)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--owner", help="A | B | C | D | 공통")
    st = sub.add_parser("set")
    st.add_argument("key")
    st.add_argument("value", help='YAML 값 (예: 0.004, "[0, 0, 0.135]", 특, null)')
    st.add_argument("--note", default="", help="무엇으로 쟀는지 (캘리퍼스, color_session truth ...)")
    sub.add_parser("undo")
    sub.add_parser("history")
    args = ap.parse_args(argv)

    if args.cmd == "list":
        items = [p for p in scan() if not args.owner or args.owner in p.owner]
        for owner in sorted({p.owner for p in items}):
            group = [p for p in items if p.owner == owner]
            print(f"\n[역할 {owner}] {len(group)}개")
            for p in group:
                print(f"  {p.reason:11s} {p.key:45s} = {p.value:22s} {p.comment[:60]}")
        print(f"\n현장에서 정할 값 {len(items)}개")
    elif args.cmd == "set":
        print(set_value(args.key, args.value, args.note))
    elif args.cmd == "undo":
        print(undo())
    else:
        for r in history():
            print(f"  {r['time']}  {r['key']:40s} {r['old']} -> {r['new']}  {r['note']}")


if __name__ == "__main__":
    main()
