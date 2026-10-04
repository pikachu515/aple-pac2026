"""aple/schema.py → docs/interfaces.md (Mermaid 다이어그램) 생성.

schema.py 가 원본 모델이고 다이어그램은 그 결과물이다. 다이어그램을 직접 고치지 않는다.

    python tools/gen_interfaces.py          # 생성
    python tools/gen_interfaces.py --check  # 최신 상태가 아니면 실패 (테스트·CI용)
"""

from __future__ import annotations

import dataclasses
import enum
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aple import schema  # noqa: E402

OUT = ROOT / "docs" / "interfaces.md"


def dataclasses_in(mod) -> list[type]:
    return [c for c in vars(mod).values()
            if isinstance(c, type) and dataclasses.is_dataclass(c) and c.__module__ == mod.__name__]


def enums_in(mod) -> list[type]:
    return [c for c in vars(mod).values()
            if isinstance(c, type) and issubclass(c, enum.Enum) and c.__module__ == mod.__name__]


def member_type(t: str) -> str:
    """Mermaid 클래스 멤버에 쓸 수 있는 형태로 타입 표기를 바꾼다."""
    optional = t.endswith(" | None")
    if optional:
        t = t[: -len(" | None")]
    t = t.replace("np.ndarray", "ndarray")
    t = t.replace("[", "~").replace("]", "~").replace(", ", ",")
    return t + ("?" if optional else "")


def referenced(t: str, names: set[str]) -> list[str]:
    return [n for n in re.findall(r"[A-Za-z_]\w*", t) if n in names]


def build() -> str:
    dcs, ens = dataclasses_in(schema), enums_in(schema)
    names = {c.__name__ for c in dcs + ens}

    flow = ["flowchart LR"]
    for key, label in schema.MODULES.items():
        flow.append(f'    {key}["{label}"]')
    merged: dict[tuple[str, str], list[str]] = {}
    for src, dst, cls in schema.FLOWS:
        merged.setdefault((src, dst), []).append(cls.__name__)
    for (src, dst), labels in merged.items():
        flow.append(f'    {src} -->|"{", ".join(labels)}"| {dst}')

    cd = ["classDiagram", "    direction LR"]
    rels = []
    for c in dcs:
        cd.append(f"    class {c.__name__} {{")
        for f in dataclasses.fields(c):
            cd.append(f"        +{member_type(str(f.type))} {f.name}")
            for ref in referenced(str(f.type), names):
                rels.append(f"    {c.__name__} --> {ref} : {f.name}")
        cd.append("    }")
    for e in ens:
        cd.append(f"    class {e.__name__} {{")
        cd.append("        <<enumeration>>")
        for m in e:
            cd.append(f"        {m.name}")
        cd.append("    }")
    cd += rels

    table = ["| 보내는 모듈 | 받는 모듈 | 데이터 | 필드 |", "|---|---|---|---|"]
    plain = {k: v.replace("<br/>", " · ") for k, v in schema.MODULES.items()}
    for src, dst, cls in schema.FLOWS:
        fields = ", ".join(f"`{f.name}`" for f in dataclasses.fields(cls))
        table.append(f"| {plain[src]} | {plain[dst]} | `{cls.__name__}` | {fields} |")

    return "\n".join([
        "# 모듈 간 인터페이스",
        "",
        "> 이 파일은 `tools/gen_interfaces.py` 가 `aple/schema.py` 로부터 자동 생성한다. **직접 고치지 않는다.**",
        "> 인터페이스를 바꾸려면 `aple/schema.py` 의 클래스와 `FLOWS` 표를 고친다.",
        "",
        "## 데이터 흐름",
        "",
        "```mermaid", *flow, "```",
        "",
        *table,
        "",
        "## 데이터 구조",
        "",
        "```mermaid", *cd, "```",
        "",
    ])


def main() -> int:
    text = build()
    if "--check" in sys.argv:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print("docs/interfaces.md 가 schema.py 와 다르다 — python tools/gen_interfaces.py 실행")
            return 1
        return 0
    OUT.write_text(text, encoding="utf-8", newline="\n")
    print(f"생성: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
