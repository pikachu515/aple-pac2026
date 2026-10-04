"""configs/*.yaml 을 하나의 dict 로 읽는다. 최상위 키(workcell, varieties, inspection, ...)는 파일끼리 겹치지 않는다."""

from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_config(config_dir: str | Path = CONFIG_DIR) -> dict:
    cfg: dict = {}
    for p in sorted(Path(config_dir).glob("*.yaml")):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        dup = set(cfg) & set(data)
        if dup:
            raise ValueError(f"{p.name}: 최상위 키 중복 {sorted(dup)}")
        cfg.update(data)
    return cfg
