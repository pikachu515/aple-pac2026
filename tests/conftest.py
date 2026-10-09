"""시험은 '현장 실측 전' 설정을 기준으로 한다.

현장에서 configs/*.yaml 에 실측값(field_values.py set, touch_calib --save)을 넣어도 시험 결과가 바뀌지 않게,
저장소 설정을 읽을 때 아래 실측 키를 null 로 되돌린다. 실측값이 필요한 시험은 cfg 에 직접 넣는다.
"""

from __future__ import annotations

import copy
from pathlib import Path

import aple.config as _config

#: 현장에서 채우는 값 — 시험에서는 비어 있는 것으로 본다
FIELD_KEYS = ("workcell.T_base_camera", "workcell.tcp_offset_m", "color.card_json",
              "workcell.auto_detect.enabled")      # 자동 검출은 그 시험(test_workcell_detect)에서만 켠다

_original_load = _config.load_config


def _pristine(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for dotted in FIELD_KEYS:
        *parents, leaf = dotted.split(".")
        node = cfg
        for key in parents:
            node = node.get(key, {})
        if leaf in node:
            node[leaf] = None
    return cfg


def load_config(config_dir=_config.CONFIG_DIR):
    cfg = _original_load(config_dir)
    return _pristine(cfg) if Path(config_dir).resolve() == Path(_config.CONFIG_DIR).resolve() else cfg


_config.load_config = load_config


#: 실측 전 줄 (주석 포함). 설정 파일을 복사해 쓰는 시험이 '실측 전' 상태에서 시작하게 한다
PRISTINE_LINES = {
    "workcell.yaml": {
        "  T_base_camera:": "  T_base_camera: null           # 4x4, eye-to-hand 캘리브레이션 결과 (m)",
        "  tcp_offset_m:": "  tcp_offset_m: null            # 핑거 어댑터 반영",
        "  auto_detect:": "  auto_detect: {enabled: false}",
    },
    "color.yaml": {
        "  card_json:": "  card_json: null                 # tools/color_session.py card 결과 경로 (현장 세션 폴더의 card.json). CardMonitor 가 읽는다",
    },
}


def reset_field_values(config_dir) -> None:
    """복사한 configs 폴더의 실측 키를 실측 전 줄로 되돌린다."""
    for name, lines in PRISTINE_LINES.items():
        path = Path(config_dir) / name
        text = path.read_text(encoding="utf-8").split("\n")
        for i, line in enumerate(text):
            for prefix, pristine in lines.items():
                if line.startswith(prefix):
                    text[i] = pristine
        path.write_text("\n".join(text), encoding="utf-8")


def pristine_copy(src_name: str, dest) -> Path:
    """configs/<src_name> 을 dest 로 복사하고 실측 키를 되돌린다."""
    import shutil
    dest = Path(dest)
    shutil.copy(Path(_config.CONFIG_DIR) / src_name, dest)
    lines = PRISTINE_LINES.get(src_name, {})
    text = dest.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(text):
        for prefix, pristine in lines.items():
            if line.startswith(prefix):
                text[i] = pristine
    dest.write_text("\n".join(text), encoding="utf-8")
    return dest
