"""실측값을 설정 파일에 써 넣기 — 주석·다른 품종·다른 키를 건드리지 않는지 확인."""

import shutil

import pytest
import yaml

from aple.config import CONFIG_DIR, set_config_value


@pytest.fixture
def varieties(tmp_path):
    p = tmp_path / "varieties.yaml"
    shutil.copy(CONFIG_DIR / "varieties.yaml", p)
    return p


def test_sets_nested_value_in_the_right_variety_only(varieties):
    before = yaml.safe_load(varieties.read_text(encoding="utf-8"))
    line = set_config_value(varieties, "varieties.fuji.hue_range_deg", [-40, 52], note="10/06")
    after = yaml.safe_load(varieties.read_text(encoding="utf-8"))
    assert after["varieties"]["fuji"]["hue_range_deg"] == [-40, 52]
    assert after["varieties"]["hongro"]["hue_range_deg"] == before["varieties"]["hongro"]["hue_range_deg"]
    assert "[실측 10/06]" in line and "[임시]" not in line
    assert "#" in line                                              # 주석 유지


def test_other_lines_and_comments_are_untouched(varieties):
    old = varieties.read_text(encoding="utf-8").split("\n")
    set_config_value(varieties, "inspection.top_view_error", 0.08, note="10/06")
    new = varieties.read_text(encoding="utf-8").split("\n")
    changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
    assert len(old) == len(new) and len(changed) == 1
    assert yaml.safe_load("\n".join(new))["inspection"]["top_view_error"] == 0.08


def test_refuses_missing_or_parent_keys(varieties):
    with pytest.raises(KeyError):
        set_config_value(varieties, "varieties.fuji.no_such_key", 1)
    with pytest.raises(ValueError):
        set_config_value(varieties, "varieties.fuji", 1)
