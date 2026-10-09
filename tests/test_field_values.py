"""현장 실측 값 도구 — 남은 값 목록, 한 줄 변경(주석 유지·표시 교체), 깨지면 복구, 되돌리기, 이력."""

import shutil

import pytest

from aple.config import CONFIG_DIR, load_config
from tests.conftest import reset_field_values
from tools import field_values as fv


@pytest.fixture
def cfgdir(tmp_path):
    d = tmp_path / "configs"
    shutil.copytree(CONFIG_DIR, d)
    reset_field_values(d)                                                 # 현장에서 채운 값과 무관하게
    return d, tmp_path / "logs" / "field_changes.csv"


def test_list_finds_provisional_values_by_owner(cfgdir):
    d, _ = cfgdir
    items = {p.key: p for p in fv.scan(d)}
    assert items["workcell.T_base_camera"].reason == "null" and items["workcell.T_base_camera"].owner == "A"
    assert items["varieties.fuji.hue_range_deg"].reason == "임시" and items["varieties.fuji.hue_range_deg"].owner == "C"
    assert items["piper_web.iface"].reason == "PLACEHOLDER" and items["piper_web.iface"].owner == "B"
    assert "size_bins.bins[0]" in items                                   # 리스트 항목도 번호로
    assert "inspection.cos_theta_min" not in items                        # 정해진 값은 안 나온다


def test_set_keeps_comment_and_marks_measured(cfgdir):
    d, log = cfgdir
    line = fv.set_value("varieties.fuji.min_chroma", "18", "suggest", config_dir=d, log=log)
    assert "min_chroma: 18" in line and "[실측" in line and "무채색" in line
    assert load_config(d)["varieties"]["fuji"]["min_chroma"] == 18
    assert "varieties.fuji.min_chroma" not in {p.key for p in fv.scan(d)}  # 목록에서 빠진다


def test_set_handles_placeholder_lists_and_inline_maps(cfgdir):
    d, log = cfgdir
    fv.set_value("piper_web.iface", "can_follower", config_dir=d, log=log)
    fv.set_value("workcell.tcp_offset_m", "[0, 0, 0.135]", "캘리퍼스", config_dir=d, log=log)
    fv.set_value("execution.placement_verify.height_tol_m", "0.008", config_dir=d, log=log)   # { } 안의 값
    cfg = load_config(d)
    assert cfg["piper_web"]["iface"] == "can_follower"
    assert cfg["workcell"]["tcp_offset_m"] == [0, 0, 0.135]
    assert cfg["execution"]["placement_verify"]["height_tol_m"] == 0.008
    keys = {p.key for p in fv.scan(d)}
    assert "piper_web.iface" not in keys and "execution.placement_verify" not in keys


def test_typo_is_rejected_and_file_untouched(cfgdir):
    d, log = cfgdir
    before = (d / "workcell.yaml").read_text(encoding="utf-8")
    with pytest.raises(KeyError, match="오타"):
        fv.set_value("workcell.tcp_ofset_m", "[0, 0, 0.1]", config_dir=d, log=log)
    assert (d / "workcell.yaml").read_text(encoding="utf-8") == before


def test_undo_restores_previous_values_in_order(cfgdir):
    d, log = cfgdir
    fv.set_value("inspection.top_view_error", "0.07", config_dir=d, log=log)
    fv.set_value("inspection.top_view_error", "0.05", config_dir=d, log=log)
    fv.undo(config_dir=d, log=log)
    assert load_config(d)["inspection"]["top_view_error"] == 0.07
    fv.undo(config_dir=d, log=log)
    assert load_config(d)["inspection"]["top_view_error"] == 0.10
    with pytest.raises(SystemExit, match="되돌릴"):
        fv.undo(config_dir=d, log=log)
    rows = fv.history(log)
    assert [r["new"] for r in rows] == ["0.07", "0.05", "0.07", "0.1"]    # 이력은 모두 남는다


def test_unused_values_are_not_listed(cfgdir):
    d, _ = cfgdir
    keys = {p.key for p in fv.scan(d)}
    assert "color.patch_json" not in keys                                 # [미사용 10/8] 레퍼런스 패치
    assert "color.card_json" in keys                                      # 회색 카드는 현장에서 정한다


def test_optional_values_are_separated_from_required(cfgdir):
    d, _ = cfgdir
    by_key = {p.key: p for p in fv.scan(d)}
    assert by_key["live_camera.serial"].reason == "선택"                  # null = 첫 RealSense 자동 선택
    assert by_key["size_bins.bins[0]"].reason == "선택"                   # open_width_mm null = 직경 + 여유
    assert by_key["workcell.T_base_camera"].reason == "null"              # 필수는 그대로
    assert by_key["gripper_attachment.assembled.tcp_offset_mm"].owner == "D"


def test_set_list_item_value(cfgdir):
    d, log = cfgdir
    line = fv.set_value("size_bins.bins[2].open_width_mm", "92", "캘리퍼스", config_dir=d, log=log)
    assert "open_width_mm: 92" in line and "range_mm: [70, 85]" in line
    cfg = load_config(d)
    assert cfg["size_bins"]["bins"][2]["open_width_mm"] == 92
    assert cfg["size_bins"]["bins"][1]["open_width_mm"] is None               # 다른 항목은 그대로
    fv.undo(config_dir=d, log=log)
    assert load_config(d)["size_bins"]["bins"][2]["open_width_mm"] is None
