"""home·scan 자세 가르치기 — 읽은 관절각을 workcell.yaml 에 저장하고, 안전 범위 밖이면 거부한다."""

import math

import pytest
import yaml

from tests.conftest import pristine_copy
from tools import teach_pose

POSE = (0.0, 1.2, -0.9, 0.0, 0.6, 0.1)


def test_teach_writes_pose_and_keeps_other_lines(tmp_path):
    workcell = pristine_copy("workcell.yaml", tmp_path / "workcell.yaml")
    teach_pose.main(["scan", "--workcell", str(workcell)], joint_reader=lambda cfg, iface: POSE)
    poses = yaml.safe_load(workcell.read_text(encoding="utf-8"))["workcell"]["poses"]
    assert poses["scan_joint_rad"] == pytest.approx(POSE)
    assert poses["home_joint_rad"] is None


def test_dry_run_does_not_write(tmp_path):
    workcell = pristine_copy("workcell.yaml", tmp_path / "workcell.yaml")
    before = workcell.read_text(encoding="utf-8")
    teach_pose.main(["home", "--workcell", str(workcell), "--dry-run"], joint_reader=lambda cfg, iface: POSE)
    assert workcell.read_text(encoding="utf-8") == before


def test_pose_outside_safety_limits_is_refused(tmp_path, monkeypatch):
    from aple.config import load_config
    cfg = load_config()
    cfg["safety"]["joint_limits_rad"] = [[-2.5, 2.5], [-0.07, 3.05], [-2.88, 0.07], [-1.66, 1.66], [-1.05, 1.05], [-2.0, 2.0]]
    monkeypatch.setattr(teach_pose, "load_config", lambda: cfg)
    workcell = pristine_copy("workcell.yaml", tmp_path / "workcell.yaml")
    bad = (0.0, 1.2, -0.9, 0.0, math.radians(80), 0.0)                      # joint5 > 60°
    with pytest.raises(SystemExit, match="joint5"):
        teach_pose.main(["home", "--workcell", str(workcell)], joint_reader=lambda cfg, iface: bad)


def test_iface_override_enables_read_only_in_memory():
    seen = {}

    class FakeStudio:
        def __init__(self, cfg):
            seen.update(cfg["piper_web"])

        def joint_state(self):
            return POSE

    import aple.hal.piper_studio as ps
    original = ps.PiperStudio
    ps.PiperStudio = FakeStudio
    try:
        from aple.config import load_config
        assert teach_pose.read_joints(load_config(), "can_follower") == POSE
    finally:
        ps.PiperStudio = original
    assert seen["enabled"] is True and seen["iface"] == "can_follower"


def test_teach_box_hover_pose(tmp_path):
    workcell = pristine_copy("workcell.yaml", tmp_path / "workcell.yaml")
    before = yaml.safe_load(workcell.read_text(encoding="utf-8"))["workcell"]["poses"]
    teach_pose.main(["hover_MIDDLE", "--workcell", str(workcell)], joint_reader=lambda cfg, iface: POSE)
    poses = yaml.safe_load(workcell.read_text(encoding="utf-8"))["workcell"]["poses"]
    assert poses["hover_MIDDLE_joint_rad"] == pytest.approx(POSE)
    assert poses["hover_UPPER_joint_rad"] == before["hover_UPPER_joint_rad"]   # 다른 자세는 그대로
