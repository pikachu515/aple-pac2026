"""tools/color_session.py 를 합성 스냅샷 폴더로 끝까지 돌려 보는 검증 (화요일 실측과 같은 흐름)."""

import csv
import json

import numpy as np
import pytest

from tests.synthetic import INTR, Apple, full_truth, half_red, render, rot_x, rot_z, side_band, top_cap, visible_truth
from tools import color_session as cs

CARD = (20, 20, 80, 60)
WARM = (1.3, 1.0, 0.7)


def save(folder, i, color, depth):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder / f"snap_{i:04d}.npz", color=color, depth_m=depth,
                        intrinsics=np.array(INTR), stamp_s=float(i))


@pytest.fixture
def desk(tmp_path):
    """따뜻한 조명 아래 사과 3개 + 회색 카드, 2장."""
    apples = [Apple((-0.13, 0.03, 0.60), 0.038, side_band(0.6)),     # 착색 적음
              Apple((0.0, 0.03, 0.60), 0.042, half_red),
              Apple((0.13, 0.03, 0.60), 0.040, side_band(-0.7))]     # 착색 많음
    for i in range(2):
        color, depth, _ = render(apples, card_rect=CARD, tint=WARM, seed=i)
        save(tmp_path / "desk", i, color, depth)
    return tmp_path / "desk", apples


def test_card_then_analyze_matches_truth(desk):
    folder, apples = desk
    cs.cmd_card(folder, ",".join(map(str, CARD)), None)
    assert json.loads((folder / "card.json").read_text())["roi"] == list(CARD)

    rows = cs.analyze_folder(folder, cs.find_card(folder, None), cs.params("fuji"))
    first = [r for r in rows if r["file"] == "snap_0000.npz"]
    assert [r["apple"] for r in first] == [1, 2, 3]
    for r, a in zip(first, apples):
        assert r["ratio"] == pytest.approx(visible_truth(a), abs=0.04)
        assert r["diameter_mm"] == pytest.approx(a.radius * 2000, rel=0.05)
        assert r["light_dE"] < 1.0
    assert (folder / "_analysis" / "results.csv").exists()
    assert (folder / "_analysis" / "snap_0000_overlay.png").exists()


def test_suggest_finds_boundary_between_red_and_yellow(desk):
    folder, _ = desk
    cs.cmd_card(folder, ",".join(map(str, CARD)), None)
    th = cs.cmd_suggest(folder, None, "fuji")
    assert 45 < th < 95                         # 적색(약 25°)과 황록색(약 100°) 사이


@pytest.fixture
def truth(tmp_path):
    """사과 3개 × 6자세 (그대로, 뒤집기, 옆으로 눕혀 꼭지 축을 90°씩 돌린 4방향).

    눕힌 뒤 테이블 위에서 제자리로 돌리기만 하면(수직축 회전) 같은 면이 계속 위를 향한다 — 꼭지 축으로 돌려야 한다.
    """
    poses = [np.eye(3), rot_x(np.pi)] + [rot_x(np.pi / 2) @ rot_z(k * np.pi / 2) for k in range(4)]
    patterns = {"apple_01": half_red, "apple_02": side_band(0.4), "apple_03": top_cap(0.3)}
    for name, pat in patterns.items():
        for i, R in enumerate(poses):
            color, depth, _ = render([Apple((0.0, 0.0, 0.60), 0.04, pat, R)], card_rect=CARD)
            save(tmp_path / "truth" / name, i, color, depth)
    return tmp_path / "truth", patterns


def test_truth_estimates_full_surface_and_suggests_error(truth):
    root, patterns = truth
    rows, s = cs.truth_root(root, None, cs.params("fuji"))
    for r in rows:
        assert r["views"] == 6
        assert r["full_ratio"] == pytest.approx(full_truth(patterns[r["apple"]]), abs=0.05)
    # 꼭대기만 착색된 사과는 탑뷰가 전표면을 크게 과대평가한다 → 오차 제안값이 그만큼 커진다
    top_cap_row = next(r for r in rows if r["apple"] == "apple_03")
    assert top_cap_row["diff"] > 0.3
    assert s["suggested_top_view_error"] >= s["p90"]


def test_compare_reports_agreement(truth, tmp_path):
    root, _ = truth
    rows, _ = cs.truth_root(root, None, cs.params("fuji"))
    cs.write_csv(cs.analysis_dir(root) / "truth.csv", rows)
    labels = tmp_path / "labels.csv"
    with labels.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["apple", "label"])
        w.writerows([["apple_01", "상"], ["apple_02", "중"], ["apple_03", "애매"]])
    st = cs.cmd_compare(root / "_analysis" / "truth.csv", labels, "fuji")
    assert st["n"] == 2                          # '애매'는 일치율 계산에서 제외
    assert st["full_agree"] == 2


def test_next_folder_increments_trailing_number(tmp_path):
    from tools.capture_snapshot import next_folder
    assert next_folder(tmp_path / "apple_01").name == "apple_02"
    assert next_folder(tmp_path / "apple_09").name == "apple_10"
    assert next_folder(tmp_path / "desk").name == "desk_2"


def test_capture_tool_saves_snapshots_with_fake_camera(tmp_path, monkeypatch):
    """장비 없이 촬영 도구 흐름 확인: 설정 고정 → Enter 2번 저장 → q. 저장물은 analyze 로 바로 읽힌다."""
    import sys

    from tests import fake_realsense
    from tools import capture_snapshot

    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_realsense)
    keys = iter(["", "", "q"])
    monkeypatch.setattr("builtins.input", lambda *_: next(keys))
    out = tmp_path / "desk"
    monkeypatch.setattr(sys, "argv", ["capture_snapshot.py", "--out", str(out), "--no-preview", "--note", "t"])
    capture_snapshot.main()

    assert sorted(p.name for p in out.glob("*.npz")) == ["snap_0000.npz", "snap_0001.npz"]
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    assert meta["camera"]["white_balance"] == 4600 and meta["usb"] == "3.2"
    assert fake_realsense._Device.color.opts["enable_auto_exposure"] == 0      # 자동 노출 꺼짐
    rows = cs.analyze_folder(out, None, cs.params("fuji"), save_images=False)
    assert len(rows) == 2 and rows[0]["ratio"] == pytest.approx(0.5, abs=0.05)


def test_rs_check_passes_with_fake_camera(monkeypatch):
    import sys

    from tests import fake_realsense
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_realsense)
    sys.modules.pop("tools.rs_check", None)
    from tools import rs_check
    monkeypatch.setattr(sys, "argv", ["rs_check.py", "--seconds", "0.2"])
    clock = iter(i * 0.01 for i in range(100_000))                       # 호출마다 10 ms 흐른 것으로
    monkeypatch.setattr(rs_check.time, "time", lambda: next(clock))
    assert rs_check.main() == 0                                          # USB 3.2, 프레임 정상 → 통과
