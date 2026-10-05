"""Piper Studio 공유 메모리 → 실시간 판정 경로 검증.

Piper Studio 발행자(piper_shm.Publisher)와 같은 형식으로 세그먼트 파일을 직접 만들어 읽힌다.
"""

import json
import struct

import numpy as np
import pytest

from aple.hal.depth_align import DepthEncoding
from aple.hal.shm_frames import (HEADER_FMT, HEADER_SIZE, MAGIC, ShmError, ShmReader, find_realsense_serials,
                                 read_pair, segment_path)
from tests.synthetic import Apple, render, side_band
from tests.test_depth_align import BASELINE, C_INTR, D_INTR, SIZE, piper_encode
from tools import judge_live

SERIAL = "123456789012"


def write_segment(path, frame, seq=5, wall_ns=1_000_000_000, n_slots=4, version=1, magic=MAGIC):
    """piper_shm.Publisher 와 같은 배치: 헤더 64 B + n_slots 개 슬롯, 최신은 slot[seq % n_slots]."""
    h, w, ch = frame.shape
    slot = h * w * ch
    buf = bytearray(HEADER_SIZE + slot * n_slots)
    buf[:struct.calcsize(HEADER_FMT)] = struct.pack(HEADER_FMT, magic, version, w, h, ch, 1, n_slots, slot, seq, wall_ns)
    off = HEADER_SIZE + (seq % n_slots) * slot
    buf[off:off + slot] = frame.tobytes()
    path.write_bytes(bytes(buf))


@pytest.fixture
def piper(tmp_path):
    apples = [Apple((-0.10, 0.0, 0.60), 0.040, side_band(0.6)), Apple((0.09, 0.0, 0.60), 0.038, side_band(-0.7))]
    color, _, _ = render(apples, size=SIZE, intr=C_INTR.tuple())
    shifted = [Apple(tuple(np.asarray(a.center) - BASELINE), a.radius, a.pattern) for a in apples]
    _, depth_d, _ = render(shifted, size=SIZE, intr=D_INTR.tuple())
    shm = tmp_path / "shm"
    shm.mkdir()
    write_segment(segment_path(f"rs:{SERIAL}:color", shm), color)
    write_segment(segment_path(f"rs:{SERIAL}:depth", shm), piper_encode(depth_d, DepthEncoding(150, 1200)),
                  wall_ns=1_010_000_000)                                   # 10 ms 차이
    calib = tmp_path / "d435.json"
    calib.write_text(json.dumps({
        "depth": {**{k: getattr(D_INTR, k) for k in ("fx", "fy", "cx", "cy", "width", "height")}},
        "color": {**{k: getattr(C_INTR, k) for k in ("fx", "fy", "cx", "cy", "width", "height")}},
        "depth_to_color": {"R": np.eye(3).ravel().tolist(), "t": BASELINE.tolist()},
    }), encoding="utf-8")
    return shm, calib, color


def test_segment_naming_matches_piper_studio(tmp_path):
    assert segment_path("rs:123:color", tmp_path).name == "piper.cam.rs_123_color"
    (tmp_path / "piper.cam.rs_123_color").touch()
    (tmp_path / "piper.cam.v4l_video0").touch()
    assert find_realsense_serials(tmp_path) == ["123"]


def test_reader_returns_latest_slot_and_pairs_frames(piper):
    shm, _, color = piper
    c = ShmReader(segment_path(f"rs:{SERIAL}:color", shm))
    d = ShmReader(segment_path(f"rs:{SERIAL}:depth", shm))
    frame, seq, wall = c.read()
    assert seq == 5 and np.array_equal(frame, color)
    pair = read_pair(c, d, max_skew_ms=50)
    assert pair is not None and pair[0].shape == (480, 848, 3)
    assert read_pair(c, d, max_skew_ms=5, timeout_s=0.05) is None             # 시각 차 10 ms > 5 ms
    c.close()
    d.close()


@pytest.mark.parametrize("kw, msg", [(dict(version=2), "버전"), (dict(magic=0x1234), "형식")])
def test_reader_refuses_unknown_format(tmp_path, kw, msg):
    p = tmp_path / "piper.cam.rs_1_color"
    write_segment(p, np.zeros((4, 4, 3), np.uint8), **kw)
    with pytest.raises(ShmError, match=msg):
        ShmReader(p)


def test_judge_live_once_grades_from_shared_memory(piper, tmp_path):
    shm, calib, _ = piper
    out = tmp_path / "live"
    rows = judge_live.main(["--once", "--no-window", "--shm-dir", str(shm), "--calib", str(calib),
                            "--out", str(out), "--variety", "fuji"])
    assert [r["grade"] for r in rows] == ["중", "상"]
    assert rows[0]["diameter_mm"] == pytest.approx(80, abs=4)
    assert (out / "snap_0000.npz").exists() and (out / "live_0000_overlay.png").exists()


def test_saved_live_snapshot_reanalyzes_offline(piper, tmp_path):
    from tools import color_session as cs
    shm, calib, _ = piper
    out = tmp_path / "live"
    judge_live.main(["--once", "--no-window", "--shm-dir", str(shm), "--calib", str(calib), "--out", str(out)])
    rows = cs.analyze_folder(out, None, cs.params("fuji"), save_images=False)
    assert [r["grade"] for r in rows] == ["중", "상"]


def test_missing_calib_explains_how_to_make_it(piper, tmp_path):
    shm, _, _ = piper
    with pytest.raises(SystemExit, match="save-calib"):
        judge_live.main(["--once", "--no-window", "--shm-dir", str(shm), "--calib", str(tmp_path / "none.json")])
