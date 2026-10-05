"""Piper Studio 깊이(인코딩·비정렬) → 컬러 정렬 미터 깊이 검증.

실제 상황을 흉내 낸다: 깊이 카메라는 컬러보다 15 mm 옆에 있고 화각이 넓으며(fx 425 vs 615),
깊이는 Piper Studio 와 같은 방식으로 uint8 인코딩된다.
"""

import json

import numpy as np
import pytest

from aple.hal.depth_align import (D435Calib, DepthEncoding, Intrinsics, align_depth_to_color, decode_depth,
                                  snapshot_from_piper)
from aple.inspection.coloration import measure_view
from aple.inspection.standalone import segment_apples
from tests.synthetic import Apple, half_red, render, side_band, visible_truth
from tests.test_inspection import P

SIZE = (480, 848)
C_INTR = Intrinsics(615.0, 615.0, 424.0, 240.0, 848, 480)
D_INTR = Intrinsics(425.0, 425.0, 424.0, 240.0, 848, 480)
BASELINE = np.array([-0.015, 0.0, 0.0])          # 컬러 좌표에서 본 깊이 카메라 위치
CALIB = D435Calib(D_INTR, C_INTR, np.eye(3), BASELINE)   # P_color = P_depth + t


def piper_encode(depth_m: np.ndarray, enc: DepthEncoding) -> np.ndarray:
    """wego-piper-web rs/piper_rs/depth.py encode_depth 와 같은 계산 (raw mm 단위, D435)."""
    d = depth_m * 1000.0
    invalid = depth_m <= 0
    v = (d - enc.near_mm) / (enc.far_mm - enc.near_mm) * 254
    v = np.clip(v, 0, 254).astype(np.uint8)
    v[invalid] = 255
    return np.repeat(v[:, :, None], 3, axis=2)


def scene(apples):
    color, depth_c, masks = render(apples, size=SIZE, intr=C_INTR.tuple())
    shifted = [Apple(tuple(np.asarray(a.center) - BASELINE), a.radius, a.pattern, a.R_cam_apple) for a in apples]
    _, depth_d, _ = render(shifted, size=SIZE, intr=D_INTR.tuple(), depth_dropout=0.1)
    return color, depth_c, depth_d, masks


def test_decode_inverts_piper_encoding_within_half_step():
    enc = DepthEncoding(150, 1200)
    d = np.linspace(0.2, 1.1, 1000, dtype=np.float32)[None, :]
    back = decode_depth(piper_encode(d, enc), enc)
    step = (1200 - 150) / 254 / 1000
    assert np.abs(back - d).max() <= step / 2 + 1e-6
    assert decode_depth(piper_encode(np.zeros((1, 3), np.float32), enc), enc).max() == 0   # 무효는 0


def test_alignment_recovers_color_frame_depth():
    apples = [Apple((-0.08, 0.0, 0.60), 0.040, half_red), Apple((0.10, 0.03, 0.60), 0.036, half_red)]
    color, depth_true, depth_d, masks = scene(apples)
    aligned = align_depth_to_color(depth_d, CALIB)
    for m in masks:
        both = m & (aligned > 0)
        assert both.sum() / m.sum() > 0.9
        assert np.median(np.abs(aligned[both] - depth_true[both])) < 0.002


def test_unaligned_depth_would_misplace_apples():
    """정렬 없이 깊이를 컬러에 그대로 겹치면 사과 영역이 크게 어긋난다 — 정렬이 필요한 이유."""
    apples = [Apple((0.10, 0.0, 0.60), 0.040, half_red)]
    color, _, depth_d, masks = scene(apples)
    naive = segment_apples(color, depth_d, C_INTR.tuple())
    good = segment_apples(color, align_depth_to_color(depth_d, CALIB), C_INTR.tuple())
    iou = lambda a, b: (a & b).sum() / (a | b).sum()
    assert iou(good[0].mask, masks[0]) > 0.85
    assert not naive or iou(naive[0].mask, masks[0]) < 0.5


def test_piper_snapshot_gives_same_grading_inputs():
    enc = DepthEncoding(150, 1200)
    apples = [Apple((-0.10, 0.0, 0.60), 0.040, side_band(0.4)), Apple((0.09, 0.02, 0.60), 0.038, side_band(-0.5))]
    color, _, depth_d, masks = scene(apples)
    snap = snapshot_from_piper(color, piper_encode(depth_d, enc), enc, CALIB, scene_id=7)
    assert snap.scene_id == 7 and snap.intrinsics == C_INTR.tuple()
    dets = segment_apples(snap.color, snap.depth_m, snap.intrinsics)
    assert len(dets) == 2
    for d, a in zip(dets, apples):
        vm = measure_view(snap.color, snap.depth_m, d.mask, snap.intrinsics, P)
        assert vm.ratio_visible == pytest.approx(visible_truth(a), abs=0.04)
        assert vm.sphere.radius == pytest.approx(a.radius, rel=0.05)


def test_calib_file_round_trip(tmp_path):
    p = tmp_path / "d435.json"
    p.write_text(json.dumps({
        "depth": {"fx": 425, "fy": 425, "cx": 424, "cy": 240, "width": 848, "height": 480},
        "color": {"fx": 615, "fy": 615, "cx": 424, "cy": 240, "width": 848, "height": 480},
        "depth_to_color": {"R": np.eye(3).ravel().tolist(), "t": [-0.015, 0, 0]},
    }), encoding="utf-8")
    c = D435Calib.load(p)
    assert c.depth == D_INTR and np.allclose(c.t, BASELINE)
