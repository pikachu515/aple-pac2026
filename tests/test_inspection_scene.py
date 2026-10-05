"""M2 진입점(inspect_scene)과 리젝트 곡선 검증."""

import time

import numpy as np
import pytest

from aple.inspection.evaluation import pick_delta_up, reject_curve
from aple.inspection.pipeline import SceneMismatch, inspect_scene
from aple.inspection.standalone import segment_apples
from aple.schema import (AppleDetection2D, AppleState, CaptureMode, Grade, InspectionPolicy, LightingStatus,
                         LightState, RejectReason, SceneSnapshot, Source)
from tests.synthetic import INTR, Apple, render, side_band
from tests.test_inspection import P


def scene(scene_id=5):
    apples = [Apple((-0.13, 0.0, 0.60), 0.038, side_band(0.7)),     # 착색 적음 → 중
              Apple((0.0, 0.0, 0.60), 0.042, side_band(-0.8)),     # 착색 많음 → 상
              Apple((0.13, 0.0, 0.60), 0.040, side_band(0.0))]     # 반반 → 경계
    color, depth, _ = render(apples)
    snap = SceneSnapshot(scene_id, Source.TOP, CaptureMode.FULL, color, depth, INTR, 0.0)
    dets = segment_apples(color, depth, INTR, scene_id=scene_id)
    return snap, dets


def state(scene_id, apple_id, det_id, diam_mm=78, occluded=False):
    return AppleState(scene_id, apple_id, det_id, (0, 0, 0), diam_mm / 1000, 0, 0.03, 0.001, 0.9, 1.0, 0.0, occluded)


def test_scene_without_3d_state_grades_each_detection():
    snap, dets = scene()
    res = inspect_scene(snap, dets, None, P)
    assert [r.apple_id for r in res] == [1, 2, 3]
    assert [r.grade for r in res] == [Grade.MIDDLE, Grade.UPPER, Grade.REJECT]
    assert res[2].reject_reason == RejectReason.BORDERLINE
    assert all(r.scene_id == 5 for r in res)


def test_scene_with_3d_state_uses_apple_ids_and_size_rules():
    snap, dets = scene()
    apples = [state(5, 11, 1), state(5, 12, 2, diam_mm=104), state(5, 13, 3, occluded=True), state(5, 14, 9)]
    res = {r.apple_id: r for r in inspect_scene(snap, dets, apples, P)}
    assert res[11].grade == Grade.MIDDLE
    assert res[12].reject_reason == RejectReason.OVERSIZE        # 착색과 무관하게 크기 우선
    assert res[13].reject_reason == RejectReason.OCCLUDED        # 리젝트가 아니라 재촬영 대상
    assert res[14].reject_reason == RejectReason.OCCLUDED        # 검출이 없는 사과


def test_lighting_fault_holds_back_every_apple():
    snap, dets = scene()
    res = inspect_scene(snap, dets, None, P, light=LightingStatus(LightState.FAULT, 9.0, True))
    assert all(r.reject_reason == RejectReason.LOW_CONF for r in res)


def test_adaptive_policy_sends_borderline_apple_to_multi_view():
    snap, dets = scene()
    res = inspect_scene(snap, dets, None, P, policy=InspectionPolicy.ADAPTIVE)
    assert [r.final for r in res] == [True, True, False]


def test_scene_mismatch_is_refused():
    snap, dets = scene(scene_id=5)
    stale = [AppleDetection2D(4, d.det_id, d.bbox, d.confidence, d.mask) for d in dets]
    with pytest.raises(SceneMismatch):
        inspect_scene(snap, stale, None, P)
    with pytest.raises(SceneMismatch):
        inspect_scene(snap, dets, [state(4, 1, 1)], P)


def test_five_apples_judged_quickly():
    """사이클 시간 부담 확인: 848×480 스냅샷, 사과 5개 판정."""
    apples = [Apple((x, 0.0, 0.60), 0.04, side_band(0.2)) for x in (-0.2, -0.1, 0.0, 0.1, 0.2)]
    color, depth, _ = render(apples, size=(480, 848), intr=(615.0, 615.0, 424.0, 240.0))
    intr = (615.0, 615.0, 424.0, 240.0)
    snap = SceneSnapshot(1, Source.TOP, CaptureMode.FULL, color, depth, intr, 0.0)
    t0 = time.perf_counter()
    dets = segment_apples(color, depth, intr, scene_id=1)
    res = inspect_scene(snap, dets, None, P)
    dt = time.perf_counter() - t0
    print(f"\n분할 + 판정 5개: {dt * 1000:.0f} ms")
    assert len(res) == 5 and dt < 3.0


# ---------------------------------------------------------------- 리젝트 곡선

def sample(n=200, noise=0.08, seed=0):
    rng = np.random.default_rng(seed)
    full = rng.uniform(0.1, 0.8, n)
    top = np.clip(full + rng.normal(0, noise, n), 0, 1)        # 탑뷰는 전표면에 잡음이 낀 값
    truth = [Grade.UPPER if f >= P.coloring_threshold else Grade.MIDDLE for f in full]
    return list(top), truth


def test_reject_curve_trades_rejects_for_fewer_upgrades():
    top, truth = sample()
    curve = reject_curve(top, truth, P)
    rej = [c.reject_rate for c in curve]
    up = [c.up_error_rate for c in curve]
    assert all(b >= a - 1e-12 for a, b in zip(rej, rej[1:]))      # delta_up ↑ → 리젝트 ↑
    assert all(b <= a + 1e-12 for a, b in zip(up, up[1:]))         # delta_up ↑ → '중→상' ↓
    assert up[0] > 0 and up[-1] == 0


def test_pick_delta_up_reaches_zero_costly_errors_with_least_rejects():
    top, truth = sample()
    curve = reject_curve(top, truth, P)
    best = pick_delta_up(curve)
    assert best.up_error_rate == 0
    assert all(c.reject_rate >= best.reject_rate for c in curve if c.up_error_rate == 0)
