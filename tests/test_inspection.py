"""역할 C (M2 색상 품질 판별) — 합성 장면의 정답과 비교하는 검증."""

import numpy as np
import pytest

from aple.config import load_config
from aple.inspection.color import (CardCalib, apply_gain, bgr_to_lab, calibrate_card, card_brightness,
                                   hue_chroma, lighting_status, suggest_hue_max)
from aple.inspection.coloration import (InHandView, accumulate_views, color_ratio, measure_view,
                                        params_from_config)
from aple.inspection.geometry import sphere_from_mask
from aple.inspection.grade_classifier import classify, decide
from aple.inspection.standalone import segment_apples
from aple.schema import (AppleDetection2D, AppleState, CaptureMode, ColorRatio, Grade, InspectionPolicy,
                         InspectMode, LightingStatus, LightState, RejectReason, SceneSnapshot, Source)
from tests.synthetic import (INTR, Apple, coverage_truth, full_truth, half_red, render, rot_x, rot_z, side_band,
                             top_cap, visible_truth)

P = params_from_config(load_config())


# ---------------------------------------------------------------- 색 공간

def test_lab_matches_reference_values():
    lab = bgr_to_lab(np.array([[[0, 0, 255], [0, 255, 255], [128, 128, 128]]], np.uint8))[0]
    assert lab[0] == pytest.approx([53.24, 80.09, 67.20], abs=0.1)      # sRGB 빨강
    assert lab[1] == pytest.approx([97.14, -21.55, 94.48], abs=0.1)     # sRGB 노랑
    assert abs(lab[2][1]) < 0.01 and abs(lab[2][2]) < 0.01              # 회색은 무채색


def test_hue_separates_red_and_yellow_green():
    h, c = hue_chroma(bgr_to_lab(np.array([[[38, 30, 175], [75, 190, 205]]], np.uint8))[0])
    lo, hi = P.hue_range_deg
    assert lo <= h[0] <= hi and h[1] > hi        # 착색부는 범위 안, 황록색은 밖
    assert (c >= P.min_chroma).all()


# ---------------------------------------------------------------- 회색 카드 · 조명 감시

CARD_ROI = (20, 20, 80, 60)


def test_card_calibration_removes_warm_tint():
    a = Apple((0.0, 0.0, 0.60), 0.04, half_red)
    ref, _, _ = render([a], card_rect=CARD_ROI)
    warm, depth, masks = render([a], card_rect=CARD_ROI, tint=(1.35, 1.0, 0.65))
    calib = calibrate_card(warm, CARD_ROI)
    fixed = apply_gain(warm, calib)

    L, a_, b_ = bgr_to_lab(fixed[40:60, 40:80]).reshape(-1, 3).mean(axis=0)
    assert abs(a_) < 1.0 and abs(b_) < 1.0 and L == pytest.approx(50, abs=2)
    r_ref = measure_view(ref, depth, masks[0], INTR, P).ratio_visible
    r_warm = measure_view(warm, depth, masks[0], INTR, P).ratio_visible
    r_fix = measure_view(fixed, depth, masks[0], INTR, P).ratio_visible
    assert abs(r_fix - r_ref) < 0.02             # 보정 후에는 기준 조명과 같은 결과

    # 황록색 바탕의 색상각: 따뜻한 조명은 적색 쪽(작은 각)으로 끌어당기고, 보정하면 원래대로 돌아온다
    bg = masks[0] & ~measure_view(ref, depth, masks[0], INTR, P).colored & measure_view(ref, depth, masks[0], INTR, P).valid
    h = {k: hue_chroma(bgr_to_lab(img[bg]))[0].mean() for k, img in (("ref", ref), ("warm", warm), ("fix", fixed))}
    assert h["ref"] - h["warm"] > 5
    assert abs(h["fix"] - h["ref"]) < 1.5


def test_card_brightness_reports_target_gray():
    img, _, _ = render([], card_rect=CARD_ROI)
    assert card_brightness(img, CARD_ROI) == pytest.approx(118, abs=2)


def test_lighting_status_detects_change():
    img, _, _ = render([], card_rect=CARD_ROI)
    calib = calibrate_card(img, CARD_ROI)
    assert lighting_status(img, calib).light_state == LightState.OK
    dim, _, _ = render([], card_rect=CARD_ROI, tint=(0.6, 0.6, 0.6))
    assert lighting_status(dim, calib).light_state != LightState.OK
    blue, _, _ = render([], card_rect=CARD_ROI, tint=(0.85, 1.0, 1.4))
    assert lighting_status(blue, calib).light_state == LightState.FAULT


def test_card_calib_round_trips_through_dict():
    img, _, _ = render([], card_rect=CARD_ROI)
    c = calibrate_card(img, CARD_ROI)
    assert CardCalib.from_dict(c.to_dict()) == c


# ---------------------------------------------------------------- 구 기하

def test_sphere_from_mask_recovers_size_and_position():
    a = Apple((0.05, -0.03, 0.60), 0.040, half_red)
    _, depth, masks = render([a])
    s = sphere_from_mask(masks[0], depth, INTR)
    assert s.radius == pytest.approx(0.040, rel=0.03)
    assert np.linalg.norm(s.center - np.array(a.center)) < 0.003
    assert s.fit_residual < 0.002


# ---------------------------------------------------------------- 착색비율 (모드 A)

def test_top_view_coverage_is_about_quarter_sphere():
    """원근 효과로 0.25보다 약간 작다 (0.6 m, r 40 mm → 약 0.226)."""
    a = Apple((0.0, 0.0, 0.60), 0.04, half_red)
    color, depth, masks = render([a])
    vm = measure_view(color, depth, masks[0], INTR, P)
    assert vm.coverage == pytest.approx(coverage_truth(a), abs=0.01)
    assert 0.20 < vm.coverage < 0.25


@pytest.mark.parametrize("pattern", [half_red, side_band(0.5), side_band(-0.4), top_cap(0.8)])
def test_visible_ratio_matches_truth(pattern):
    a = Apple((0.02, 0.01, 0.60), 0.04, pattern)
    color, depth, masks = render([a])
    vm = measure_view(color, depth, masks[0], INTR, P)
    assert vm.ratio_visible == pytest.approx(visible_truth(a), abs=0.03)


def test_area_weighting_beats_pixel_count():
    """꼭대기 원형 착색: 단순 픽셀 수는 가장자리(비스듬한 면)를 덜 세어 착색을 과대평가한다."""
    a = Apple((0.0, 0.0, 0.60), 0.04, top_cap(0.8))
    color, depth, masks = render([a])
    vm = measure_view(color, depth, masks[0], INTR, P)
    naive = vm.colored.sum() / vm.valid.sum()
    truth = visible_truth(a)
    assert naive - truth > 0.03
    assert abs(vm.ratio_visible - truth) < abs(naive - truth)
    assert vm.ratio_visible == pytest.approx(truth, abs=0.03)


def test_depth_holes_do_not_break_measurement():
    a = Apple((0.0, 0.0, 0.60), 0.04, side_band(0.2))
    color, depth, masks = render([a], depth_dropout=0.4)
    vm = measure_view(color, depth, masks[0], INTR, P)
    assert vm.ratio_visible == pytest.approx(visible_truth(a), abs=0.03)


def test_realistic_sensor_noise():
    """D435 수준 잡음 (컬러 σ3, 깊이 σ1.5 mm) + 깊이 결측 20%에서도 비율·크기·분할이 유지된다."""
    apples = [Apple((-0.10, 0.02, 0.62), 0.037, side_band(0.3)), Apple((0.08, -0.02, 0.62), 0.043, top_cap(0.6))]
    color, depth, masks = render(apples, card_rect=CARD_ROI, color_noise=3.0, depth_noise_m=0.0015,
                                 depth_dropout=0.2, seed=3)
    dets = segment_apples(color, depth, INTR)
    assert len(dets) == 2
    assert [d.det_id for d in dets] == [1, 2]     # 한 줄로 놓인 사과는 왼→오 순서
    for d, a in zip(dets, apples):
        vm = measure_view(color, depth, d.mask, INTR, P)
        assert vm.ratio_visible == pytest.approx(visible_truth(a), abs=0.04)
        assert vm.sphere.radius == pytest.approx(a.radius, rel=0.06)


def test_specular_highlight_is_excluded():
    a = Apple((0.0, 0.0, 0.60), 0.04, lambda n: np.ones(len(n), bool))     # 전체 착색
    color, depth, masks = render([a], highlight_on=0)
    vm = measure_view(color, depth, masks[0], INTR, P)
    assert vm.ratio_visible > 0.97              # 하이라이트를 비착색으로 세지 않는다


def test_color_ratio_interval_uses_top_view_error():
    a = Apple((0.0, 0.0, 0.60), 0.04, half_red)
    color, depth, masks = render([a])
    snap = SceneSnapshot(3, Source.TOP, CaptureMode.FULL, color, depth, INTR, 0.0)
    det = AppleDetection2D(3, 1, (0, 0, 1, 1), 1.0, masks[0])
    cr = color_ratio(snap, det, apple_id=7, p=P)
    assert (cr.scene_id, cr.apple_id, cr.inspect_mode) == (3, 7, InspectMode.TOP_SINGLE)
    assert cr.ratio_hi - cr.ratio_lo == pytest.approx(2 * P.top_view_error, abs=1e-9)
    assert cr.ratio_lo <= visible_truth(a) <= cr.ratio_hi


# ---------------------------------------------------------------- 모드 B: 다면 누적

def inhand_views(apple_pattern):
    """탑뷰 1장 + 툴 수평(사과 90° 기울임) 후 J6 90°씩 4회."""
    poses = [np.eye(3)] + [rot_x(np.pi / 2) @ rot_z(k * np.pi / 2) for k in range(4)]
    views = []
    for R in poses:
        a = Apple((0.0, 0.0, 0.45), 0.04, apple_pattern, R)
        color, depth, masks = render([a])
        views.append(InHandView(color, depth, INTR, masks[0], R))
    return views


@pytest.mark.parametrize("pattern", [side_band(0.3), half_red, top_cap(0.0)])
def test_multi_view_recovers_full_surface_ratio(pattern):
    """보지 못한 극(약 9%)의 착색이 한쪽으로 치우치면 그만큼 오차가 생길 수 있다 — 판정 구간이 이를 포함한다."""
    cr = accumulate_views(inhand_views(pattern), P)
    truth = full_truth(pattern)
    assert cr.coverage > 0.85
    assert abs(cr.ratio - truth) <= (1 - cr.coverage) + 0.02
    assert cr.ratio_lo <= truth <= cr.ratio_hi
    assert cr.view_disagreement < 0.1           # 같은 사과를 본 뷰끼리 일관
    assert cr.inspect_mode == InspectMode.IN_HAND_MULTI and cr.num_views == 5


def test_top_view_alone_can_mislead_but_multi_view_does_not():
    """빨간 면이 아래로 놓인 사과: 탑뷰는 거의 비착색, 다면 검사는 실제 비율을 찾는다."""
    bottom_red = lambda n: n[:, 2] > 0.0          # 사과 좌표 +z = 테이블 쪽
    top = measure_view(*_single(bottom_red), INTR, P)
    multi = accumulate_views(inhand_views(bottom_red), P)
    assert top.ratio_visible < 0.05
    assert multi.ratio > 0.4 and multi.ratio_lo <= 0.5 <= multi.ratio_hi


def _single(pattern):
    color, depth, masks = render([Apple((0.0, 0.0, 0.60), 0.04, pattern)])
    return color, depth, masks[0]


# ---------------------------------------------------------------- 임시 분할 · 임계 제안

def test_segment_apples_finds_each_apple_and_ignores_card():
    apples = [Apple((-0.12, 0.0, 0.60), 0.038, half_red), Apple((0.0, 0.0, 0.60), 0.042, half_red),
              Apple((0.12, 0.0, 0.60), 0.035, half_red)]
    color, depth, masks = render(apples, card_rect=CARD_ROI)
    dets = segment_apples(color, depth, INTR)
    assert len(dets) == 3
    for d, m in zip(dets, masks):               # 왼→오 순서
        iou = (d.mask & m).sum() / (d.mask | m).sum()
        assert iou > 0.85


def test_numbering_follows_rows_then_columns():
    """2줄 × 3개, 줄마다 높이가 조금씩 어긋나도 위 줄 왼→오, 아래 줄 왼→오 순서."""
    xs, ys = (-0.13, 0.0, 0.13), (-0.06, 0.07)
    jitter = (0.012, -0.010, 0.006)
    apples = [Apple((x, y + j, 0.60), 0.035, half_red) for y in ys for x, j in zip(xs, jitter)]
    color, depth, masks = render(apples)
    dets = segment_apples(color, depth, INTR)
    assert len(dets) == 6
    for d, m in zip(dets, masks):
        assert (d.mask & m).sum() / (d.mask | m).sum() > 0.85


def test_suggest_hue_max_falls_between_red_and_yellow():
    color, depth, masks = render([Apple((0.0, 0.0, 0.60), 0.04, half_red)])
    lab = bgr_to_lab(color[masks[0]])
    h, c = hue_chroma(lab)
    th = suggest_hue_max(h, c, P.min_chroma)
    red_h = hue_chroma(bgr_to_lab(np.array([[[38, 30, 175]]], np.uint8)))[0].item()
    yel_h = hue_chroma(bgr_to_lab(np.array([[[75, 190, 205]]], np.uint8)))[0].item()
    assert red_h < th < yel_h


# ---------------------------------------------------------------- 등급 · 리젝트

def cr_(ratio, lo=None, hi=None, coverage=0.25, mode=InspectMode.TOP_SINGLE, dis=0.0):
    e = P.top_view_error
    return ColorRatio(1, 1, mode, ratio, ratio - e if lo is None else lo, ratio + e if hi is None else hi,
                      coverage, dis, 1, "fuji")


def apple_(diam_mm=75, conf=0.9, occluded=False):
    return AppleState(1, 1, 1, (0, 0, 0), diam_mm / 1000, 0, 0.02, 0.001, conf, 1.0, 0.0, occluded)


def test_grade_rule_is_asymmetric():
    T = P.coloring_threshold
    assert decide(cr_(T + P.delta_up + P.top_view_error + 0.01), P) == (Grade.UPPER, RejectReason.NONE)
    assert decide(cr_(T + P.delta_mid - P.top_view_error - 0.01), P) == (Grade.MIDDLE, RejectReason.NONE)
    assert decide(cr_(T), P) == (Grade.REJECT, RejectReason.BORDERLINE)


@pytest.mark.parametrize("kwargs, cr, reason", [
    (dict(apple=apple_(diam_mm=105)), cr_(0.9), RejectReason.OVERSIZE),
    (dict(apple=apple_(diam_mm=25)), cr_(0.9), RejectReason.UNDERSIZE),
    (dict(apple=apple_(occluded=True)), cr_(0.9), RejectReason.OCCLUDED),
    (dict(), None, RejectReason.OCCLUDED),
    (dict(apple=apple_(conf=0.2)), cr_(0.9), RejectReason.LOW_CONF),
    (dict(light=LightingStatus(LightState.FAULT, 9.0, True)), cr_(0.9), RejectReason.LOW_CONF),
    (dict(), cr_(0.9, coverage=0.05), RejectReason.LOW_COVERAGE),
    (dict(), cr_(0.9, mode=InspectMode.IN_HAND_MULTI, dis=0.5), RejectReason.VIEW_DISAGREE),
])
def test_reject_reasons_follow_priority(kwargs, cr, reason):
    g = classify(cr, P, **kwargs)
    assert g.reject_reason == reason and g.grade == Grade.REJECT


def test_oversize_wins_over_low_confidence():
    assert classify(cr_(0.9), P, apple=apple_(diam_mm=110, conf=0.1)).reject_reason == RejectReason.OVERSIZE


@pytest.mark.parametrize("policy, ratio, final", [
    (InspectionPolicy.NEVER, 0.40, True),
    (InspectionPolicy.ALWAYS, 0.90, False),
    (InspectionPolicy.ADAPTIVE, 0.40, False),   # 경계 → 다면 검사 대상
    (InspectionPolicy.ADAPTIVE, 0.90, True),    # 확실한 상
])
def test_final_flag_by_policy(policy, ratio, final):
    assert classify(cr_(ratio), P, policy=policy).final is final


def test_multi_view_result_is_always_final():
    g = classify(cr_(0.40, mode=InspectMode.IN_HAND_MULTI), P, policy=InspectionPolicy.ALWAYS)
    assert g.final is True
