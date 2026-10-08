"""탑뷰 판정 구간의 치우침 보정 — 10/7 실측(꼭지가 위면 탑뷰가 +4~+18 %p 높다)을 합성 사과로 재현해 검증한다.

어깨(위쪽)가 더 붉은 사과는 위에서 보면 전표면보다 착색이 높게 보인다. 이 치우침을 대칭 구간(±e)으로 덮으면
구간이 넓어져 경계 리젝트가 늘고, 덮지 못하면 '중'을 '상'으로 보낸다. 치우침을 빼고 남는 흔들림만 구간으로
잡으면 둘 다 줄어든다.
"""

from dataclasses import replace

import numpy as np
import pytest

from aple.inspection.coloration import accumulate_views, ratio_from_view, top_view_interval
from aple.inspection.grade_classifier import decide
from aple.schema import Grade
from tests.synthetic import Apple, full_truth, render, rot_x, rot_z
from tests.test_color_session import CARD, save
from tests.test_inspection import P
from tools import color_session as cs


def shoulder_blush(cut: float):
    """꼭지 쪽(평소 자세에서 위, 합성 장면의 카메라 쪽 = −z)이 붉다 — 햇빛 받는 어깨.

    cut 이 낮을수록 붉은 면이 넓다. 착색 경계는 뚜렷하게 둔다 (정답 비율과 측정이 같은 정의가 되도록).
    """
    return lambda n: -n[:, 2] > cut


CUTS = [-0.5, -0.3, -0.1, 0.1, 0.3]                    # 사과 5개, 착색 많음 → 적음
POSES = [np.eye(3), rot_x(np.pi)] + [rot_x(np.pi / 2) @ rot_z(k * np.pi / 2) for k in range(4)]


@pytest.fixture(scope="module")
def shoulder_truth(tmp_path_factory):
    root = tmp_path_factory.mktemp("bias") / "truth"
    for i, cut in enumerate(CUTS):
        for j, R in enumerate(POSES):
            color, depth, _ = render([Apple((0.0, 0.0, 0.60), 0.04, shoulder_blush(cut), R)], card_rect=CARD)
            save(root / f"apple_{i:02d}", j, color, depth)
    rows, s = cs.truth_root(root, None, cs.params("fuji"))
    return rows, s


def test_interval_shifts_center_by_bias_and_clips():
    q = replace(P, top_view_bias=0.12, top_view_error=0.07)
    est, lo, hi = top_view_interval(0.50, q)
    assert (est, lo, hi) == pytest.approx((0.38, 0.31, 0.45))
    assert top_view_interval(0.05, q) == pytest.approx((0.0, 0.0, 0.07))     # 0 아래로 내려가지 않는다
    assert top_view_interval(0.5, replace(P, top_view_bias=0.0, top_view_error=0.1)) == pytest.approx((0.5, 0.4, 0.6))


def test_upright_apples_are_overestimated_like_the_field(shoulder_truth):
    rows, s = shoulder_truth
    upright = [v for v in s["views"] if v["view"] == 0]
    assert all(v["diff"] > 0 for v in upright)                               # 꼭지가 위면 탑뷰가 항상 높다
    bias, err = s["first"]
    assert bias >= 0.04                                                      # 실측(+4~+18 %p)과 같은 방향 (크기는 무늬에 따라 다름)


def test_bias_interval_covers_truth_with_a_narrower_band(shoulder_truth):
    rows, s = shoulder_truth
    upright = [v for v in s["views"] if v["view"] == 0]
    bias, err = s["first"]
    sym = max(0.03, float(np.percentile([abs(v["diff"]) for v in upright], 90)))   # 치우침 무시 대칭 구간
    biased = replace(P, top_view_bias=bias, top_view_error=err)
    for v in upright:
        _, lo, hi = top_view_interval(v["top_ratio"], biased)
        assert lo - 0.02 <= v["full_ratio"] <= hi + 0.02                    # 정답을 덮는다 (이산화 여유 0.02)
    assert err < sym                                                         # 구간이 좁아진다
    assert err <= 0.5 * sym + 0.03


def test_fewer_borderline_rejects_and_no_middle_sent_up(shoulder_truth):
    """같은 사과들을 대칭 구간 vs 치우침 보정 구간으로 판정해 비교한다 (정답은 전표면)."""
    rows, s = shoulder_truth
    upright = [v for v in s["views"] if v["view"] == 0]
    bias, err = s["first"]
    sym = max(0.03, float(np.percentile([abs(v["diff"]) for v in upright], 90)))

    def judge(q):
        out = []
        for v in upright:
            est, lo, hi = top_view_interval(v["top_ratio"], q)
            vm_cr = ratio_from_view(type("VM", (), {"ratio_visible": v["top_ratio"], "coverage": 0.25})(), q)
            out.append(decide(vm_cr, q)[0])
        return out

    truth_grade = [Grade.UPPER if v["full_ratio"] >= P.coloring_threshold else Grade.MIDDLE for v in upright]
    g_sym = judge(replace(P, top_view_bias=0.0, top_view_error=sym))
    g_bias = judge(replace(P, top_view_bias=bias, top_view_error=err))
    up_errors = lambda g: sum(a == Grade.UPPER and t == Grade.MIDDLE for a, t in zip(g, truth_grade))
    assert up_errors(g_bias) == 0 and up_errors(g_sym) == 0                  # 둘 다 '중→상' 없음
    assert sum(g == Grade.REJECT for g in g_bias) <= sum(g == Grade.REJECT for g in g_sym)


def test_mode_b_is_not_shifted_by_top_view_bias():
    """다면 검사(모드 B)는 전표면을 직접 보므로 탑뷰 치우침 보정을 쓰지 않는다."""
    from tests.test_inhand import APPLE_IN_TCP, POSES as HAND_POSES, T_BASE_CAM, tcp_pose
    from aple.inspection.inhand import apple_center_cam, make_view, rot_cam_apple
    from aple.schema import CaptureMode, SceneSnapshot, Source
    from tests.synthetic import INTR

    pattern = shoulder_blush(0.0)
    views = []
    for Rb in HAND_POSES:
        T = tcp_pose(Rb)
        R, c = rot_cam_apple(T_BASE_CAM, T), apple_center_cam(T_BASE_CAM, T, APPLE_IN_TCP)
        color, depth, masks = render([Apple(tuple(c), 0.04, pattern, R)])
        snap = SceneSnapshot(1, Source.IN_HAND, CaptureMode.FULL, color, depth, INTR, 0.0)
        views.append(make_view(snap, masks[0], T_BASE_CAM, T, APPLE_IN_TCP, 0.04))
    a = accumulate_views(views, P)
    b = accumulate_views(views, replace(P, top_view_bias=0.15))
    assert (a.ratio, a.ratio_lo, a.ratio_hi) == (b.ratio, b.ratio_lo, b.ratio_hi)
    assert a.ratio_lo <= full_truth(pattern) <= a.ratio_hi


def test_field_like_bias_halves_rejects_without_sending_middle_up():
    """10/7 실측 분포(꼭지 위: 탑뷰 − 전표면 = +4~+18 %p)로 사과 400개를 만들어 대칭 구간과 비교한다."""
    rng = np.random.default_rng(7)
    full = rng.uniform(0.1, 0.9, 400)
    top = np.clip(full + rng.uniform(0.04, 0.18, 400), 0, 1)
    bias, err = cs._bias_and_error(list(top - full))
    sym = max(0.03, float(np.ceil(np.percentile(np.abs(top - full), 90) * 100) / 100))

    def run(q):
        g = [decide(ratio_from_view(type("VM", (), {"ratio_visible": t, "coverage": 0.25})(), q), q)[0] for t in top]
        up = sum(a == Grade.UPPER and f < P.coloring_threshold for a, f in zip(g, full))
        return up, sum(a == Grade.REJECT for a in g)

    up_now, _ = run(P)                                                    # 지금 기본값 ±0.10, 치우침 무시
    up_sym, rej_sym = run(replace(P, top_view_bias=0.0, top_view_error=sym))
    up_b, rej_b = run(replace(P, top_view_bias=bias, top_view_error=err))
    assert up_now > 0                                                     # 지금 기본값은 '중'을 '상'으로 보낸다
    assert up_sym == 0 and up_b == 0
    assert rej_b < 0.6 * rej_sym                                          # 리젝트가 크게 준다
