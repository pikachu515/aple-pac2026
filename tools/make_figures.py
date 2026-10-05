"""발표용 그림 — 판정 모듈의 설계 근거를 합성 장면으로 보여 준다 (역할 C).

실측 데이터가 생기면 reject_curve 등은 color_session.py curve 결과로 바꾼다.
그림 글자는 영어로 둔다 (Ubuntu 기본 글꼴에 한글이 없어 깨질 수 있다).

    python tools/make_figures.py                 # data/figures/ 에 PNG 6장
    python tools/make_figures.py --out <폴더>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aple.config import load_config  # noqa: E402
from aple.inspection.color import apply_gain, calibrate_card  # noqa: E402
from aple.inspection.coloration import InHandView, accumulate_views, measure_view, params_from_config  # noqa: E402
from aple.inspection.evaluation import pick_delta_up, plot_curve, reject_curve  # noqa: E402
from aple.inspection.inhand import expected_coverage  # noqa: E402
from aple.schema import Grade  # noqa: E402
from tests.synthetic import (INTR, Apple, blush, full_truth, half_red, render, rot_x, rot_z, top_cap,  # noqa: E402
                             visible_truth)

RED, GRAY, BLUE, GREEN = "#c0392b", "#7f8c8d", "#2c6fbb", "#2e8b57"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _overlay(color, vm):
    img = color[..., ::-1].astype(float).copy()
    img[vm.valid & ~vm.colored] = 0.45 * img[vm.valid & ~vm.colored] + [0, 40, 140]
    img[vm.colored] = 0.45 * img[vm.colored] + [140, 0, 0]
    return np.clip(img, 0, 255).astype(np.uint8)


def _crop(img, c=(240, 320), h=70):
    return img[c[0] - h:c[0] + h, c[1] - h:c[1] + h]


def fig_area_weighting(p, out):
    plt = _plt()
    a = Apple((0.0, 0.0, 0.60), 0.04, top_cap(0.8))
    color, depth, masks = render([a])
    vm = measure_view(color, depth, masks[0], INTR, p)
    naive = vm.colored.sum() / vm.valid.sum()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 3.6), gridspec_kw={"width_ratios": [1, 1.3]})
    ax0.imshow(_crop(_overlay(color, vm)))
    ax0.set_title("red = colored, blue = not colored")
    ax0.axis("off")
    vals = [visible_truth(a), vm.ratio_visible, naive]
    ax1.bar(["true (surface area)", "ours (area-weighted)", "pixel count"], [v * 100 for v in vals],
            color=[GRAY, BLUE, RED])
    for i, v in enumerate(vals):
        ax1.text(i, v * 100 + 1, f"{v * 100:.1f}%", ha="center")
    ax1.set_ylabel("colored ratio of visible surface (%)")
    ax1.set_title("Rim pixels are tilted surface: pixel count over-weights the top", fontsize=10)
    ax1.set_ylim(0, max(vals) * 130)
    fig.tight_layout()
    fig.savefig(out / "fig1_area_weighting.png", dpi=140)
    plt.close(fig)


def fig_coverage(out):
    plt = _plt()
    R_top = [np.eye(3)]
    labels, covs = ["top view only"], [expected_coverage(R_top)]
    for deg in (45, 60, 90):
        Rs = R_top + [rot_x(np.radians(deg)) @ rot_z(k * np.pi / 2) for k in range(4)]
        labels.append(f"+ tilt {deg} deg, wrist x4")
        covs.append(expected_coverage(Rs))
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.bar(labels, [c * 100 for c in covs], color=[RED, GRAY, GRAY, GREEN])
    for i, c in enumerate(covs):
        ax.text(i, c * 100 + 1.5, f"{c * 100:.0f}%", ha="center")
    ax.set_ylabel("observed surface (%)")
    ax.set_ylim(0, 105)
    ax.set_title("How much of the apple can one camera see?")
    fig.tight_layout()
    fig.savefig(out / "fig2_coverage.png", dpi=140)
    plt.close(fig)


def fig_graycard(p, out):
    plt = _plt()
    card = (20, 20, 80, 60)
    apples = [Apple((-0.06, 0.0, 0.60), 0.04, blush())]               # 적색→황색으로 서서히 바뀌는 사과
    ref, depth, masks = render(apples, card_rect=card)
    warm, _, _ = render(apples, card_rect=card, tint=(1.35, 1.0, 0.65))
    fixed = apply_gain(warm, calibrate_card(warm, card))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, img, title in zip(axes, (ref, warm, fixed), ("reference light", "warm light (raw)", "after gray card")):
        ax.imshow(img[:300, :400, ::-1])
        r = measure_view(img, depth, masks[0], INTR, p).ratio_visible
        ax.set_title(f"{title}\ncolored {r * 100:.0f}%")
        ax.axis("off")
    fig.suptitle("Warm light makes yellow look red -> gray card restores the reference")
    fig.tight_layout()
    fig.savefig(out / "fig3_graycard.png", dpi=140)
    plt.close(fig)


def fig_reject_curve(p, out):
    rng = np.random.default_rng(1)
    full = rng.uniform(0.1, 0.8, 60)
    top = np.clip(full + rng.normal(0, 0.08, 60), 0, 1)
    truth = [Grade.UPPER if f >= p.coloring_threshold else Grade.MIDDLE for f in full]
    curve = reject_curve(list(top), truth, p)
    plot_curve(curve, pick_delta_up(curve), out / "fig4_reject_curve_demo.png")


def fig_shadow(p, out):
    plt = _plt()
    a = Apple((0.0, 0.0, 0.60), 0.04, half_red)
    base, depth, masks = render([a], color_noise=3.0, seed=1)
    shaded, _, _ = render([a], shadow_rect=(0, 0, 315, 480), shadow_factor=0.04, color_noise=3.0, seed=1)
    truth = measure_view(base, depth, masks[0], INTR, p).ratio_visible
    vm = measure_view(shaded, depth, masks[0], INTR, p)
    keep = vm.valid.copy()
    from aple.inspection.color import bgr_to_lab
    dark = keep & (bgr_to_lab(shaded)[..., 0] < p.shadow_max_L)
    excluded = (vm.colored & ~dark).sum() / max((keep & ~dark).sum(), 1)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    vals = [truth, excluded, vm.ratio_visible]
    ax.bar(["no shadow", "shadow pixels removed", "shadow kept (ours)"], [v * 100 for v in vals],
           color=[GRAY, RED, BLUE])
    for i, v in enumerate(vals):
        ax.text(i, v * 100 + 1.5, f"{v * 100:.0f}%", ha="center")
    ax.set_ylabel("colored ratio (%)")
    ax.set_title(f"Removing shadow biases the result -> we re-shoot instead\n(shadow on {vm.shadow_fraction * 100:.0f}% of apple)")
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(out / "fig5_shadow.png", dpi=140)
    plt.close(fig)


def fig_top_vs_multi(p, out):
    plt = _plt()
    bottom_red = lambda n: n[:, 2] > 0.0                       # 빨간 면이 테이블 쪽
    color, depth, masks = render([Apple((0.0, 0.0, 0.60), 0.04, bottom_red)])
    top = measure_view(color, depth, masks[0], INTR, p).ratio_visible
    views = []
    for R in [np.eye(3)] + [rot_x(np.pi / 2) @ rot_z(k * np.pi / 2) for k in range(4)]:
        c, d, m = render([Apple((0.0, 0.0, 0.45), 0.04, bottom_red, R)])
        views.append(InHandView(c, d, INTR, m[0], R))
    multi = accumulate_views(views, p)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    vals = [full_truth(bottom_red), top, multi.ratio]
    ax.bar(["true (whole surface)", "top view only", "multi-view (tilt + wrist)"], [v * 100 for v in vals],
           color=[GRAY, RED, GREEN])
    ax.errorbar([2], [multi.ratio * 100], yerr=[[(multi.ratio - multi.ratio_lo) * 100], [(multi.ratio_hi - multi.ratio) * 100]],
                fmt="none", ecolor="k", capsize=6)
    for i, v in enumerate(vals):
        ax.text(i + 0.12, v * 100 + 2, f"{v * 100:.0f}%")
    ax.set_ylabel("colored ratio (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Red side facing down: top view misses it, multi-view finds it")
    fig.tight_layout()
    fig.savefig(out / "fig6_top_vs_multiview.png", dpi=140)
    plt.close(fig)


def main(argv: list[str] | None = None) -> list[Path]:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "figures")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    p = params_from_config(load_config(), "fuji")
    fig_area_weighting(p, args.out)
    fig_coverage(args.out)
    fig_graycard(p, args.out)
    fig_reject_curve(p, args.out)
    fig_shadow(p, args.out)
    fig_top_vs_multi(p, args.out)
    files = sorted(args.out.glob("fig*.png"))
    for f in files:
        print(f)
    return files


if __name__ == "__main__":
    main()
