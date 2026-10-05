"""리젝트 곡선 (설계서 reject_curve_evaluator, 전략 S6-6).

판정 여유 delta_up 을 키우면 '상'으로 보내는 기준이 엄격해져 리젝트가 늘고 '중→상' 오분류가 준다.
그 관계를 표·그래프로 만들어 운용점을 고르고, 발표에서 "리젝트 몇 %로 오분류를 얼마까지 줄였다"를 보여 준다.

정답 등급은 전표면 착색비율(color_session truth) 또는 육안 판정표에서 온다.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from aple.inspection.coloration import InspectParams
from aple.inspection.grade_classifier import decide
from aple.schema import ColorRatio, Grade, InspectMode


@dataclass
class CurvePoint:
    delta_up: float
    reject_rate: float          # 전체 중 리젝트 비율
    up_error_rate: float        # 전체 중 '중'을 '상'으로 보낸 비율 (가장 비싼 오류)
    down_error_rate: float      # 전체 중 '상'을 '중'으로 보낸 비율
    n: int


def _cr(ratio: float, p: InspectParams) -> ColorRatio:
    e = p.top_view_error
    return ColorRatio(0, 0, InspectMode.TOP_SINGLE, ratio, max(0.0, ratio - e), min(1.0, ratio + e),
                      1.0, 0.0, 1, p.variety)


def reject_curve(measured: list[float], truth: list[Grade], p: InspectParams,
                 deltas: np.ndarray | None = None) -> list[CurvePoint]:
    """measured: 탑뷰 착색비율, truth: 정답 등급 (UPPER / MIDDLE)."""
    if deltas is None:
        deltas = np.round(np.arange(-0.10, 0.301, 0.01), 3)
    n = len(measured)
    out = []
    for d in deltas:
        q = replace(p, delta_up=float(d))
        pred = [decide(_cr(r, q), q)[0] for r in measured]
        rej = sum(g == Grade.REJECT for g in pred)
        up = sum(g == Grade.UPPER and t == Grade.MIDDLE for g, t in zip(pred, truth))
        down = sum(g == Grade.MIDDLE and t == Grade.UPPER for g, t in zip(pred, truth))
        out.append(CurvePoint(float(d), rej / n, up / n, down / n, n))
    return out


def pick_delta_up(curve: list[CurvePoint], max_up_error: float = 0.0) -> CurvePoint:
    """'중→상' 오류가 max_up_error 이하인 점 중 리젝트가 가장 적은 점 (같으면 delta_up 이 작은 쪽)."""
    ok = [c for c in curve if c.up_error_rate <= max_up_error + 1e-12]
    if not ok:
        return min(curve, key=lambda c: (c.up_error_rate, c.reject_rate))
    return min(ok, key=lambda c: (c.reject_rate, c.delta_up))


def plot_curve(curve: list[CurvePoint], chosen: CurvePoint | None, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rej = [c.reject_rate * 100 for c in curve]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    ax.plot(rej, [c.up_error_rate * 100 for c in curve], "o-", color="#c0392b", ms=3,
            label="middle sent to UPPER (costly)")
    ax.plot(rej, [(c.up_error_rate + c.down_error_rate) * 100 for c in curve], "s--", color="#555", ms=3,
            label="all misgrades")
    if chosen is not None:
        ax.plot([chosen.reject_rate * 100], [chosen.up_error_rate * 100], "*", ms=14, color="#1f77b4",
                label=f"chosen delta_up = {chosen.delta_up:+.2f}")
    ax.set_xlabel("reject rate (%)")
    ax.set_ylabel("residual misgrade rate (%)")
    ax.set_title(f"Reject curve (n = {curve[0].n})")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
