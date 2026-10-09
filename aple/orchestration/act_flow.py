"""스크립트 + ACT 실행 흐름의 판단 로직 (10/10). 장치 없이 시험한다 — 실행은 tools/act_cycle.py.

흐름: 탑뷰 촬영·판정 → ACT ① 집기·다면 보여주기(hold 자세에서 멈춤) → 집은 사과의 등급 →
스크립트로 등급 상자 위(hover_<등급>) → ACT ② 아래 빈 칸에 놓기 → 스크립트로 접힌 자세.

ACT 는 Studio 추론으로 계속 돈다. 언제 멈출지는 팔로워 관절·그리퍼 폭을 보고 여기서 정한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from aple.schema import AppleState, Grade, GradeResult


@dataclass
class Sample:
    t: float
    joints: tuple[float, ...]          # rad
    width_m: float                     # 그리퍼 핑거 안쪽 간격


@dataclass
class PickWatcher:
    """ACT ① 이 사과를 집고 hold 자세에 와서 멈췄는지.

    집음 = 그리퍼가 열렸다가(최대 폭 − grasp_drop_m 이상 닫힘) 폭이 settle_s 동안 안정. closed_min_m 이 있으면 그보다
    넓어야 한다(완전히 닫힘 = 빈손). ⚠ 10/10 수제 핑거는 사과를 쥐어도 그리퍼 값이 완전히 닫힘(~1.8%)과 같다 →
    실기는 closed_min_m=None. 끝 = 집은 뒤 모든 관절이 hold 에서 hold_tol_rad 안이고 still_s 동안 거의 안 움직임."""
    hold_rad: Sequence[float]
    closed_min_m: float | None = None
    hold_tol_rad: float = 0.25
    still_rad: float = 0.02
    still_s: float = 1.0
    grasp_drop_m: float = 0.01
    settle_s: float = 0.5
    samples: list[Sample] = field(default_factory=list)
    grasp_index: int | None = None
    max_width_m: float = 0.0

    def update(self, s: Sample) -> bool:
        self.samples.append(s)
        if self.grasp_index is None:
            self.max_width_m = max(self.max_width_m, s.width_m)
            recent = [x for x in self.samples if s.t - x.t <= self.settle_s]
            if (s.t - self.samples[0].t >= self.settle_s and self._not_empty(s)
                    and self.max_width_m - s.width_m >= self.grasp_drop_m
                    and max(x.width_m for x in recent) - min(x.width_m for x in recent) < 0.002):
                self.grasp_index = len(self.samples) - 1 - (len(recent) - 1)   # 닫혀 안정되기 시작한 순간
            return False
        if not self._not_empty(s):
            return False                                   # 놓쳤다 — 끝으로 보지 않는다 (시간 제한이 처리)
        if any(abs(a - b) > self.hold_tol_rad for a, b in zip(s.joints, self.hold_rad)):
            return False
        window = [x for x in self.samples if s.t - x.t <= self.still_s]
        if s.t - window[0].t < self.still_s * 0.9:
            return False
        return all(max(x.joints[j] for x in window) - min(x.joints[j] for x in window) < self.still_rad
                   for j in range(len(s.joints)))

    def _not_empty(self, s: Sample) -> bool:
        return self.closed_min_m is None or s.width_m > self.closed_min_m

    @property
    def grasp_sample(self) -> Sample | None:
        return None if self.grasp_index is None else self.samples[self.grasp_index]

    @property
    def held_width_m(self) -> float | None:
        return None if self.grasp_index is None else self.samples[-1].width_m


@dataclass
class PlaceWatcher:
    """ACT ② 가 사과를 놓았는지: 그리퍼가 쥔 폭보다 open_m 이상 열린 뒤 after_s 가 지나면 끝 (빼는 동작 시간)."""
    held_width_m: float
    open_m: float = 0.015
    after_s: float = 1.5
    opened_at: float | None = None

    def update(self, s: Sample) -> bool:
        if self.opened_at is None and s.width_m >= self.held_width_m + self.open_m:
            self.opened_at = s.t
        return self.opened_at is not None and s.t - self.opened_at >= self.after_s


def picked_grade(tcp_xy: Sequence[float], apples: Sequence[AppleState], grades: Sequence[GradeResult],
                 max_dist_m: float = 0.06) -> tuple[int | None, Grade]:
    """그리퍼가 닫힌 순간 핑거 끝(x, y)과 가장 가까운 사과의 등급. 멀거나 등급이 없으면 (None, REJECT) — 모르면 리젝트."""
    by_id = {g.apple_id: g for g in grades}
    best = min(apples, key=lambda a: math.dist(tcp_xy, a.center_m[:2]), default=None)
    if best is None or math.dist(tcp_xy, best.center_m[:2]) > max_dist_m or best.apple_id not in by_id:
        return None, Grade.REJECT
    return best.apple_id, by_id[best.apple_id].grade
