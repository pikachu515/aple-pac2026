"""적재 칸·고도 계획 (설계서 placement_planner). 필수.

칸 선택     빈 칸(예약 제외)을 base_link 거리 내림차순 → 동률이면 벽 인접 우선 (Far-to-Near)
hover_z     = max(벽 높이, 인접 칸 최대 높이) + r + margin
release_z   = 칸 높이 + r + release_gap
planned_drop_m = release_gap ≤ max_drop_m (0.03), 아니면 feasible=False
계획한 칸은 검증 완료 전까지 예약.
"""

from __future__ import annotations

from aple.schema import AppleState, BoxOccupancy, GradeResult, PlacementPlan


class PlacementPlanner:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.reserved: set[tuple[int, int]] = set()   # (box_id, cell_index)

    def plan(self, apple: AppleState, grade: GradeResult, boxes: list[BoxOccupancy]) -> PlacementPlan:
        raise NotImplementedError

    def release(self, box_id: int, cell_index: int) -> None:
        self.reserved.discard((box_id, cell_index))
