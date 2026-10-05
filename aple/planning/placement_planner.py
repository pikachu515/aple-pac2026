"""적재 칸·고도 계획 (설계서 placement_planner). 필수.

칸 선택     빈 칸(예약 제외)을 base_link 거리 내림차순 → 동률이면 벽 인접 우선 (Far-to-Near)
hover_z     = max(벽 높이, 선택 칸·인접 칸 높이) + r + margin,
              그리고 release_z + margin 보다 낮지 않음
release_z   = 칸 높이 + r + release_gap
planned_drop_m = release_gap ≤ max_drop_m (0.03), 아니면 feasible=False
계획한 칸은 검증 완료 전까지 예약.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import replace

from aple.schema import AppleState, BoxOccupancy, GradeResult, PlacementPlan


class PlacementPlanner:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.reserved: set[tuple[int, int]] = set()   # (box_id, cell_index)
        self._last_valid_boxes: dict[int, BoxOccupancy] = {}

    def plan(self, apple: AppleState, grade: GradeResult, boxes: list[BoxOccupancy]) -> PlacementPlan:
        if grade.scene_id > apple.scene_id:
            raise ValueError("grade cannot come from a newer scene than the apple")
        if not grade.final:
            raise ValueError("cannot plan placement before the grade is final")
        if not math.isfinite(apple.diameter_m) or apple.diameter_m <= 0:
            raise ValueError("apple.diameter_m must be finite and positive")
        if len(apple.center_m) != 3 or not all(math.isfinite(value) for value in apple.center_m):
            raise ValueError("apple.center_m must contain three finite coordinates")

        placement_cfg = self.cfg["workcell"]["placement"]
        max_drop_m = float(placement_cfg["max_drop_m"])
        margin_m = float(placement_cfg["margin_m"])
        release_gap_m = float(placement_cfg["release_gap_m"])
        if (not all(math.isfinite(v) for v in (max_drop_m, margin_m, release_gap_m))
                or max_drop_m < 0 or margin_m < 0 or release_gap_m < 0):
            raise ValueError("placement distances must be finite and non-negative")

        radius_m = apple.diameter_m / 2.0
        if release_gap_m > max_drop_m:
            return self._infeasible(apple, grade)

        candidates: list[tuple[float, int, int, BoxOccupancy, float, float]] = []
        effective_boxes: list[BoxOccupancy] = []
        for box in boxes:
            if box.scene_id != apple.scene_id:
                raise ValueError(f"box {box.box_id} belongs to a different scene")
            cache_key = int(box.box_id)
            if box.valid:
                self._last_valid_boxes[cache_key] = deepcopy(box)
                effective_boxes.append(box)
            else:
                previous = self._last_valid_boxes.get(cache_key)
                if previous is not None:
                    effective_boxes.append(replace(previous, scene_id=box.scene_id))

        for box in effective_boxes:
            if box.box_id != grade.grade:
                continue
            cell_count = box.rows * box.cols
            if (box.rows <= 0 or box.cols <= 0 or not math.isfinite(box.cell_pitch_m)
                    or box.cell_pitch_m <= 0):
                raise ValueError(f"box {box.box_id} has invalid grid geometry")
            if len(box.heights_m) != cell_count or len(box.occupied) != cell_count:
                raise ValueError(f"box {box.box_id} occupancy arrays do not match its grid")
            if len(box.origin_m) != 3:
                raise ValueError(f"box {box.box_id}.origin_m must contain three coordinates")
            if not all(math.isfinite(value) for value in (*box.origin_m, *box.heights_m, box.wall_height_m)):
                raise ValueError(f"box {box.box_id} contains non-finite geometry")

            for cell_index, occupied in enumerate(box.occupied):
                if occupied or (int(box.box_id), cell_index) in self.reserved:
                    continue
                row, col = divmod(cell_index, box.cols)
                x = box.origin_m[0] + (col + 0.5) * box.cell_pitch_m
                y = box.origin_m[1] + (row + 0.5) * box.cell_pitch_m
                z = box.heights_m[cell_index]
                distance = math.sqrt(x * x + y * y + z * z)
                wall_distance = min(row, box.rows - 1 - row, col, box.cols - 1 - col)
                candidates.append((distance, wall_distance, cell_index, box, x, y))

        if not candidates:
            return self._infeasible(apple, grade)

        # Farthest from base_link first; ties prefer cells nearer a box wall.
        candidates.sort(key=lambda candidate: (-candidate[0], candidate[1], candidate[2]))
        _, _, cell_index, box, x, y = candidates[0]
        z = box.heights_m[cell_index]
        row, col = divmod(cell_index, box.cols)
        adjacent_heights = [
            box.heights_m[r * box.cols + c]
            for r, c in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
            if 0 <= r < box.rows and 0 <= c < box.cols
        ]
        hover_base_z = max([box.wall_height_m, z, *adjacent_heights])
        release_z = z + radius_m + release_gap_m
        hover_z = max(hover_base_z + radius_m + margin_m, release_z + margin_m)
        plan = PlacementPlan(
            scene_id=apple.scene_id,
            apple_id=apple.apple_id,
            box_id=grade.grade,
            cell_index=cell_index,
            place_center_m=(x, y, z + radius_m + release_gap_m),
            hover_z_m=hover_z,
            release_z_m=release_z,
            planned_drop_m=release_gap_m,
            feasible=True,
        )
        self.reserved.add((int(grade.grade), cell_index))
        return plan

    @staticmethod
    def _infeasible(apple: AppleState, grade: GradeResult) -> PlacementPlan:
        return PlacementPlan(
            scene_id=apple.scene_id,
            apple_id=apple.apple_id,
            box_id=grade.grade,
            cell_index=-1,
            place_center_m=apple.center_m,
            hover_z_m=apple.center_m[2],
            release_z_m=apple.center_m[2],
            planned_drop_m=0.0,
            feasible=False,
        )

    def release(self, box_id: int, cell_index: int) -> None:
        self.reserved.discard((box_id, cell_index))
