"""사이클 기록 (설계서 metrics_logger). 필수 — CSV 1행/사이클."""

from __future__ import annotations

import csv
from pathlib import Path

from aple.schema import CycleRecord


class MetricsLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, rec: CycleRecord) -> None:
        row = rec.to_dict()
        new = not self.path.exists()
        with self.path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
