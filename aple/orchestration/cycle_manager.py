"""Hardware-independent cycle orchestration (design: capture → plan → execute → reset)."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import TypeVar

from aple.orchestration.metrics_logger import MetricsLogger
from aple.schema import CycleRecord, SceneSnapshot

PlanT = TypeVar("PlanT")
ExecutionT = TypeVar("ExecutionT")


class CycleManager:
    """Coordinate one or more cycles using explicit, injectable stage providers.

    The caller owns perception/judgment and plan construction in ``plan_scene``.
    Robot work is only invoked through ``execute_scene`` and is never called in
    dry-run mode. In dry-run mode, ``dry_run_capture`` must provide a replay or
    synthetic snapshot rather than a live camera capture.
    """

    def __init__(
        self,
        cfg: dict,
        *,
        capture: Callable[[], SceneSnapshot],
        plan_scene: Callable[[SceneSnapshot], PlanT],
        make_records: Callable[
            [SceneSnapshot, PlanT, ExecutionT | None], Sequence[CycleRecord]
        ],
        execute_scene: Callable[[PlanT], ExecutionT] | None = None,
        dry_run_capture: Callable[[], SceneSnapshot] | None = None,
        reset: Callable[[], None] | None = None,
        logger: MetricsLogger | None = None,
        dry_run: bool = False,
    ) -> None:
        if not isinstance(cfg, dict):
            raise TypeError("cfg must be a dict")
        if not callable(capture) or not callable(plan_scene) or not callable(make_records):
            raise TypeError("capture, plan_scene, and make_records must be callable")
        if execute_scene is not None and not callable(execute_scene):
            raise TypeError("execute_scene must be callable when provided")
        if reset is not None and not callable(reset):
            raise TypeError("reset must be callable when provided")
        if logger is not None and not isinstance(logger, MetricsLogger):
            raise TypeError("logger must be a MetricsLogger when provided")
        if not isinstance(dry_run, bool):
            raise TypeError("dry_run must be bool")
        if dry_run and not callable(dry_run_capture):
            raise ValueError("dry_run requires a replay or synthetic dry_run_capture")
        if not dry_run and execute_scene is None:
            raise ValueError("live execution requires an execute_scene callback")

        self.cfg = cfg
        self.dry_run = dry_run
        self._capture = dry_run_capture if dry_run else capture
        self._plan_scene = plan_scene
        self._execute_scene = execute_scene
        self._make_records = make_records
        self._reset = reset
        self._logger = (
            logger if logger is not None else self._logger_from_config(cfg)
        )

    @staticmethod
    def _logger_from_config(cfg: dict) -> MetricsLogger | None:
        orchestration = cfg.get("orchestration")
        if orchestration is None:
            return None
        if not isinstance(orchestration, dict):
            raise TypeError("orchestration config must be a dict")
        logging_cfg = orchestration.get("logging")
        if logging_cfg is None:
            return None
        if not isinstance(logging_cfg, dict):
            raise TypeError("orchestration.logging config must be a dict")
        csv_dir = logging_cfg.get("csv_dir")
        if not isinstance(csv_dir, str) or not csv_dir.strip():
            raise ValueError("orchestration.logging.csv_dir must be a non-empty path")
        return MetricsLogger(Path(csv_dir) / "cycles.csv")

    def run(self, cycles: int = 1) -> list[CycleRecord]:
        """Run a bounded number of cycles and return their per-apple records.

        A failed robot operation is recorded by ``make_records``; it does not
        silently trigger another pick. Retry/rescan policy remains in the
        injected stage callbacks until the team agrees on its recovery contract.
        """
        if isinstance(cycles, bool) or not isinstance(cycles, int) or cycles <= 0:
            raise ValueError("cycles must be a positive integer")

        records: list[CycleRecord] = []
        for cycle_index in range(1, cycles + 1):
            snapshot = self._capture()
            if not isinstance(snapshot, SceneSnapshot):
                raise TypeError("capture provider must return a SceneSnapshot")

            plan = self._plan_scene(snapshot)
            execution = None
            if not self.dry_run:
                assert self._execute_scene is not None
                execution = self._execute_scene(plan)

            cycle_records = self._make_records(snapshot, plan, execution)
            if not isinstance(cycle_records, Sequence) or isinstance(
                cycle_records, (str, bytes)
            ):
                raise TypeError("make_records must return a sequence of CycleRecord")
            if not cycle_records:
                raise ValueError("make_records returned no CycleRecord for a completed cycle")

            for record in cycle_records:
                if not isinstance(record, CycleRecord):
                    raise TypeError("make_records may only return CycleRecord values")
                if record.scene_id != snapshot.scene_id:
                    raise ValueError(
                        "CycleRecord scene_id must match the captured SceneSnapshot"
                    )
                indexed = replace(record, cycle_index=cycle_index)
                if self._logger is not None:
                    self._logger.write(indexed)
                records.append(indexed)

            if not self.dry_run and self._reset is not None:
                self._reset()

        return records
