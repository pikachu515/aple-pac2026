"""Bounded M4 mission supervision around the existing verified scene runner.

Recovery readiness is a hardware proof callback, never inferred from a failure
name. The live adapter only grants it before any grasp command was issued.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import json
import logging
import math
import time

from aple.execution.pick_place_pipeline import SceneWorkPlan
from aple.orchestration.recovery import RecoveryAction
from aple.schema import (CaptureMode, CycleRecord, ExecResult, FailStage, FailureCode,
                         Grade, Phase, RejectReason, SceneSnapshot,
                         Source, TaskAction)

_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class MissionPlan:
    root: SceneSnapshot
    scene: object

    @property
    def timings(self):
        return self.scene.timings


@dataclass(frozen=True)
class MissionResult:
    records: tuple[CycleRecord, ...]
    halted: bool


class MissionController:
    def __init__(self, cfg, runner, capture, *, prepare_recovery=None,
                 observe_drop=None, recalibrate_light=None, event_sink=None,
                 clock=time.monotonic):
        self.cfg, self.runner, self.capture = cfg, runner, capture
        self.prepare_recovery = prepare_recovery
        self.observe_drop = observe_drop
        self.recalibrate_light = recalibrate_light
        self.event_sink, self.clock = event_sink, clock
        self.halted, self.state = False, 'IDLE'
        self.halt_reason = None
        self.placed, self.skipped = set(), set()
        self.ids = None
        self.events = []
        oc = cfg['orchestration']
        self.max_rescan = runner._recovery._max_rescan
        self.max_retry = runner._recovery._max_retry
        self.expected = cfg['workcell']['expected_apple_count']
        if isinstance(self.expected, bool) or not isinstance(self.expected, int) or self.expected < 0:
            raise ValueError('workcell.expected_apple_count must be a nonnegative integer')
        self._started = self.clock()
        self._event_path = Path(oc['logging']['csv_dir']) / 'mission_events.jsonl'
        self._active = None
        self._plans = []
        self._root_scene_id = 0

    def _emit(self, kind, **details):
        event = dict(index=len(self.events)+1, kind=kind, state=self.state,
                     elapsed_s=self.clock()-self._started, mission_scene_id=self._root_scene_id, **details)
        # Persist first: a failed sink cannot erase the operator-facing record.
        self._event_path.parent.mkdir(parents=True, exist_ok=True)
        with self._event_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False, allow_nan=False)+'\n')
        self.events.append(event)
        _LOG.warning('M4 %s: %s', kind, details)
        if self.event_sink is not None:
            try:
                self.event_sink(event)
            except Exception as exc:
                _LOG.error('M4 event sink failed: %s', type(exc).__name__)

    def _halt(self, reason):
        self.halted, self.state = True, 'HALTED'
        self.halt_reason = reason
        self.runner._execution_halted = True
        self._emit('ALERT', reason=reason, apple_id=self._active.apple.apple_id if self._active else 0,
                   holding_state='unknown', restart='inspect scene and start a new explicit run')

    def stop(self):
        """Stop before the next operation; ongoing moves retain HAL E-stop/deadline supervision."""
        self._halt('operator stop requested')

    def _ready(self, scene):
        if scene.blocked == 'lighting':
            return False
        ids = [a.apple_id for a in scene.apples]
        valid = (scene.snapshot.source == Source.TOP and scene.snapshot.capture_mode == CaptureMode.FULL
                 and math.isfinite(scene.snapshot.stamp_s)
                 and all(a.scene_id == scene.snapshot.scene_id for a in scene.apples)
                 and scene.snapshot.tray_clear and not any(a.occluded for a in scene.apples)
                 and len(ids) == len(set(ids)) and len(ids) == self.expected-len(self.placed))
        if self.ids is not None:
            valid = valid and set(ids) == self.ids-self.placed
        if valid and self.ids is None:
            self.ids = set(ids)
        return valid

    def _fresh(self, old):
        self.state = 'SCAN'
        started = self.clock()
        snap = self.capture()
        capture_duration = self.clock()-started
        if (not isinstance(snap, SceneSnapshot) or snap.source != Source.TOP
                or snap.capture_mode != CaptureMode.FULL or snap.scene_id <= old.scene_id
                or not math.isfinite(snap.stamp_s) or snap.stamp_s <= old.stamp_s):
            raise ValueError('M4 rescan requires a newer FULL TOP exposure')
        self._emit('RESCAN', scene_id=snap.scene_id)
        planned = self.runner.plan(snap)
        planned.timings['t_scan_s'] = planned.timings.get('t_scan_s', 0.) + capture_duration
        self._plans.append(planned)
        return planned

    def plan(self, snap):
        if self.state == 'FINISHED':
            self.placed.clear()
            self.skipped.clear()
            self.ids = None
        self.state = 'PLAN'
        self._root_scene_id = snap.scene_id
        planned = self.runner.plan(snap)
        self._plans = [planned]
        return MissionPlan(snap, planned)

    def execute(self, plan):
        self._started = self.clock()
        if self.halted:
            return MissionResult((self._failure_row(plan.root, FailureCode.ABORTED, FailStage.NONE),), True)
        current, rows = plan.scene, []
        rescans, retries = 0, {}
        while not self.halted:
            if not self._ready(current):
                self._emit('SCENE_NOT_READY', scene_id=current.snapshot.scene_id,
                           observed=len(current.apples), expected=self.expected-len(self.placed),
                           tray_clear=current.snapshot.tray_clear, blocked=current.blocked)
                if rescans >= self.max_rescan:
                    self._halt('scene completeness/lighting rescan budget exhausted')
                    break
                if current.blocked == 'lighting':
                    # No scan movement under unknown lighting. Calibration callback
                    # must explicitly establish restored lighting before returning True.
                    if self.recalibrate_light is None or self.recalibrate_light() is not True:
                        self._halt('lighting recalibration requires operator action')
                        break
                    self._emit('LIGHT_RECALIBRATED')
                try:
                    current = self._fresh(current.snapshot)
                except Exception as exc:
                    self._halt(f'rescan failed: {type(exc).__name__}: {exc}')
                    break
                rescans += 1
                continue
            if current.work is None:
                if self.ids-self.placed-self.skipped:
                    self._halt('planner returned no work for remaining apples')
                    break
                self.state = 'FINISHED'
                self._emit('FINISHED', placed=len(self.placed), skipped=len(self.skipped))
                break
            if (current.work.scene_id != current.snapshot.scene_id
                    or len(current.work.items) != len(current.apples)
                    or {i.apple.apple_id for i in current.work.items} != {a.apple_id for a in current.apples}):
                self._halt('planner omitted or duplicated an apple')
                break
            pending = [i for i in current.work.items if i.apple.apple_id not in self.placed|self.skipped]
            for item in pending:
                if item.task.action == TaskAction.SKIP:
                    self.skipped.add(item.apple.apple_id)
                    self._emit('SKIP', apple_id=item.apple.apple_id, reason='infeasible grasp or size')
                    part = replace(current, work=SceneWorkPlan(current.work.scene_id, (item,)))
                    rows.extend(self.runner.records(current.snapshot, part, part.work))
            pending = [i for i in pending if i.apple.apple_id not in self.skipped]
            if not pending:
                self.state = 'FINISHED'
                self._emit('FINISHED', placed=len(self.placed), skipped=len(self.skipped))
                break
            item = next((i for i in pending if i.task.action in
                         (TaskAction.PICK_PLACE, TaskAction.PICK_INSPECT_PLACE)
                         and (i.placement is None or i.placement.feasible)), None)
            if item is None:
                if rescans >= self.max_rescan:
                    self._halt('deferred apples or unavailable box cells remain after rescan budget')
                    break
                try:
                    current = self._fresh(current.snapshot)
                except Exception as exc:
                    self._halt(f'deferred rescan failed: {type(exc).__name__}: {exc}')
                    break
                rescans += 1
                continue
            self._active = item
            part = replace(current, work=SceneWorkPlan(current.work.scene_id, (item,)))
            self.state = 'EXECUTE'
            self._emit('EXECUTE', apple_id=item.apple.apple_id, scene_id=current.snapshot.scene_id)
            execution_started = self.clock()
            try:
                result = self.runner.execute(part)
                done = result.items[0]
            except Exception as exc:
                done = replace(item, execution=ExecResult(False,
                    FailureCode.TIMEOUT if isinstance(exc, TimeoutError) else FailureCode.ABORTED, Phase.IDLE),
                    timings={**item.timings, 't_total_s': self.clock()-execution_started})
                result = SceneWorkPlan(part.work.scene_id, (done,))
                rows.extend(self.runner.records(current.snapshot, part, result))
                self._halt(f'execution exception: {type(exc).__name__}: {exc}')
                break
            rows.extend(self.runner.records(current.snapshot, part, result))
            if self.halted:
                break
            ex = done.execution
            if ex is not None and ex.success:
                self.placed.add(item.apple.apple_id)
                self._active = None
                self._emit('PLACED', apple_id=item.apple.apple_id)
                if len(self.placed|self.skipped) == self.expected:
                    self.state = 'FINISHED'
                    self._emit('FINISHED', placed=len(self.placed), skipped=len(self.skipped))
                    break
                try:
                    post = getattr(self.runner, 'last_verified_snapshot', None)
                    if post is not None:
                        if post.scene_id <= current.snapshot.scene_id or post.stamp_s <= current.snapshot.stamp_s:
                            raise ValueError('verified snapshot must be newer than the executed scene')
                        current = self.runner.plan(post)
                        self._plans.append(current)
                    else:
                        current = self._fresh(current.snapshot)
                except Exception as exc:
                    self._halt(f'post-placement planning failed: {type(exc).__name__}: {exc}')
                    break
                continue
            code = FailureCode.ABORTED if ex is None else ex.failure_code
            aid = item.apple.apple_id
            count = retries.get(aid, 0)
            decision = self.runner._recovery.decide(failure_code=code, retry_count=count, rescan_count=rescans)
            self.state = 'RECOVER'
            self._emit('RECOVERY', apple_id=aid, failure_code=code.name,
                       action=decision.action.value, attempt=count,
                       phase=ex.phase_reached.name if ex else 'UNKNOWN')
            if code == FailureCode.DROP and self.observe_drop is None:
                self._halt('drop requires a fresh observation before recovery')
                break
            if code == FailureCode.DROP and self.observe_drop is not None:
                try:
                    observed = self.observe_drop()
                    if (not isinstance(observed, SceneSnapshot) or observed.source != Source.TOP
                            or observed.capture_mode != CaptureMode.FULL
                            or observed.scene_id <= current.snapshot.scene_id
                            or not math.isfinite(observed.stamp_s)
                            or observed.stamp_s <= current.snapshot.stamp_s):
                        raise ValueError('drop observation must be a fresh FULL TOP frame')
                    self._emit('DROP_OBSERVED', scene_id=observed.scene_id)
                except Exception as exc:
                    self._halt(f'drop observation failed: {type(exc).__name__}: {exc}')
                    break
            skip = decision.action == RecoveryAction.SKIP_AND_ALERT
            recoverable = decision.action in (RecoveryAction.RESCAN_AND_RETRY, RecoveryAction.FULL_RESCAN)
            if (not (recoverable or skip) or (not skip and count >= self.max_retry) or rescans >= self.max_rescan
                    or self.prepare_recovery is None):
                self._halt('failure requires operator inspection or retry budget is exhausted')
                break
            try:
                if self.prepare_recovery(done) is not True:
                    self._halt('empty gripper and safe rescan cannot be confirmed')
                    break
                if self.halted:
                    break
                # Only after positive hardware proof may a pre-grasp reservation
                # be abandoned. Never clear reservations after uncertain placement.
                if done.placement is not None and ex is not None and ex.phase_reached in (Phase.APPROACH, Phase.DESCEND):
                    self.runner.pipeline.release_placement_reservation(int(done.placement.box_id), done.placement.cell_index)
                if skip:
                    self.skipped.add(aid)
                    self._emit('SKIP_ALERT', apple_id=aid, reason='retry budget exhausted; empty state confirmed')
                self.runner._execution_halted = False
                current = self._fresh(current.snapshot)
            except Exception as exc:
                self._halt(f'recovery preparation failed: {type(exc).__name__}: {exc}')
                break
            if not skip:
                retries[aid] = count+1
            rescans += 1
        if self.halted:
            # This scene-level row records incompleteness even if earlier apples
            # succeeded; do not mislabel those successes as failed executions.
            rows.append(self._failure_row(plan.root, FailureCode.ABORTED, FailStage.NONE))
        # Count each perception/replan once, including scans of deferred apples.
        if rows:
            scan = sum(p.timings.get('t_scan_s', 0.) for p in self._plans)/len(rows)
            planning = sum(p.timings.get('t_plan_s', 0.) for p in self._plans)/len(rows)
            rows = [replace(r, scene_id=plan.root.scene_id, t_scan_s=scan, t_plan_s=planning,
                t_total_s=max(0., r.t_total_s-r.t_scan_s-r.t_plan_s)+scan+planning) for r in rows]
        return MissionResult(tuple(rows), self.halted)

    def records(self, snapshot, plan, result):
        if result is None:  # compatibility for an explicit dry-run caller
            rows = self.runner.records(plan.scene.snapshot, plan.scene, None)
            return [replace(r, scene_id=snapshot.scene_id) for r in rows]
        return list(result.records) or [self._failure_row(snapshot, FailureCode.NONE, FailStage.NONE, success=True)]

    def failure_records(self, snapshot, plan, stage, error):
        self._root_scene_id = 0 if snapshot is None else snapshot.scene_id
        self._halt(f'{stage} exception: {type(error).__name__}: {error}')
        code = FailureCode.TIMEOUT if isinstance(error, TimeoutError) else FailureCode.ABORTED
        return [self._failure_row(snapshot, code, FailStage.D1 if stage in ('capture', 'plan') else FailStage.NONE)]

    def _failure_row(self, snapshot, code, stage, *, success=False):
        nan = math.nan
        return CycleRecord(cycle_index=0, scene_id=0 if snapshot is None else snapshot.scene_id,
            apple_id=0, variety=self.runner._params.variety, inspection_policy=self.runner._policy,
            task_action=TaskAction.DEFER, backend=self.runner._backend, diameter_m=nan,
            ratio=nan, ratio_lo=nan, ratio_hi=nan, coverage=nan, grade=Grade.REJECT,
            reject_reason=RejectReason.NONE, target_box=Grade.REJECT, target_cell=-1,
            grasp_force_set_n=nan, planned_drop_m=nan, settle_error_m=nan,
            t_scan_s=0., t_plan_s=0., t_pick_s=nan, t_inspect_s=nan, t_place_s=nan,
            t_total_s=0., success=success, failure_code=code, fail_stage=stage)
