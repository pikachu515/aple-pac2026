"""Nested wall/physics time measurements for offline simulation benchmarks."""
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import wraps
import time


def stage_timer(owner, stage):
    recorder = getattr(owner, 'timing', None)
    if recorder is None:
        recorder = getattr(getattr(owner, '_transport', None), 'timing', None)
    return recorder.measure(stage) if recorder is not None else nullcontext()


def timed_stage(stage):
    def decorate(method):
        @wraps(method)
        def measured(self, *args, **kwargs):
            with stage_timer(self, stage):
                return method(self, *args, **kwargs)
        return measured
    return decorate


class SimulationTiming:
    def __init__(self, simulation_clock, *, wall_clock=time.perf_counter):
        self.simulation_clock = simulation_clock
        self.wall_clock = wall_clock
        self.records = []
        self.stack = []
        self.apple = None

    @contextmanager
    def measure(self, stage):
        entry = dict(stage=stage, apple=self.apple, wall_start=self.wall_clock(),
                     sim_start=self.simulation_clock(), child_wall=0., child_sim=0.)
        self.stack.append(entry)
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            wall = self.wall_clock()-entry['wall_start']
            sim = self.simulation_clock()-entry['sim_start']
            self.stack.pop()
            self.records.append(dict(stage=stage, apple=entry['apple'], failed=failed,
                wall_time_s=wall, simulated_time_s=sim,
                exclusive_wall_time_s=max(0., wall-entry['child_wall']),
                exclusive_simulated_time_s=max(0., sim-entry['child_sim'])))
            if self.stack:
                self.stack[-1]['child_wall'] += wall
                self.stack[-1]['child_sim'] += sim

    def summary(self):
        totals = defaultdict(lambda: dict(calls=0, failures=0, wall_time_s=0.,
            simulated_time_s=0., exclusive_wall_time_s=0., exclusive_simulated_time_s=0.))
        for record in self.records:
            total = totals[record['stage']]
            total['calls'] += 1
            total['failures'] += int(record['failed'])
            for key in ('wall_time_s', 'simulated_time_s', 'exclusive_wall_time_s',
                        'exclusive_simulated_time_s'):
                total[key] += record[key]
        return dict(stages=dict(totals), records=self.records)
