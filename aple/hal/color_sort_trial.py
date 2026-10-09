"""Simulation-only colour sort: central apples → red to the left box, yellow to the right.

Apple colours stand in for the M2 grade; positions stand in for M1. Execution reuses
the shared MotionExecutor, the Piper trajectory adapter and the MuJoCo contact
transport, exactly as the single-apple trial does. No hardware is opened.
"""
from __future__ import annotations

import copy
import math
import secrets
import time
from pathlib import Path

import numpy as np
import yaml

from aple.hal.in_hand_simulation import apply_layout, radial_grasps, scene_settings
from aple.hal.pick_place_trial import PARK_X_M, _positive, _probe_apple, resolve_start_pose

COLORS = {'red': [.80, .08, .06, 1.], 'yellow': [.95, .80, .10, 1.]}


def load_sort(path):
    sort = yaml.safe_load(Path(path).read_text(encoding='utf-8'))['color_sort_trial']
    apples = sort['apples']
    count = apples['count']
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError('apples.count must be a positive integer')
    bounds = np.asarray(apples['region_xy_m'], float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or (bounds[:, 0] >= bounds[:, 1]).any():
        raise ValueError('apples.region_xy_m must be increasing finite X/Y bounds')
    for key in ('diameter_m', 'mass_kg', 'min_distance_m'):
        if not _positive(apples[key]):
            raise ValueError(f'apples.{key} must be positive')
    if set(sort['boxes']) != set(COLORS):
        raise ValueError('boxes must define exactly red and yellow')
    for color, box in sort['boxes'].items():
        if box['grade'] not in ('UPPER', 'MIDDLE', 'REJECT'):
            raise ValueError(f'boxes.{color}.grade must be a box name')
        if len(box['center_m']) != 2 or not np.isfinite(box['center_m']).all():
            raise ValueError(f'boxes.{color}.center_m must be finite [x, y]')
        grid = box['grid']
        if len(grid) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in grid):
            raise ValueError(f'boxes.{color}.grid must be two positive integers')
    if sort['boxes']['red']['grade'] == sort['boxes']['yellow']['grade']:
        raise ValueError('red and yellow need different boxes')
    if not isinstance(sort.get('set') or [], list):
        raise ValueError('set must be a list of KEY=VALUE strings')
    if len(sort['start_xy_m']) != 2:
        raise ValueError('start_xy_m must be [x, y]')
    return sort


def sort_config(cfg, sort, base_scene_path, *, seed=None):
    """In-memory virtual config: random central apples and colours, enlarged boxes."""
    cfg, settings = scene_settings(cfg, base_scene_path)
    seed = secrets.randbits(32) if seed is None else seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    cfg = copy.deepcopy(cfg)
    apples = sort['apples']
    if not apples['diameter_m'] < settings['preopen_m']:
        raise ValueError('apple diameter must be smaller than the virtual preopen width')
    boxes = cfg['workcell']['boxes']
    centers = {}
    for color, box in sort['boxes'].items():
        boxes[box['grade']]['grid'] = list(box['grid'])
        centers[box['grade']] = np.asarray(box['center_m'], float)
    for name, y in zip([n for n in boxes if n not in centers], (0.,)):
        centers[name] = np.array([PARK_X_M, y])
    names = list(boxes)
    rng = np.random.default_rng(seed)
    bounds = np.asarray(apples['region_xy_m'], float)
    radius = apples['diameter_m']/2
    spec = dict(diameter_m=apples['diameter_m'], mass_kg=apples['mass_kg'], red_fraction=.5)
    for attempt in range(sort.get('max_attempts', 500)):
        points = []
        for _ in range(apples['count']):
            for _ in range(200):
                p = rng.uniform(bounds[:, 0], bounds[:, 1])  # apple centres
                if all(np.linalg.norm(p-q) >= apples['min_distance_m'] for q in points):
                    points.append(p)
                    break
        if len(points) != apples['count']:
            continue
        out = apply_layout(cfg, settings, [spec]*len(points), points, [centers[n] for n in names],
                           seed=seed, layout_attempts=attempt+1)
        probes = [_probe_apple(p, radius) for p in points]
        for i, probe in enumerate(probes):
            probe.apple_id = i+1
        if all(g.collision_free for g in radial_grasps(probes, out)):
            break
    else:
        raise ValueError(f'Cannot place {apples["count"]} graspable apples in the region (seed={seed})')
    colors = [str(c) for c in rng.choice(sorted(COLORS), size=len(points))]
    for apple, color in zip(out['simulation']['apples'], colors):
        apple['rgba'] = COLORS[color]
        apple['color'] = color
    scene = out['simulation']['in_hand_scene']
    scene['inspection_marker'] = False
    scene['align_before_box_transfer'] = False
    tilt = sort.get('transfer_tilt_rad')
    scene['tilt_rad'] = math.radians(out['grasp_planner']['tool_down_rpy_deg'][1]) if tilt is None else tilt
    if sort.get('transfer_height_m') is not None:
        scene['transfer_height_m'] = sort['transfer_height_m']
    out['simulation']['color_sort'] = dict(
        seed=seed, layout_attempts=attempt+1, colors=colors,
        boxes={c: dict(b, center_m=list(map(float, b['center_m']))) for c, b in sort['boxes'].items()},
        start_xy_m=list(sort['start_xy_m']), start_joint_rad=sort.get('start_joint_rad'))
    return out


def use_hardware_motion(cfg, hardware_cfg):
    """Copy the real arm's path settings (execution.transfer, Cartesian step) into a virtual config."""
    out = copy.deepcopy(cfg)
    out['execution']['transfer'] = copy.deepcopy(hardware_cfg['execution']['transfer'])
    out['execution']['trajectory']['max_cartesian_step_m'] = hardware_cfg['execution']['trajectory']['max_cartesian_step_m']
    return out


class RealtimeClock:
    """Count computer planning time as the arm holding still, so simulated time
    equals the elapsed time of a real arm driven by the same planning code.

    Wall time spent outside physics stepping (IK, segment checks, route choice)
    is inserted as holding physics before the next arm/gripper command.
    """

    def __init__(self):
        self.origin = None
        self.physics_wall_s = 0.
        self.waited_s = 0.

    def attach(self, transport):
        step = transport.step

        def timed_step(seconds, _step=step):
            began = time.perf_counter()
            try:
                return _step(seconds)
            finally:
                self.physics_wall_s += time.perf_counter()-began
        transport.step = timed_step
        for name in ('move_joint_goal', 'set_gripper_width'):
            original = getattr(transport, name)

            def paced(*args, _original=original, **kwargs):
                self.catch_up(transport)
                return _original(*args, **kwargs)
            setattr(transport, name, paced)

    def start(self):
        self.origin = time.perf_counter()
        self.physics_wall_s = self.waited_s = 0.

    def catch_up(self, transport):
        if self.origin is None:
            return
        computing = time.perf_counter()-self.origin-self.physics_wall_s
        hold = computing-self.waited_s
        if hold > transport.model.opt.timestep:
            self.waited_s += hold
            transport.step(hold)


class FrameRecorder:
    """Joint/object positions at the transport's frame rate for real-time playback."""

    def __init__(self, on_played=None):
        self.frames = []
        self.on_played = on_played
        self.played = None

    def __call__(self, model, data):
        self.frames.append((float(data.time), data.qpos.copy()))

    def play(self, model, *, rate=1.0, camera='side'):
        import mujoco
        import mujoco.viewer
        data = mujoco.MjData(model)
        with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
            wall0, sim0 = time.perf_counter(), self.frames[0][0]
            for sim_time, qpos in self.frames:
                if not viewer.is_running():
                    return
                delay = (sim_time-sim0)/rate-(time.perf_counter()-wall0)
                if delay > 0:
                    time.sleep(delay)
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
                viewer.sync()
            self.played = (self.frames[-1][0]-sim0, time.perf_counter()-wall0)
            if self.on_played:
                self.on_played(*self.played)
            while viewer.is_running():
                time.sleep(.1)


def run_sort(cfg, assumptions, *, ik=None, fk=None, sample_callback=None, clock=None):
    """Pick every apple, place it in its colour's box, return to the start pose."""
    from aple.execution.motion_executor import MotionExecutor
    from aple.execution.segment_validator import build_piper_segment_validator
    from aple.execution.trajectory import build_piper_kinematics
    from aple.hal.gripper_physics import evaluate_outcomes
    from aple.hal.in_hand_simulation import CentralInspectionAdapter, ContactTransport
    from aple.hal.simulation_timing import SimulationTiming
    from aple.planning.grasp_profile import select
    from aple.planning.placement_planner import PlacementPlanner
    from aple.schema import (AppleState, BoxOccupancy, Grade, GradeResult, InspectMode,
                             RejectReason)

    started = time.perf_counter()
    if ik is None or fk is None:
        ik, fk = build_piper_kinematics(cfg)
    retry = cfg['simulation']['in_hand_scene'].get('optimization', {}).get('ik_seed_retry_rad')
    if retry is not None:
        from aple.hal.simulation_optimization import SimulationIKRetry
        ik = SimulationIKRetry(ik, cfg, retry)
    validator = build_piper_segment_validator(cfg, fk)
    sort = cfg['simulation']['color_sort']
    start = resolve_start_pose(cfg, ik, sort)
    cfg['simulation']['initial_joint_rad'] = list(start)
    cfg['workcell']['poses']['home_joint_rad'] = list(start)
    transport = ContactTransport(cfg, assumptions, fk, sample_callback=sample_callback)
    timing = SimulationTiming(lambda: float(transport.data.time))
    transport.timing = timing
    # execution.transfer set: drive the hardware adapter and the mode-A pick_place path,
    # so the simulated motion is the motion the real arm code produces.
    real_path = (cfg['execution'].get('transfer') or {}).get('height_m') is not None
    if real_path:
        from aple.hal.piper_motion import PiperTrajectoryMotionAdapter
        adapter = PiperTrajectoryMotionAdapter(transport, cfg, ik, validator)
    else:
        adapter = CentralInspectionAdapter(transport, cfg, ik, validator)
    executor = MotionExecutor(adapter, cfg)

    scene_id = 1
    apples = []
    for i, physical in enumerate(cfg['simulation']['apples']):
        c = tuple(map(float, physical['center_m']))
        apples.append(AppleState(scene_id, i+1, i+1, c, 2*physical['radius_m'],
                                 c[2]+physical['radius_m'], .05, 0., 1., 1.))
    boxes = []
    for name, box in cfg['workcell']['boxes'].items():
        rows, cols = box['grid']
        floor = float(box['origin_m'][2])
        boxes.append(BoxOccupancy(scene_id, Grade[name], rows, cols, box['cell_pitch_m'],
                                  box['wall_height_m'], box['wall_height_m'], tuple(box['origin_m']),
                                  [floor]*(rows*cols), [False]*(rows*cols), True))
    planner = PlacementPlanner(cfg)  # one planner: reserved cells fill each box in turn
    candidates = {g.apple_id: g for g in radial_grasps(apples, cfg)}
    # Nearest apple to the start pose first, then nearest to the previous apple.
    order, here = [], np.asarray(sort['start_xy_m'], float)
    remaining = list(apples)
    while remaining:
        nxt = min(remaining, key=lambda a: np.linalg.norm(np.asarray(a.center_m[:2])-here))
        order.append(nxt)
        remaining.remove(nxt)
        here = np.asarray(nxt.center_m[:2])

    report = dict(mode='simulation_only_color_sort', success=False, errors=[], sort=sort,
                  motion_path='hardware_adapter_pick_place' if real_path else 'simulation_adapter',
                  realtime=clock is not None, apples=[])
    targets = [dict(center_m=list(a['center_m'])) for a in cfg['simulation']['apples']]
    if clock:
        clock.attach(transport)
        clock.start()
    t0, wall0 = float(transport.data.time), time.perf_counter()
    try:
        for apple in order:
            color = sort['colors'][apple.apple_id-1]
            grade = Grade[sort['boxes'][color]['grade']]
            result = GradeResult(scene_id, apple.apple_id, grade, .5, .5, .5, 1., 0., RejectReason.NONE,
                                 True, InspectMode.TOP_SINGLE, 'simulation')
            placement = planner.plan(apple, result, boxes)
            if not placement.feasible:
                raise RuntimeError(f'no free cell in the {color} box')
            targets[apple.apple_id-1]['center_m'] = list(placement.place_center_m)
            profile = select(apple, cfg)
            begin = float(transport.data.time)
            timing.apple = apple.apple_id
            if real_path:
                with timing.measure('pick_place'):
                    placed = executor.pick_place(apple, profile, candidates[apple.apple_id], placement)
                if not placed.success:
                    raise RuntimeError(f'apple {apple.apple_id} pick-place failed: '
                                       f'{placed.failure_code.name} at {placed.phase_reached.name}')
            else:
                with timing.measure('pick'):
                    picked = executor.pick_for_inspection(apple, profile, candidates[apple.apple_id])
                if not picked.success:
                    raise RuntimeError(f'apple {apple.apple_id} pick failed: {picked.failure_code.name}')
                with timing.measure('place'):
                    placed = executor.place_held_apple(apple, profile, picked.apple_in_tcp, placement)
                if not placed.success:
                    raise RuntimeError(f'apple {apple.apple_id} place failed: {placed.failure_code.name}')
            report['apples'].append(dict(apple=apple.apple_id, color=color, box=grade.name,
                                         cell_index=placement.cell_index,
                                         time_s=float(transport.data.time)-begin))
        timing.apple = None
        with timing.measure('return_home'):
            if not executor.move_named('home').success:
                raise RuntimeError('return to the start pose failed')
        if clock:
            clock.catch_up(transport)
        report['total_time_s'] = float(transport.data.time)-t0
        report['total_wall_time_s'] = time.perf_counter()-wall0
        transport.step(assumptions['final_settle_s'])
        outcomes = evaluate_outcomes(cfg, dict(apples=targets), assumptions, transport.initial,
                                     transport.peak, transport.positions())
        report['outcomes'] = outcomes
        report['success'] = all(o['success'] for o in outcomes)
        if not report['success']:
            report['errors'].append('an apple did not settle inside its box')
    except Exception as exc:
        report['errors'].append(str(exc))
    report.update(planning_wait_s=clock.waited_s if clock else 0.,
                  wall_time_s=time.perf_counter()-started, slips=transport.slips,
                  motion_errors=transport.motion_errors, trace_apples=targets)
    return report, transport
