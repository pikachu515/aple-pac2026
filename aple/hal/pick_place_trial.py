"""Simulation-only single-apple pick → box → return timing trial. No hardware.

The apple position stands in for the M1 output of another module; the box
position is sampled from a seed unless fixed. Execution reuses the shared
MotionExecutor, the Piper trajectory adapter and the MuJoCo contact transport.
"""
from __future__ import annotations

import math
import secrets
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from aple.hal.in_hand_simulation import apply_layout, radial_grasps, scene_settings

PARK_Y_M = (-0.22, 0.22)  # unused boxes sit behind the base, outside every route
PARK_X_M = -0.45
START_IK_SEEDS = ((0, .75, -.15, 0, -.51, 0), (0, 1.67, -1.1, 0, 1.08, 0), (0, 1.2, -1.0, 0, .6, 0),
                  (0, 1.0, -.5, 0, .2, 0))


def load_trial(path):
    return load_trial_dict(yaml.safe_load(Path(path).read_text(encoding='utf-8'))['pick_place_trial'])


def load_trial_dict(trial):
    """Validate the trial settings (also after command-line edits)."""
    apple = trial['apple']
    if len(apple['center_m']) != 2 or not np.isfinite(apple['center_m']).all():
        raise ValueError('apple.center_m must be finite [x, y] in base_link (m)')
    for key in ('diameter_m', 'mass_kg'):
        if not _positive(apple[key]):
            raise ValueError(f'apple.{key} must be positive and finite')
    error = apple.get('perception_error_m', [0., 0., 0.])
    if len(error) != 3 or not np.isfinite(error).all():
        raise ValueError('apple.perception_error_m must be three finite values')
    box = trial['box']
    if box['center_m'] is not None and (len(box['center_m']) != 2 or not np.isfinite(box['center_m']).all()):
        raise ValueError('box.center_m must be null or finite [x, y]')
    bounds = np.asarray(box['region_xy_m'], float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or (bounds[:, 0] >= bounds[:, 1]).any():
        raise ValueError('box.region_xy_m must be increasing finite X/Y bounds')
    radii = np.asarray(box['reach_radius_m'], float)
    if radii.shape != (2,) or not 0 < radii[0] < radii[1]:
        raise ValueError('box.reach_radius_m must be increasing positive radii')
    if not _positive(box['min_distance_from_apple_m']):
        raise ValueError('box.min_distance_from_apple_m must be positive')
    attempts = box['max_attempts']
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError('box.max_attempts must be a positive integer')
    if trial['other_boxes'] not in ('park', 'random'):
        raise ValueError("other_boxes must be 'park' or 'random'")
    tilt = trial.get('transfer_tilt_rad')
    if tilt is not None and (isinstance(tilt, bool) or not math.isfinite(tilt)):
        raise ValueError('transfer_tilt_rad must be null or finite')
    if not isinstance(trial.get('set') or [], list) or not all(isinstance(i, str) and '=' in i for i in trial.get('set') or []):
        raise ValueError('set must be a list of KEY=VALUE strings')
    height = trial.get('transfer_height_m')
    if height is not None and not _positive(height):
        raise ValueError('transfer_height_m must be null or positive')
    if trial.get('start_joint_rad') is not None:
        if len(trial['start_joint_rad']) != 6 or not np.isfinite(trial['start_joint_rad']).all():
            raise ValueError('start_joint_rad must be null or six finite joint angles')
    elif len(trial['start_xy_m']) != 2 or not np.isfinite(trial['start_xy_m']).all():
        raise ValueError('start_xy_m must be finite [x, y] when start_joint_rad is null')
    return trial


def trial_config(cfg, trial, base_scene_path, *, seed=None):
    """Return an in-memory virtual config with one apple and the sampled boxes."""
    cfg, settings = scene_settings(cfg, base_scene_path)
    seed = secrets.randbits(32) if seed is None else seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    target = trial['target_box']
    if target not in cfg['workcell']['boxes']:
        raise ValueError(f'unknown target_box {target}')
    apple = trial['apple']
    radius = apple['diameter_m']/2
    if not apple['diameter_m'] < settings['preopen_m']:
        raise ValueError('apple diameter must be smaller than the virtual preopen width')
    apple_xy = np.asarray(apple['center_m'], float)
    names = list(cfg['workcell']['boxes'])
    halves = {name: np.array(box['grid'][::-1])*settings['box_cell_pitch_m']/2 + .002
              for name, box in cfg['workcell']['boxes'].items()}
    box = trial['box']
    sampled = [target] if trial['other_boxes'] == 'park' else [target]+[n for n in names if n != target]
    rng = np.random.default_rng(seed)
    for attempt in range(box['max_attempts']):
        centers = {}
        for name in sampled:
            fixed = name == target and box['center_m'] is not None
            for _ in range(1 if fixed else 200):
                p = (np.asarray(box['center_m'], float) if fixed else
                     rng.uniform(np.asarray(box['region_xy_m'])[:, 0]+halves[name],
                                 np.asarray(box['region_xy_m'])[:, 1]-halves[name]))
                if _box_clear(p, halves[name], apple_xy, radius, centers, halves, box, settings, fixed):
                    centers[name] = p
                    break
            else:
                if fixed:
                    raise ValueError('box.center_m overlaps the apple or another box')
                break
        if len(centers) != len(sampled):
            continue
        for name, y in zip([n for n in names if n not in centers], PARK_Y_M):
            centers[name] = np.array([PARK_X_M, y])
        spec = dict(diameter_m=apple['diameter_m'], mass_kg=apple['mass_kg'], red_fraction=.5)
        out = apply_layout(cfg, settings, [spec], [apple_xy], [centers[n] for n in names],
                           seed=seed, layout_attempts=attempt+1)
        if radial_grasps([_probe_apple(apple_xy, radius)], out)[0].collision_free:
            break
        if box['center_m'] is not None and trial['other_boxes'] == 'park':
            raise ValueError('fixed box leaves no finger clearance around the apple')
    else:
        raise ValueError(f'Cannot place the box(es) around the apple (seed={seed})')
    scene = out['simulation']['in_hand_scene']
    scene['inspection_marker'] = False
    scene['align_before_box_transfer'] = False
    tilt = trial.get('transfer_tilt_rad')
    scene['tilt_rad'] = math.radians(out['grasp_planner']['tool_down_rpy_deg'][1]) if tilt is None else tilt
    if trial.get('transfer_height_m') is not None:
        scene['transfer_height_m'] = trial['transfer_height_m']
    out['simulation']['pick_place_trial'] = dict(
        target_box=target, seed=seed, layout_attempts=attempt+1,
        start_joint_rad=trial.get('start_joint_rad'), start_xy_m=trial.get('start_xy_m'),
        apple_center_m=[*map(float, apple_xy), radius],
        perception_error_m=list(apple.get('perception_error_m', [0., 0., 0.])),
        box_centers_m={n: [*map(float, c)] for n, c in centers.items()},
        other_boxes=trial['other_boxes'])
    return out


def split_overrides(items):
    """``trial.KEY=V`` edits the trial file; everything else edits the virtual config."""
    return ([i.removeprefix('trial.') for i in items if i.startswith('trial.')],
            [i for i in items if not i.startswith('trial.')])


def apply_overrides(cfg, assumptions, items):
    """Apply ``a.b.c=value`` edits; ``assumed_gripper.`` targets the physics assumptions."""
    for item in items:
        key, sep, raw = item.partition('=')
        if not sep or not key:
            raise ValueError(f'override must be key=value: {item}')
        value = yaml.safe_load(raw)
        root, path = (assumptions, key.split('.')[1:]) if key.startswith('assumed_gripper.') else (cfg, key.split('.'))
        if assumptions is None and root is None:
            raise KeyError(f'unknown setting {key}')
        node = root
        for part in path[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f'unknown setting {key}')
            node = node[part]
        if not isinstance(node, dict) or path[-1] not in node:
            raise KeyError(f'unknown setting {key}')
        node[path[-1]] = value
    scene = cfg.get('simulation', {}).get('in_hand_scene')
    for key in () if scene is None else ('empty_joint_velocity_rad_s', 'loaded_joint_velocity_rad_s', 'transfer_height_m'):
        if not _positive(scene[key]):
            raise ValueError(f'{key} must be positive and finite')


def run_trial(cfg, assumptions, *, ik=None, fk=None, sample_callback=None):
    """Pick the configured apple, place it in the target box, return; report times."""
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
    start = resolve_start_pose(cfg, ik)
    cfg['simulation']['initial_joint_rad'] = list(start)
    cfg['workcell']['poses']['home_joint_rad'] = list(start)
    transport = ContactTransport(cfg, assumptions, fk, sample_callback=sample_callback)
    timing = SimulationTiming(lambda: float(transport.data.time))
    transport.timing = timing
    executor = MotionExecutor(CentralInspectionAdapter(transport, cfg, ik, validator), cfg)

    trial = cfg['simulation']['pick_place_trial']
    physical = cfg['simulation']['apples'][0]
    seen = np.add(physical['center_m'], trial['perception_error_m'])
    scene_id = 1
    apple = AppleState(scene_id, 1, 1, tuple(map(float, seen)), 2*physical['radius_m'],
                       float(seen[2]+physical['radius_m']), 0.05, 0., 1., 1.)
    grade = Grade[trial['target_box']]
    result = GradeResult(scene_id, 1, grade, .5, .5, .5, 1., 0., RejectReason.NONE, True,
                         InspectMode.TOP_SINGLE, 'simulation')
    boxes = []
    for name, box in cfg['workcell']['boxes'].items():
        rows, cols = box['grid']
        floor = float(box['origin_m'][2])
        boxes.append(BoxOccupancy(scene_id, Grade[name], rows, cols, box['cell_pitch_m'],
                                  box['wall_height_m'], box['wall_height_m'], tuple(box['origin_m']),
                                  [floor]*(rows*cols), [False]*(rows*cols), True))
    profile = select(apple, cfg)
    candidate = radial_grasps([apple], cfg)[0]
    placement = PlacementPlanner(cfg).plan(apple, result, boxes)
    report = dict(mode='simulation_only_pick_place_return', success=False, errors=[],
                  trial=trial, placement=dict(box=grade.name, cell_index=placement.cell_index,
                  place_center_m=list(placement.place_center_m), feasible=placement.feasible))
    t0 = float(transport.data.time)
    try:
        if not placement.feasible:
            raise RuntimeError('no free cell in the target box')
        with timing.measure('pick'):
            picked = executor.pick_for_inspection(apple, profile, candidate)
        if not picked.success:
            raise RuntimeError(f'pick failed: {picked.failure_code.name} at {picked.phase_reached.name}')
        with timing.measure('place'):
            placed = executor.place_held_apple(apple, profile, picked.apple_in_tcp, placement)
        if not placed.success:
            raise RuntimeError(f'place failed: {placed.failure_code.name} at {placed.phase_reached.name}')
        with timing.measure('return_home'):
            back = executor.move_named('home')
        if not back.success:
            raise RuntimeError('return to the start pose failed')
        report['cycle_time_s'] = float(transport.data.time)-t0
        with timing.measure('final_settle'):
            transport.step(assumptions['final_settle_s'])
        outcome = evaluate_outcomes(cfg, dict(apples=[dict(center_m=list(placement.place_center_m))]),
                                    assumptions, transport.initial, transport.peak, transport.positions())[0]
        report['outcome'] = outcome
        report['success'] = bool(outcome['success'])
        if not outcome['success']:
            report['errors'].append('apple did not settle inside the target box')
    except Exception as exc:
        report['errors'].append(str(exc))
    stages = timing.summary()['stages']
    report['stages'] = {name: dict(simulated_time_s=v['simulated_time_s'], wall_time_s=v['wall_time_s'])
                        for name, v in stages.items()}
    report.update(simulated_time_s=float(transport.data.time)-t0, wall_time_s=time.perf_counter()-started,
                  slips=transport.slips, motion_errors=transport.motion_errors,
                  route_choices=transport.route_choices, ik_seed_failures=getattr(ik, 'failures', []))
    return report, transport


def resolve_start_pose(cfg, ik, trial=None):
    """Start/return joints: explicit, or IK at start_xy on the transfer plane."""
    from aple.execution.trajectory import ToolPose
    from aple.hal.in_hand_simulation import radial_orientation
    trial = trial or cfg['simulation']['pick_place_trial']
    if trial.get('start_joint_rad') is not None:
        return tuple(map(float, trial['start_joint_rad']))
    scene = cfg['simulation']['in_hand_scene']
    xy = trial['start_xy_m']
    pose = ToolPose((float(xy[0]), float(xy[1]), scene['transfer_height_m']),
                    radial_orientation(xy, scene['tilt_rad']))
    yaw = math.atan2(xy[1], xy[0])
    limits = np.asarray(cfg['safety']['joint_limits_rad'], float)
    # The numerical IK does not enforce limits; keep a margin for servo overshoot.
    for seed in START_IK_SEEDS:
        solution = ik.solve(pose, (yaw, *seed[1:]))
        if solution is not None and ((limits[:, 0]+.03 <= solution) & (solution <= limits[:, 1]-.03)).all():
            return tuple(map(float, solution))
    raise ValueError('No IK start pose at start_xy_m on the transfer plane; '
                     'lower transfer_height_m or set start_joint_rad')


def _box_clear(p, half, apple_xy, radius, centers, halves, box, settings, fixed):
    if not fixed and not box['reach_radius_m'][0] <= np.linalg.norm(p) <= box['reach_radius_m'][1]:
        return False
    if np.linalg.norm(p-apple_xy) < box['min_distance_from_apple_m'] and not fixed:
        return False
    if (abs(p-apple_xy) < half+radius+settings['clearance_m']).all():
        return False
    return not any((abs(p-q) < half+halves[n]+settings['clearance_m']).all() for n, q in centers.items())


def _probe_apple(xy, radius):
    return SimpleNamespace(scene_id=1, apple_id=1, center_m=(*map(float, xy), radius), diameter_m=2*radius)


def _positive(value):
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0
