"""Reproducible, non-overlapping virtual layouts; no robot or file mutation."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import secrets

import numpy as np
import yaml


def load_random_settings(path):
    settings = yaml.safe_load(Path(path).read_text(encoding='utf-8'))['random_scene']
    for key in ('apple_center_xy_m', 'box_center_xy_m'):
        bounds = np.asarray(settings[key], dtype=float)
        if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or (bounds[:, 0] >= bounds[:, 1]).any():
            raise ValueError(f'{key} must contain finite increasing X/Y bounds in meters')
    for key in ('apple_min_separation_m', 'object_clearance_m', 'pregrasp_contact_gap_m'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be positive and finite')
    attempts = settings['max_layout_attempts']
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError('max_layout_attempts must be a positive integer')
    return settings


def randomize_scene(cfg, settings, *, seed=None):
    """Sample the whole scene, rejecting overlaps, with a bounded retry count.

    Box labels stay attached to their independently sampled positions. Planning
    and Studio consume this same config; no already-planned path is translated.
    """
    if seed is None:
        seed = secrets.randbits(32)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    rng = np.random.default_rng(seed)
    apples = cfg['simulation']['apples']
    boxes = cfg['workcell']['boxes']
    preopen = settings['pregrasp_contact_gap_m']
    lower, upper = cfg['piper_web']['gripper_width_range_m']
    if not lower <= preopen <= upper or any(2*a['radius_m'] >= preopen for a in apples):
        raise ValueError('pregrasp_contact_gap_m must fit the gripper range and exceed every apple diameter')
    apple_bounds = np.asarray(settings['apple_center_xy_m'])
    box_bounds = np.asarray(settings['box_center_xy_m'])
    margin = settings['object_clearance_m']
    tray = cfg['workcell']['tray']
    tray_center = np.asarray(tray['center_m'][:2])
    tray_half = np.asarray(tray['size_m']) / 2
    workspace = np.asarray(cfg['safety']['workspace_aabb_m'])[:2]
    # Grid is [rows, cols]; +X follows cols, +Y follows rows.
    half_sizes = [np.asarray(b['grid'][::-1]) * b['cell_pitch_m'] / 2 + .002 for b in boxes.values()]
    for attempt in range(1, settings['max_layout_attempts'] + 1):
        fruit_xy = rng.uniform(apple_bounds[:, 0], apple_bounds[:, 1], size=(len(apples), 2))
        valid = True
        for i, apple in enumerate(apples):
            radius = apple['radius_m']
            if (np.abs(fruit_xy[i] - tray_center) + radius + margin > tray_half).any():
                valid = False
            if (fruit_xy[i] - radius < workspace[:, 0]).any() or (fruit_xy[i] + radius > workspace[:, 1]).any():
                valid = False
            for j in range(i):
                distance = max(settings['apple_min_separation_m'], radius + apples[j]['radius_m'] + margin)
                if np.linalg.norm(fruit_xy[i] - fruit_xy[j]) < distance:
                    valid = False
        if valid:
            apple_attempts = attempt
            break
    else:
        raise ValueError(f'No non-overlapping random apple layout (seed={seed})')
    for attempt in range(1, settings['max_layout_attempts'] + 1):
        bin_xy = rng.uniform(box_bounds[:, 0], box_bounds[:, 1], size=(len(boxes), 2))
        valid = True
        for i, half in enumerate(half_sizes):
            if (bin_xy[i] - half < workspace[:, 0]).any() or (bin_xy[i] + half > workspace[:, 1]).any():
                valid = False
            for j in range(i):
                if (np.abs(bin_xy[i] - bin_xy[j]) < half + half_sizes[j] + margin).all():
                    valid = False
            for xy, apple in zip(fruit_xy, apples):
                distance_to_box = np.maximum(np.abs(xy - bin_xy[i]) - half, 0)
                if np.linalg.norm(distance_to_box) < apple['radius_m'] + margin:
                    valid = False
        if not valid:
            continue
        out = copy.deepcopy(cfg)
        out['simulation']['color_pattern_frame'] = 'camera_tangent'
        for size_bin in out['size_bins']['bins']:
            size_bin['open_width_mm'] = preopen * 1000
        out['grasp_planner']['open_margin_m'] = max((preopen - 2*a['radius_m']) / 2 for a in apples)
        for apple, xy in zip(out['simulation']['apples'], fruit_xy):
            apple['center_m'] = [*map(float, xy), cfg['simulation']['table_z_m'] + apple['radius_m']]
        for box, xy in zip(out['workcell']['boxes'].values(), bin_xy):
            half = np.asarray(box['grid'][::-1]) * box['cell_pitch_m'] / 2
            box['origin_m'] = [*map(float, xy - half), box['origin_m'][2]]
        out['simulation']['random_scene'] = dict(seed=seed, layout_attempts=apple_attempts + attempt,
                                                sampling_settings=copy.deepcopy(settings))
        return out
    raise ValueError(f'No non-overlapping random layout after {settings["max_layout_attempts"]} attempts (seed={seed})')
