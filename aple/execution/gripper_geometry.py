"""Mounted finger adapters: explicitly identified mm geometry → SI runtime config.

Photo point-to-point distances are evidence, not axis vectors or a TCP offset.
The manufacturer's base opening range is not the assembled contact width range.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GripperAttachmentGeometry:
    tcp_offset_m: tuple[float, float, float]
    contact_width_range_m: tuple[float, float]
    finger_thickness_m: float
    finger_width_m: float
    grasp_height_offset_mm: float
    collision_spheres: tuple[tuple[tuple[float, float, float], float], ...]


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number in mm')
    return float(value)


def _vector(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count:
        raise ValueError(f'{name} must contain {count} coordinates in mm')
    return tuple(_number(v, name) for v in value)


def reference_distances_m(cfg):
    """Report all photo distances in meters, without assigning axis meanings."""
    reference = cfg.get('gripper_attachment', {}).get('reference', {})
    return {key: _number(value, f'point_distances_mm.{key}') / 1000
            for key, value in reference.get('point_distances_mm', {}).items()}


def attachment_geometry(cfg):
    data = cfg.get('gripper_attachment', {})
    installed = data.get('installed', False)
    if not isinstance(installed, bool):
        raise ValueError('gripper_attachment.installed must be true or false')
    if not installed:
        return None
    if data.get('mounting') != 'existing_moving_fingers':
        raise ValueError('only existing_moving_fingers adapters are supported')
    geometry = data.get('assembled', {})
    if geometry.get('tcp_axes_match_flange') is not True:
        raise ValueError('assembled.tcp_axes_match_flange must be confirmed true; rotation offsets are unsupported')
    tcp = _vector(geometry.get('tcp_offset_mm'), 3, 'assembled.tcp_offset_mm')
    widths = _vector(geometry.get('contact_width_range_mm'), 2, 'assembled.contact_width_range_mm')
    if not 0 <= widths[0] < widths[1]:
        raise ValueError('assembled.contact_width_range_mm must be nonnegative and increasing')
    thickness = _number(geometry.get('finger_thickness_mm'), 'assembled.finger_thickness_mm')
    width = _number(geometry.get('finger_width_mm'), 'assembled.finger_width_mm')
    if thickness <= 0 or width <= 0:
        raise ValueError('assembled finger envelope dimensions must be positive')
    height = _number(geometry.get('grasp_height_offset_mm'), 'assembled.grasp_height_offset_mm')
    spheres = geometry.get('collision_spheres_flange_mm')
    if not isinstance(spheres, list) or not spheres:
        raise ValueError('assembled.collision_spheres_flange_mm must cover the added adapters across all openings')
    converted = []
    for sphere in spheres:
        if not isinstance(sphere, dict):
            raise ValueError('collision sphere must specify center_mm and radius_mm')
        center = _vector(sphere.get('center_mm'), 3, 'collision sphere center_mm')
        radius = _number(sphere.get('radius_mm'), 'collision sphere radius_mm')
        if radius <= 0:
            raise ValueError('collision sphere radius_mm must be positive')
        converted.append((tuple(v / 1000 for v in center), radius / 1000))
    return GripperAttachmentGeometry(tuple(v / 1000 for v in tcp),
                                     tuple(v / 1000 for v in widths),
                                     thickness / 1000, width / 1000, height, tuple(converted))


def apply_attachment_geometry(cfg):
    """Override runtime targets only after mounted geometry is specified.

    No files are written. Inactive configs retain identity for existing callers.
    A single attachment grasp height applies to all size bins until per-size
    contact calibration is supplied through a future model.
    """
    geometry = attachment_geometry(cfg)
    if geometry is None:
        return cfg
    result = copy.deepcopy(cfg)
    result['workcell']['tcp_offset_m'] = list(geometry.tcp_offset_m)
    result['piper_web']['gripper_width_range_m'] = list(geometry.contact_width_range_m)
    fruit_range = result['size_bins']['gripper_range_mm']
    eligible = [max(float(fruit_range[0]), geometry.contact_width_range_m[0] * 1000),
                min(float(fruit_range[1]), geometry.contact_width_range_m[1] * 1000)]
    if eligible[0] >= eligible[1]:
        raise ValueError('assembled contact width does not overlap the configured apple size range')
    result['size_bins']['gripper_range_mm'] = eligible
    result.setdefault('grasp_planner', {}).update(
        finger_thickness_m=geometry.finger_thickness_m, finger_width_m=geometry.finger_width_m)
    for size_bin in result['size_bins']['bins']:
        size_bin['grasp_height_offset_mm'] = geometry.grasp_height_offset_mm
    return result
