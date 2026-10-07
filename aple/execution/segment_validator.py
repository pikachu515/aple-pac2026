"""Sampled Piper geometry checks; not a continuous/self-collision certificate."""
from __future__ import annotations

import math
import numpy as np

from aple.execution.trajectory import JointTrajectoryPlanner
from aple.execution.gripper_geometry import attachment_geometry


class PiperSegmentValidator:
    """Check joint limits, TCP workspace and movable geometry against floor/AABBs.

    Collision spheres come from Piper's baked geometry. Static base geometry is
    excluded, matching robotd. Active finger adapters add configured flange-frame
    spheres covering all openings. Sampling cannot certify continuous clearance.
    """

    def __init__(self, cfg, *, fk, link_transforms, geometry):
        self.fk = fk
        self._links = link_transforms
        self._geometry = geometry
        attachment = attachment_geometry(cfg)
        self._attachment_spheres = attachment.collision_spheres if attachment else ()
        if attachment and (not isinstance(geometry.tip_index, int) or not 0 <= geometry.tip_index < len(geometry.names)):
            raise ValueError('invalid flange index for mounted gripper geometry')
        safety = cfg['safety']
        self._limits = np.asarray(JointTrajectoryPlanner._validate_limits(safety['joint_limits_rad']))
        self._workspace = self._bounds(safety['workspace_aabb_m'])
        settings = safety['segment_validation']
        self._step = JointTrajectoryPlanner._positive_finite(settings['sample_step_rad'], 'sample_step_rad')
        self._floor = float(settings['floor_z_m'])
        self._margin = float(settings['clearance_m'])
        if not math.isfinite(self._floor) or not math.isfinite(self._margin) or self._margin < 0:
            raise ValueError('floor_z_m and nonnegative clearance_m must be finite')
        self._obstacles = [self._bounds(b) for b in settings['obstacle_aabbs_m']]
        if not geometry.by_link or not math.isfinite(geometry.radius) or geometry.radius <= 0:
            raise ValueError('Piper collision geometry is missing or invalid')
        for index, points in geometry.by_link:
            if index < 0 or index >= len(geometry.names) or np.asarray(points).shape[1:] != (3,) or not np.isfinite(points).all():
                raise ValueError('invalid Piper collision points')

    @staticmethod
    def _bounds(value):
        bounds = np.asarray(value, dtype=float)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or (bounds[:, 0] >= bounds[:, 1]).any():
            raise ValueError('AABB must contain three finite increasing bounds')
        return bounds

    def is_segment_safe(self, start_rad, end_rad):
        start = np.asarray(JointTrajectoryPlanner._validate_joints(start_rad, 'start_rad'))
        end = np.asarray(JointTrajectoryPlanner._validate_joints(end_rad, 'end_rad'))
        if (start < self._limits[:, 0]).any() or (start > self._limits[:, 1]).any() or (end < self._limits[:, 0]).any() or (end > self._limits[:, 1]).any():
            return False
        count = max(1, math.ceil(float(np.max(np.abs(end - start))) / self._step))
        joints = start + np.linspace(0, 1, count + 1)[:, None] * (end - start)
        if (joints < self._limits[:, 0]).any() or (joints > self._limits[:, 1]).any():
            return False
        for q in joints:
            tcp = np.asarray(self.fk(q).position_m)
            if tcp.shape != (3,) or not np.isfinite(tcp).all():
                return False
            if (tcp < self._workspace[:, 0]).any() or (tcp > self._workspace[:, 1]).any():
                return False
        transforms = np.asarray(self._links(joints), dtype=float)
        if transforms.shape != (len(joints), len(self._geometry.names), 4, 4) or not np.isfinite(transforms).all():
            raise RuntimeError('invalid Piper link transforms for collision checking')
        radius = self._geometry.radius + self._margin
        for index, points in self._geometry.by_link:
            world = points @ transforms[:, index, :3, :3].transpose(0, 2, 1) + transforms[:, index, None, :3, 3]
            if not self._spheres_safe(world, radius):
                return False
        for center, tool_radius in self._attachment_spheres:
            flange = transforms[:, self._geometry.tip_index]
            world = np.asarray(center) @ flange[:, :3, :3].transpose(0, 2, 1) + flange[:, :3, 3]
            if not self._spheres_safe(world, tool_radius + self._margin):
                return False
        return True

    def _spheres_safe(self, world, radius):
        if (world[..., 2] - radius < self._floor).any():
            return False
        for box in self._obstacles:
            distance = np.maximum(np.maximum(box[:, 0] - world, world - box[:, 1]), 0)
            if (np.sum(distance ** 2, axis=-1) <= radius ** 2).any():
                return False
        return True


def build_piper_segment_validator(cfg, fk, *, kinematics_module=None):
    from importlib import import_module
    module = kinematics_module or import_module('piper_robot.kinematics')
    geometry = module.geometry()
    return PiperSegmentValidator(cfg, fk=fk,
                                 link_transforms=lambda q: module.link_transforms(q, geometry),
                                 geometry=geometry)
