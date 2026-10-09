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
        self._tray = cfg.get('workcell', {}).get('tray', {})
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
        self._scene_obstacles = []
        self._scene_spheres = []
        self._payload = None
        self._wall_thickness = settings.get('wall_thickness_m')
        self._support_tolerance = settings.get('support_contact_tolerance_m', 0.0)
        names = list(geometry.names)
        self._self_pairs = [(names.index(a), names.index(b))
                            for a, b in settings.get('self_collision_pairs', []) or []]
        self._payload_self_links = [names.index(name)
                                   for name in settings.get('payload_self_links', []) or []]
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
        worlds = {}
        for index, points in self._geometry.by_link:
            world = points @ transforms[:, index, :3, :3].transpose(0, 2, 1) + transforms[:, index, None, :3, 3]
            worlds[index] = world
            if not self._spheres_safe(world, radius):
                return False
        for center, tool_radius in self._attachment_spheres:
            flange = transforms[:, self._geometry.tip_index]
            world = np.asarray(center) @ flange[:, :3, :3].transpose(0, 2, 1) + flange[:, :3, 3]
            if not self._spheres_safe(world, tool_radius + self._margin):
                return False
            for index in self._payload_self_links:
                if index in worlds and (np.linalg.norm(world[:, None, :] - worlds[index], axis=-1)
                                        <= tool_radius + self._geometry.radius + self._margin).any():
                    return False
        for a, b in self._self_pairs:
            if a not in worlds or b not in worlds:
                raise ValueError('self collision pair has no baked collision geometry')
            delta = worlds[a][:, :, None, :] - worlds[b][:, None, :, :]
            if (np.linalg.norm(delta, axis=-1) <= 2 * self._geometry.radius + self._margin).any():
                return False
        if self._payload is not None:
            center, payload_radius, allow_support = self._payload
            poses = [self.fk(q) for q in joints]
            world = np.array([np.asarray(p.position_m) + self._rotation(p) @ center for p in poses])
            if not self._spheres_safe(world, payload_radius + self._margin,
                                     floor_radius=payload_radius if allow_support else None,
                                     floor_tolerance=self._support_tolerance if allow_support else 0.0):
                return False
            for index in self._payload_self_links:
                if index in worlds and (np.linalg.norm(world[:, None, :] - worlds[index], axis=-1)
                                        <= payload_radius + self._geometry.radius + self._margin).any():
                    return False
        return True

    @staticmethod
    def _rotation(pose):
        from aple.inspection.in_hand_provider import pose_to_matrix
        return pose_to_matrix(pose)[:3, :3]

    def set_payload(self, apple_in_tcp, radius_m, *, allow_support_contact=False):
        center = np.asarray(apple_in_tcp.position_m, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all() or not math.isfinite(radius_m) or radius_m <= 0:
            raise ValueError('payload must have a finite TCP center and positive radius')
        if allow_support_contact and (not isinstance(self._support_tolerance, (int, float))
                or not math.isfinite(self._support_tolerance) or self._support_tolerance < 0):
            raise ValueError('measured support_contact_tolerance_m is required before lifting')
        self._payload = (center, float(radius_m), allow_support_contact)

    def clear_payload(self):
        self._payload = None

    def update_scene(self, apples, boxes, *, target_apple_id):
        """Use observed apples and axis-aligned bin walls/occupied cells as obstacles."""
        wall = self._wall_thickness
        if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
            raise ValueError('measured segment_validation.wall_thickness_m is required')
        spheres, obstacles = [], []
        for apple in apples:
            if apple.apple_id != target_apple_id:
                c = np.asarray(apple.center_m, dtype=float)
                r = float(apple.diameter_m) / 2
                if not np.isfinite(c).all() or not math.isfinite(r) or r <= 0:
                    raise ValueError('invalid observed apple obstacle')
                spheres.append((c, r))
        for box in boxes:
            if not box.valid:
                raise ValueError('cannot execute with invalid box occupancy')
            x, y, z = box.origin_m
            xmax, ymax = x + box.cols * box.cell_pitch_m, y + box.rows * box.cell_pitch_m
            if box.wall_height_m > z:
                obstacles += [self._bounds(b) for b in (
                    [[x-wall, x], [y-wall, ymax+wall], [z, box.wall_height_m]],
                    [[xmax, xmax+wall], [y-wall, ymax+wall], [z, box.wall_height_m]],
                    [[x, xmax], [y-wall, y], [z, box.wall_height_m]],
                    [[x, xmax], [ymax, ymax+wall], [z, box.wall_height_m]])]
            for i, occupied in enumerate(box.occupied):
                if occupied:
                    row, col = divmod(i, box.cols)
                    px, py = x + col * box.cell_pitch_m, y + row * box.cell_pitch_m
                    obstacles.append(self._bounds([[px, px+box.cell_pitch_m],
                        [py, py+box.cell_pitch_m], [z, max(z+wall, box.heights_m[i])]]))
        tray = self._tray
        if tray.get('center_m') is not None:
            tx, ty, tz = tray['center_m']
            sx, sy = tray['size_m']
            height = float(tray['wall_height_m'])
            if not all(math.isfinite(v) for v in (tx, ty, tz, sx, sy, height)) or min(sx, sy) <= 0 or height < 0:
                raise ValueError('invalid measured tray geometry')
            if height > 0:
                x, y, xmax, ymax = tx-sx/2, ty-sy/2, tx+sx/2, ty+sy/2
                obstacles += [self._bounds(b) for b in (
                    [[x-wall, x], [y-wall, ymax+wall], [tz, tz+height]],
                    [[xmax, xmax+wall], [y-wall, ymax+wall], [tz, tz+height]],
                    [[x, xmax], [y-wall, y], [tz, tz+height]],
                    [[x, xmax], [ymax, ymax+wall], [tz, tz+height]])]
        self._scene_spheres, self._scene_obstacles = spheres, obstacles

    def _spheres_safe(self, world, radius, *, floor_radius=None, floor_tolerance=0.0):
        if (world[..., 2] - (radius if floor_radius is None else floor_radius) < self._floor - floor_tolerance - 1e-9).any():
            return False
        for box in (*self._obstacles, *self._scene_obstacles):
            distance = np.maximum(np.maximum(box[:, 0] - world, world - box[:, 1]), 0)
            if (np.sum(distance ** 2, axis=-1) <= radius ** 2).any():
                return False
        for center, other_radius in self._scene_spheres:
            if (np.linalg.norm(world-center, axis=-1) <= radius+other_radius).any():
                return False
        return True


def build_piper_segment_validator(cfg, fk, *, kinematics_module=None):
    from importlib import import_module
    module = kinematics_module or import_module('piper_robot.kinematics')
    geometry = module.geometry()
    return PiperSegmentValidator(cfg, fk=fk,
                                 link_transforms=lambda q: module.link_transforms(q, geometry),
                                 geometry=geometry)
