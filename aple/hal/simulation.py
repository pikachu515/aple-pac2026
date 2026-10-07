"""Device-free kinematic world using injected Piper math and synthetic RGB-D.

No CAN, camera, HTTP or robotd access. Ideal grasp, instantaneous settling and
fixed-effort assumptions do not predict fruit damage or real grasp success.
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import yaml

from aple.execution.goal_builder import Pose3D, compose, relative_pose
from aple.execution.trajectory import JointTrajectoryPlanner, ToolPose
from aple.schema import SceneSnapshot, Source, CaptureMode
from aple.hal.calibration import validate_T


def simulation_config(cfg, path):
    cfg = copy.deepcopy(cfg)
    overlay = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(overlay, dict) or overlay.get('simulation', {}).get('model') != 'piper':
        raise ValueError('simulation overlay must declare simulation.model: piper')

    def merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], value)
            else:
                dst[key] = copy.deepcopy(value)
    merge(cfg, overlay)
    cfg['piper_web']['enabled'] = False
    if cfg['execution']['default_backend'] != 'scripted' or cfg['inspection']['policy'] != 'NEVER':
        raise ValueError('kinematic simulation supports scripted mode A only')
    return cfg


class KinematicWorld:
    def __init__(self, cfg, fk, validator):
        self.cfg, self.fk, self.validator = cfg, fk, validator
        self.settings = cfg['simulation']
        self.q = JointTrajectoryPlanner._validate_joints(self.settings['initial_joint_rad'], 'initial_joint_rad')
        self.velocity = JointTrajectoryPlanner._positive_finite(self.settings['joint_velocity_rad_s'], 'joint_velocity_rad_s')
        self.tolerance = JointTrajectoryPlanner._positive_finite(self.settings['grasp_tolerance_m'], 'grasp_tolerance_m')
        self.camera_transform = validate_T(cfg['workcell']['T_base_camera'])
        image_size = self.settings['image_size']
        intr = np.asarray(self.settings['intrinsics'], dtype=float)
        if (len(image_size) != 2 or any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in image_size)
                or intr.shape != (4,) or not np.isfinite(intr).all() or (intr[:2] <= 0).any()):
            raise ValueError('invalid simulation image size or intrinsics')
        self.table = float(self.settings['table_z_m'])
        if not math.isfinite(self.table):
            raise ValueError('simulation.table_z_m must be finite')
        if not validator.is_segment_safe(self.q, self.q):
            raise ValueError('simulation initial pose is unsafe')
        self.apples = copy.deepcopy(self.settings['apples'])
        for apple in self.apples:
            center = np.asarray(apple['center_m'], dtype=float)
            if center.shape != (3,) or not np.isfinite(center).all():
                raise ValueError('simulation apple center must contain three finite meters')
            JointTrajectoryPlanner._positive_finite(apple['radius_m'], 'radius_m')
            if not 0 <= float(apple['red_fraction']) <= 1:
                raise ValueError('red_fraction must be in [0, 1]')
        self.width = float(cfg['piper_web']['gripper_width_range_m'][1])
        self.held = None
        self.apple_in_tcp = None
        self.scene_id = 0
        self.elapsed_s = 0.0
        self.events = []

    def joint_state(self):
        return self.q

    def tool_pose(self):
        return self.fk(self.q)

    def gripper_width_m(self):
        return self.width

    def move_joint_goal(self, position_rad, *, speed_scale, timeout_s):
        goal = JointTrajectoryPlanner._validate_joints(position_rad, 'position_rad')
        if not math.isfinite(speed_scale) or not 0 < speed_scale <= self.cfg['safety']['speed_scale_max']:
            raise ValueError('invalid simulation speed scale')
        duration = max(abs(b - a) for a, b in zip(self.q, goal)) / (self.velocity * speed_scale)
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError('timeout_s must be finite and positive')
        if duration > timeout_s or not self.validator.is_segment_safe(self.q, goal):
            return False
        self.q = goal
        self.elapsed_s += duration
        if self.held is not None:
            pose = compose(self.tool_pose(), self.apple_in_tcp)
            self.apples[self.held]['center_m'] = list(pose.position_m)
        self.events.append(dict(action='move', joints_rad=list(goal), speed_scale=speed_scale,
                                virtual_time_s=self.elapsed_s, tcp_m=list(self.tool_pose().position_m)))
        return True

    def set_gripper_width(self, width_m, effort, *, timeout_s):
        lower, upper = self.cfg['piper_web']['gripper_width_range_m']
        if not math.isfinite(width_m) or not lower <= width_m <= upper:
            raise ValueError('gripper width outside simulation range')
        if not 0 <= effort <= self.cfg['safety']['gripper_effort_max'] or timeout_s <= 0:
            raise ValueError('invalid gripper command')
        opening_delta = float(self.cfg['execution']['gripper']['opening_margin_m']) / 2
        if self.held is not None and width_m > self.width + opening_delta:
            apple = self.apples[self.held]
            drop = apple['center_m'][2] - apple['radius_m'] - self.table
            self.events.append(dict(action='release', apple_index=self.held,
                                    center_m=list(apple['center_m']), drop_m=drop))
            apple['center_m'][2] = self.table + apple['radius_m']
            self.held = self.apple_in_tcp = None
        elif self.held is None and width_m < self.width - opening_delta:
            tcp = self.tool_pose()
            candidates = [i for i, a in enumerate(self.apples)
                          if np.linalg.norm(np.asarray(a['center_m']) - tcp.position_m) <= self.tolerance
                          and abs(2 * a['radius_m'] - width_m) <= self.tolerance]
            size_matches = any(abs(2*a['radius_m'] - width_m) <= self.tolerance for a in self.apples)
            if not candidates and not size_matches:
                # Repositioning empty fingers to a pre-open width is a valid
                # command, even if narrower than the maximum opening.
                self.width = width_m
                self.events.append(dict(action='gripper', width_m=width_m, effort=effort,
                                        virtual_time_s=self.elapsed_s))
                return True
            if len(candidates) != 1:
                return False
            self.held = candidates[0]
            self.apple_in_tcp = relative_pose(tcp, Pose3D(tuple(self.apples[self.held]['center_m']), (0, 0, 0, 1)))
            self.events.append(dict(action='grasp', apple_index=self.held))
        self.width = width_m
        self.events.append(dict(action='gripper', width_m=width_m, effort=effort,
                                virtual_time_s=self.elapsed_s))
        return True

    def capture_in_hand(self, *args, **kwargs):
        raise RuntimeError('mode B RGB-D simulation is not implemented; use inspection.policy NEVER')

    def capture(self):
        """Raycast spheres and a horizontal table into aligned BGR/depth meters."""
        self.scene_id += 1
        height, width = self.settings['image_size']
        fx, fy, cx, cy = self.settings['intrinsics']
        v, u = np.indices((height, width))
        rays = np.stack(((u - cx) / fx, (v - cy) / fy, np.ones_like(u)), axis=-1)
        T = self.camera_transform
        directions = rays @ T[:3, :3].T
        origin = T[:3, 3]
        if (np.abs(directions[..., 2]) < 1e-9).any():
            raise ValueError('virtual table must intersect every camera ray')
        depth = (self.table - origin[2]) / directions[..., 2]
        if (depth <= 0).any():
            raise ValueError('virtual table must be in front of the camera')
        color = np.full((height, width, 3), 150, dtype=np.uint8)
        for i, apple in enumerate(self.apples):
            if i == self.held:
                continue
            center = np.asarray(apple['center_m'])
            delta = origin - center
            aa = np.sum(directions ** 2, axis=-1)
            bb = directions @ delta
            discriminant = bb ** 2 - aa * (delta @ delta - apple['radius_m'] ** 2)
            hit_depth = (-bb - np.sqrt(np.maximum(discriminant, 0))) / aa
            mask = (discriminant >= 0) & (hit_depth > 0) & (hit_depth < depth)
            normal = (origin + hit_depth[..., None] * directions - center) / apple['radius_m']
            red = normal[..., 1] > 1 - 2 * apple['red_fraction']
            color[mask & red] = (38, 30, 175)
            color[mask & ~red] = (75, 190, 205)
            depth[mask] = hit_depth[mask]
        return SceneSnapshot(self.scene_id, Source.TOP, CaptureMode.FULL, color,
                             depth.astype(np.float32), tuple(self.settings['intrinsics']), self.elapsed_s)


def check_piper_roundtrip(ik, fk, cfg):
    """Perturb seed for genuine numerical IK work; reject position/rotation drift."""
    seed = np.asarray(cfg['simulation']['initial_joint_rad'], dtype=float)
    limits = np.asarray(cfg['safety']['joint_limits_rad'])
    checks = []
    for delta in (0.0, -0.03, 0.03):
        joints = seed.copy()
        joints[0] += delta
        target = fk(joints)
        solution = ik.solve(ToolPose(target.position_m, target.orientation_xyzw), tuple(seed + 0.01))
        if solution is None:
            raise RuntimeError('Piper FK/IK roundtrip failed to converge')
        if (np.asarray(solution) < limits[:, 0]).any() or (np.asarray(solution) > limits[:, 1]).any():
            raise RuntimeError('Piper roundtrip solution outside configured joint limits')
        actual = fk(solution)
        distance = float(np.linalg.norm(np.asarray(actual.position_m) - target.position_m))
        angle = 2 * math.acos(min(1.0, abs(float(np.dot(actual.orientation_xyzw, target.orientation_xyzw)))))
        if distance > 0.0015 or angle > math.radians(0.3):
            raise RuntimeError(f'Piper roundtrip residual too large: {distance} m, {angle} rad')
        checks.append(dict(position_error_m=distance, rotation_error_rad=angle))
    return checks


def build_simulation(cfg, *, kinematics_module=None):
    if cfg['execution']['default_backend'] != 'scripted' or cfg['inspection']['policy'] != 'NEVER':
        raise ValueError('kinematic simulation supports scripted mode A only')
    from aple.execution.trajectory import build_piper_kinematics
    from aple.execution.segment_validator import build_piper_segment_validator
    from aple.hal.piper_motion import PiperTrajectoryMotionAdapter
    ik, fk = build_piper_kinematics(cfg, kinematics_module=kinematics_module)
    validator = build_piper_segment_validator(cfg, fk, kinematics_module=kinematics_module)
    checks = check_piper_roundtrip(ik, fk, cfg)
    world = KinematicWorld(cfg, fk, validator)
    return world, PiperTrajectoryMotionAdapter(world, cfg, ik, validator), checks
