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
            # 핑거가 닿는 거리에 사과가 없으면(접근 전 미리 벌리기) 무엇도 잡을 수 없다. 사전 개방 폭(직경 + 여유)이
            # 멀리 있는 다른 사과 직경과 비슷하면 잡기 시도로 오인해 실패하던 문제 (실물 장면 시뮬에서 확인).
            in_reach = any(np.linalg.norm(np.asarray(a['center_m']) - tcp.position_m) <= a['radius_m'] + 2 * self.tolerance
                           for a in self.apples)
            if not candidates and (not size_matches or not in_reach):
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


def virtual_camera_from_table(snap, tray_center_xy, *, step_px=4, iters=300, tol_m=0.004, center_frac=0.4):
    """실물 깊이의 테이블 평면 → 가상 셀의 T_base_camera (카메라–로봇 보정의 대용, 실측값 아님).

    base z 는 테이블 법선(카메라 쪽), base x 는 카메라 x 축을 테이블에 투영한 방향이다.
    광축이 테이블과 만나는 점을 가상 트레이 중심에 둔다 — 카메라 높이·기울기는 실물 그대로다.
    평면은 영상 가운데 center_frac 만 쓴다 — 10/7 D435 는 깊이가 화면 가장자리로 갈수록 휘어
    (재보정 전 카메라) 전체로 맞추면 평면이 크게 기운다.
    """
    from aple.perception.geometry import depth_points, fit_plane_ransac
    valid = snap.depth_m > 0
    H, W = valid.shape
    mh, mw = int(H * (1 - center_frac) / 2), int(W * (1 - center_frac) / 2)
    center = np.zeros_like(valid)
    center[mh:H - mh, mw:W - mw] = True
    valid &= center
    pts = depth_points(snap.depth_m, snap.intrinsics)[::step_px, ::step_px][valid[::step_px, ::step_px]]
    if len(pts) < 100:
        raise ValueError('not enough valid depth to find the table plane')
    n, d = fit_plane_ransac(pts, iters=iters, tol_m=tol_m, seed=snap.scene_id)
    if abs(n[2]) < 0.5:
        raise ValueError('camera must look down at the table (tilt above 60 degrees)')
    bz = np.asarray(n, dtype=float)
    bx = np.array([1.0, 0.0, 0.0]) - bz[0] * bz
    bx /= np.linalg.norm(bx)
    by = np.cross(bz, bx)
    R_bc = np.stack([bx, by, bz])                  # 행 = base 축(카메라 좌표) → 카메라→base 회전
    axis_hit = np.array([0.0, 0.0, -d / n[2]])     # 광축과 테이블의 교점 (카메라 좌표)
    T = np.eye(4)
    T[:3, :3] = R_bc
    T[:3, 3] = -R_bc @ axis_hit + np.array([tray_center_xy[0], tray_center_xy[1], 0.0])
    return validate_T(T)


class SnapshotWorld(KinematicWorld):
    """실물 RGB-D 장면 한 장을 카메라로 쓰는 가상 셀. 사과는 그 장면의 M1 인식 결과로 놓는다.

    판정·계획은 진짜 사진으로 하고, 집기·놓기는 Piper IK·구간 검사·이상적 파지로만 확인한다.
    사과를 집어도 사진은 바뀌지 않으므로 한 장면(1 사이클)만 지원한다.
    """

    def __init__(self, cfg, fk, validator, snapshot):
        super().__init__(cfg, fk, validator)
        self.snapshot = snapshot

    def virtual_boxes(self, snap):
        """사진에는 상자가 없으므로(그 자리는 바닥·의자다) 설정의 상자를 빈 가상 상자로 돌려준다."""
        from aple.schema import BoxOccupancy, Grade
        out = []
        for name, box in self.cfg['workcell']['boxes'].items():
            rows, cols = box['grid']
            z0 = float(box['origin_m'][2])
            out.append(BoxOccupancy(snap.scene_id, Grade[name], rows, cols, float(box['cell_pitch_m']),
                                    float(box['wall_height_m']), float(box['wall_height_m']),
                                    tuple(float(v) for v in box['origin_m']), [z0] * (rows * cols),
                                    [False] * (rows * cols), True))
        return out

    def capture(self):
        if self.scene_id >= 1:
            raise RuntimeError('real-scene simulation supports one scene (the photo does not change after picking)')
        self.scene_id += 1
        s = self.snapshot
        return SceneSnapshot(self.scene_id, s.source, s.capture_mode, s.color.copy(), s.depth_m.copy(),
                             s.intrinsics, self.elapsed_s)


def seed_from_snapshot(cfg, snapshot):
    """실물 장면으로 가상 셀을 맞춘다 (cfg 를 고친다): T_base_camera ← 테이블 평면, 사과 ← M1 인식."""
    from aple.perception import detector, pose_estimator
    tray = cfg['workcell']['tray']['center_m']
    T = virtual_camera_from_table(snapshot, tray[:2])
    cfg['workcell']['T_base_camera'] = T.tolist()
    first = SceneSnapshot(1, snapshot.source, snapshot.capture_mode, snapshot.color, snapshot.depth_m,
                          snapshot.intrinsics, 0.0)
    # 첫 탐색은 A4 종이 + 둘레 사과가 들어가는 넓이로 (합성 셀의 작은 트레이로는 가장자리 사과가 빠진다)
    size = cfg['workcell']['tray']['size_m']
    cfg['workcell']['tray']['size_m'] = [max(float(size[0]), 0.32), max(float(size[1]), 0.40)]
    states = pose_estimator.estimate(first, detector.detect(first, cfg), T, None, cfg)
    if not states:
        raise ValueError('no apples found in the real scene inside the virtual tray')
    _fit_tray_to_apples(cfg, states)
    states = pose_estimator.estimate(first, detector.detect(first, cfg), T, None, cfg)
    cfg['simulation']['apples'] = [dict(center_m=[float(v) for v in s.center_m], radius_m=s.diameter_m / 2,
                                        red_fraction=0.0) for s in states]
    return T, states


def _fit_tray_to_apples(cfg, states):
    """가상 트레이를 실물 사과 배치에 맞춘다. 실물(종이 위)에는 벽이 없는데, 합성 셀의 작은 트레이를 그대로 쓰면
    핑거가 가상 벽에 걸려 모든 사과가 DEFER 된다. 사과 범위 + 핑거 도달 거리 + 최소 간격만큼 키운다."""
    from aple.perception.grasp_planner import grasp_params
    p = grasp_params(cfg)
    finger = (p['open_margin_m'] + p['finger_thickness_m'] / 2
              + math.hypot(p['finger_thickness_m'] / 2, p['finger_width_m'] / 2) + p['min_clearance_m'] + 0.005)
    xy = np.array([s.center_m[:2] for s in states])
    r = max(s.diameter_m / 2 for s in states)
    lo, hi = xy.min(axis=0) - r - finger, xy.max(axis=0) + r + finger
    tray = cfg['workcell']['tray']
    tray['center_m'] = [float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(tray['center_m'][2])]
    tray['size_m'] = [float(hi[0] - lo[0]), float(hi[1] - lo[1])]


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


def build_simulation(cfg, *, kinematics_module=None, snapshot=None):
    """snapshot 을 주면 실물 장면 모드 — cfg 의 T_base_camera·사과를 그 장면에 맞춘다 (seed_from_snapshot)."""
    if cfg['execution']['default_backend'] != 'scripted' or cfg['inspection']['policy'] != 'NEVER':
        raise ValueError('kinematic simulation supports scripted mode A only')
    if snapshot is not None:
        seed_from_snapshot(cfg, snapshot)
    from aple.execution.trajectory import build_piper_kinematics
    from aple.execution.segment_validator import build_piper_segment_validator
    from aple.hal.piper_motion import PiperTrajectoryMotionAdapter
    ik, fk = build_piper_kinematics(cfg, kinematics_module=kinematics_module)
    validator = build_piper_segment_validator(cfg, fk, kinematics_module=kinematics_module)
    checks = check_piper_roundtrip(ik, fk, cfg)
    world = KinematicWorld(cfg, fk, validator) if snapshot is None else SnapshotWorld(cfg, fk, validator, snapshot)
    return world, PiperTrajectoryMotionAdapter(world, cfg, ik, validator), checks
