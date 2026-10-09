"""Simulation-only central inspection with contact physics and synthetic skin RGB-D.

No device/network access. Shared execution/inspection modules are reused unchanged.
"""
from __future__ import annotations

import copy
import math
import secrets
from pathlib import Path

import numpy as np
import yaml

from aple.execution.goal_builder import Pose3D, PlaceTargets, compose, inverse, relative_pose
from aple.execution.inspect_skill import InspectionPlan, _axis_angle, _multiply
from aple.hal.gripper_physics import build_assumed_model, physics_scene, closing_gap
from aple.hal.piper_motion import PiperTrajectoryMotionAdapter
from aple.inspection.in_hand_capture import InHandCapture
from aple.inspection.in_hand_provider import pose_to_matrix
from aple.schema import SceneSnapshot, Source, CaptureMode
from aple.hal.simulation_timing import timed_stage, stage_timer


def scene_config(cfg, path, *, seed=None):
    settings = yaml.safe_load(Path(path).read_text())['in_hand_scene']
    if 'require_grasp_clearance' in settings and not isinstance(settings['require_grasp_clearance'], bool):
        raise ValueError('require_grasp_clearance must be boolean')
    if settings.get('box_grids'):
        cfg = copy.deepcopy(cfg)
        for name, grid in settings['box_grids'].items():
            if name not in cfg['workcell']['boxes'] or not isinstance(grid, list) or len(grid) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in grid):
                raise ValueError('box_grids requires known boxes and two positive integer dimensions')
            cfg['workcell']['boxes'][name]['grid'] = list(grid)
    seed = secrets.randbits(32) if seed is None else seed
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError('seed must be a nonnegative integer')
    bounds = np.asarray(settings['map_xy_m'], float)
    center = np.asarray(settings['inspection_center_m'], float)
    if bounds.shape != (2, 2) or not np.isfinite(bounds).all() or (np.diff(bounds) <= 0).any():
        raise ValueError('map_xy_m must be increasing finite bounds')
    if center.shape != (3,) or not np.isfinite(center).all() or not np.allclose(center[:2], bounds.mean(axis=1)):
        raise ValueError('inspection XY must equal the map midpoint')
    for key in ('object_min_distance_m', 'clearance_m', 'inspection_clear_radius_m', 'preopen_m',
                'empty_joint_velocity_rad_s', 'loaded_joint_velocity_rad_s',
                'rotation_joint_velocity_rad_s', 'max_slip_m', 'transfer_height_m', 'settle_time_s',
                'transfer_arc_radius_m', 'transfer_arc_step_rad', 'box_cell_pitch_m', 'grip_open_delta_m', 'camera_height_m', 'video_fps',
                'far_place_radius_m', 'far_place_tilt_rad'):
        if isinstance(settings[key], bool) or not math.isfinite(settings[key]) or settings[key] <= 0:
            raise ValueError(f'invalid {key}')
    attempts = settings['max_layout_attempts']
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
        raise ValueError('max_layout_attempts must be a positive integer')
    radii = np.asarray(settings['reach_radius_m'], float)
    if radii.shape != (2,) or not np.isfinite(radii).all() or not 0 < radii[0] < radii[1]:
        raise ValueError('reach_radius_m must be increasing positive radii')
    angles = np.asarray(settings['rotation_angles_rad'],float)
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all() or not math.isfinite(settings['tilt_rad']):
        raise ValueError('inspection angles must be finite')
    if not settings['empty_joint_velocity_rad_s'] > settings['loaded_joint_velocity_rad_s'] >= settings['rotation_joint_velocity_rad_s']:
        raise ValueError('speeds must satisfy empty > loaded >= rotation')
    specs = settings['apple_specs']
    for spec in specs:
        if not 0 < spec['diameter_m'] < settings['preopen_m'] or not 0 < spec['mass_kg'] or not 0 <= spec['red_fraction'] <= 1:
            raise ValueError('invalid apple size/mass/color')
    if not cfg['piper_web']['gripper_width_range_m'][0] <= settings['preopen_m'] <= cfg['piper_web']['gripper_width_range_m'][1]:
        raise ValueError('preopen exceeds assumed finger stroke')
    rng = np.random.default_rng(seed)
    boxes = copy.deepcopy(list(cfg['workcell']['boxes'].values()))
    for box in boxes:
        box['cell_pitch_m'] = settings['box_cell_pitch_m']
    # All six objects sample the SAME area; no fixed apple/box zones.
    halves = [np.array([s['diameter_m']/2]*2) for s in specs]
    halves += [np.array(b['grid'][::-1])*b['cell_pitch_m']/2 + .002 for b in boxes]
    order = list(range(len(halves)))
    if settings.get('require_grasp_clearance', False):
        order.sort(key=lambda i: float(np.prod(halves[i])), reverse=True)
    sample_halves = [halves[i] for i in order]
    for attempt in range(settings['max_layout_attempts']):
        points = []
        for i, half in enumerate(sample_halves):
            for _ in range(200):
                p = rng.uniform(bounds[:, 0]+half, bounds[:, 1]-half)
                if not settings['reach_radius_m'][0] <= np.linalg.norm(p) <= settings['reach_radius_m'][1]:
                    continue
                if np.linalg.norm(np.maximum(abs(p-center[:2])-half, 0)) < settings['inspection_clear_radius_m']:
                    continue
                if any(np.linalg.norm(p-q) < settings['object_min_distance_m'] or
                       (abs(p-q) < half+sample_halves[j]+settings['clearance_m']).all()
                       for j, q in enumerate(points)):
                    continue
                points.append(p)
                break
            else:
                break
        if len(points) == len(halves):
            points = [points[order.index(i)] for i in range(len(order))]
            if settings.get('require_grasp_clearance', False):
                from types import SimpleNamespace
                probe_cfg = copy.deepcopy(cfg)
                probe_cfg['safety']['workspace_aabb_m'] = settings['workspace_aabb_m']
                probe_cfg['simulation']['in_hand_scene'] = settings
                for box, p in zip(probe_cfg['workcell']['boxes'].values(), points[len(specs):]):
                    box['cell_pitch_m'] = settings['box_cell_pitch_m']
                    half = np.array(box['grid'][::-1])*box['cell_pitch_m']/2
                    box['origin_m'] = [*map(float, p-half), 0.]
                apples = [SimpleNamespace(scene_id=1, apple_id=i+1,
                    center_m=(*p, s['diameter_m']/2), diameter_m=s['diameter_m'])
                    for i, (p, s) in enumerate(zip(points, specs))]
                if not all(g.collision_free for g in radial_grasps(apples, probe_cfg)):
                    continue
            break
    else:
        raise ValueError(f'Cannot place separated objects (seed={seed})')
    out = copy.deepcopy(cfg)
    out['simulation']['in_hand_scene'] = dict(settings, seed=seed, layout_attempts=attempt+1)
    out['simulation']['apples'] = [dict(center_m=[*map(float, p), s['diameter_m']/2],
        radius_m=s['diameter_m']/2, mass_kg=s['mass_kg'], red_fraction=s['red_fraction']) for p, s in zip(points, specs)]
    for box, p in zip(out['workcell']['boxes'].values(), points[len(specs):]):
        box['cell_pitch_m'] = settings['box_cell_pitch_m']
        half = np.array(box['grid'][::-1])*box['cell_pitch_m']/2
        box['origin_m'] = [*map(float, p-half), 0.]
    out['workcell']['tray'] = dict(center_m=[*map(float, bounds.mean(axis=1)), 0.], size_m=list(map(float, bounds[:, 1]-bounds[:, 0])), wall_height_m=0.)
    out['safety']['workspace_aabb_m'] = copy.deepcopy(settings['workspace_aabb_m'])
    out['safety']['speed_scale_max'] = 1.
    out['inspection']['policy'] = 'ALWAYS'
    out['inspection']['views'] = {k: settings[k] for k in ('tilt_rad', 'rotation_angles_rad', 'settle_time_s')}
    out['simulation']['color_pattern_frame'] = 'camera_tangent'
    out['simulation']['intrinsics'] = [400, 400, 320, 240]
    if 'camera_fovy_deg' in settings:
        fovy = settings['camera_fovy_deg']
        if isinstance(fovy,bool) or not math.isfinite(fovy) or not 0 < fovy < 180:
            raise ValueError('camera_fovy_deg must be finite and between 0 and 180')
        height, width = out['simulation']['image_size']
        focal = height/(2*math.tan(math.radians(fovy)/2))
        out['simulation']['intrinsics'] = [focal,focal,width/2,height/2]
    # Reachable, empty inspection-ready pose; the old low scan pose obstructs
    # a straight transfer when larger fruit are spread around the map.
    out['simulation']['initial_joint_rad'] = list(settings['initial_joint_rad'])
    out['workcell']['T_base_camera'] = [[1,0,0,float(center[0])], [0,-1,0,float(center[1])], [0,0,-1,settings['camera_height_m']], [0,0,0,1]]
    for b in out['size_bins']['bins']:
        b['open_width_mm'] = settings['preopen_m']*1000
    out['grasp_planner']['open_margin_m'] = .014
    return out


class ContactTransport:
    """Measured MuJoCo joint state/FK; held fruits move only through contacts."""
    def __init__(self, cfg, assumptions, fk, *, sample_callback=None, capture_callback=None):
        import mujoco
        from piper_sim.scene import JointMap, ARM_JOINTS, FINGERS
        self.mj, self.cfg, self.a, self.fk = mujoco, cfg, assumptions, fk
        self.settings = cfg['simulation']['in_hand_scene']
        self.optimization = self.settings.get('optimization', {})
        self.model = build_assumed_model(physics_scene(cfg, assumptions), assumptions)
        self.data = mujoco.MjData(self.model)
        self.joints, self.arm, self.fingers = JointMap(self.model), ARM_JOINTS, FINGERS
        for name, q in zip(self.arm, cfg['simulation']['initial_joint_rad']):
            self.data.qpos[self.joints.qadr[name]] = self.data.ctrl[self.joints.act[name]] = q
        for name in self.fingers:
            self.data.qpos[self.joints.qadr[name]] = self.data.ctrl[self.joints.act[name]] = assumptions['finger_stroke_m']
        mujoco.mj_forward(self.model, self.data)
        self.ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f'aple_apple_{i+1}') for i in range(len(cfg['simulation']['apples']))]
        self.initial = self.positions()
        self.peak = self.initial[:, 2].copy()
        self.active = None
        self.apple_in_tcp = None
        self.holding = False
        self.rotating = False
        self.scene_id = 10
        self.events, self.views, self.slips = [], [], []
        self.motion_errors = []
        self.route_choices = []
        self.landing_segments = []
        self.native_tip = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'link6')
        if self.optimization.get('native_slip_fk'):
            if self.native_tip < 0:
                raise ValueError('Native FK requires the official link6 frame')
            actual = pose_to_matrix(self.tool_pose())
            native = self.native_tcp_matrix()
            if not np.allclose(native, actual, atol=1e-5, rtol=0):
                raise ValueError('MuJoCo and official FK definitions disagree')
        self.sample_callback = sample_callback
        self.capture_callback = capture_callback
        self.steps = 0
        self.frame_stride = max(1, round(1/(self.settings['video_fps']*self.model.opt.timestep)))
        self.step(.3)

    def positions(self):
        return np.array([self.data.xpos[i].copy() for i in self.ids])

    def native_tcp_matrix(self):
        rotation = self.data.xmat[self.native_tip].reshape(3,3)
        transform = np.eye(4)
        transform[:3,:3] = rotation
        transform[:3,3] = self.data.xpos[self.native_tip]+rotation@np.array(self.cfg['workcell']['tcp_offset_m'])
        return transform

    def step(self, seconds):
        for _ in range(max(1, math.ceil(seconds/self.model.opt.timestep))):
            self.mj.mj_step(self.model, self.data)
            if self.optimization.get('native_slip_fk'):
                # Refresh positions to the integrated qpos, then reuse native FK.
                self.mj.mj_kinematics(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                raise RuntimeError('non-finite physics state')
            self.peak = np.maximum(self.peak, self.positions()[:, 2])
            self.steps += 1
            if self.sample_callback and self.steps % self.frame_stride == 0:
                self.sample_callback(self.model, self.data)
            if self.holding and self.active is not None and self.apple_in_tcp is not None:
                if self.optimization.get('native_slip_fk'):
                    native = self.native_tcp_matrix()
                    expected = native[:3,3]+native[:3,:3]@np.array(self.apple_in_tcp.position_m)
                else:
                    expected = np.array(compose(self.tool_pose(), self.apple_in_tcp).position_m)
                error = float(np.linalg.norm(self.positions()[self.active]-expected))
                if error > self.settings['max_slip_m']:
                    failure = dict(apple=self.active+1, time_s=float(self.data.time), displacement_m=error, rotating=self.rotating)
                    self.slips.append(failure)
                    raise RuntimeError(f'Apple {self.active+1} slipped {error:.4f} m; stopping simulation')

    def joint_state(self):
        return tuple(float(self.data.qpos[self.joints.qadr[n]]) for n in self.arm)

    def tool_pose(self):
        return self.fk(self.joint_state())

    def gripper_width_m(self):
        return self.a['closed_contact_gap_m'] + sum(float(self.data.qpos[self.joints.qadr[n]]) for n in self.fingers)

    def begin(self, apple, apple_in_tcp):
        # M1 IDs follow image detections, not scene-file ordering. The closing
        # command selected the physical body; keep that identity across views.
        if self.active is None:
            raise RuntimeError('No physical apple selected at grasp')
        self.apple_in_tcp = apple_in_tcp
        self.holding = True

    def confirm_holding(self, apple, apple_in_tcp, *, timeout_s):
        if self.active is None or not self.holding:
            return False
        expected = compose(self.tool_pose(), apple_in_tcp).position_m
        return bool(np.linalg.norm(self.positions()[self.active]-expected)
                    <= self.settings['max_slip_m'])

    @timed_stage("joint_motion")
    def move_joint_goal(self, position_rad, *, speed_scale, timeout_s):
        start = np.array(self.joint_state())
        target = np.array(position_rad)
        speed = self.settings['rotation_joint_velocity_rad_s'] if self.rotating else self.settings['loaded_joint_velocity_rad_s'] if self.holding else self.settings['empty_joint_velocity_rad_s']
        duration = max(.02, float(np.max(abs(target-start)))/(speed*speed_scale))
        if duration > timeout_s:
            return False
        n = math.ceil(duration/self.model.opt.timestep)
        for i in range(1, n+1):
            for name, value in zip(self.arm, start+(target-start)*i/n):
                self.data.ctrl[self.joints.act[name]] = value
            self.step(self.model.opt.timestep)
        for _ in range(math.ceil(self.a['max_goal_settle_s']/self.model.opt.timestep)):
            if np.max(abs(np.array(self.joint_state())-target)) <= self.a['joint_arrival_tolerance_rad']:
                self.events.append(dict(action='move', joints_rad=list(position_rad), speed_scale=speed_scale,
                    joint_velocity_rad_s=speed*speed_scale, holding=self.holding, rotating=self.rotating, virtual_time_s=float(self.data.time)))
                return True
            self.step(self.model.opt.timestep)
        self.events.append(dict(action='arrival_failed', goal_rad=target.tolist(),
                                actual_rad=list(self.joint_state()), tcp_m=list(self.tool_pose().position_m),
                                apple_positions=self.positions().tolist(), width_m=self.gripper_width_m(),
                                contacts=[dict(geom1=self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_GEOM, c.geom1),
                                               geom2=self.mj.mj_id2name(self.model, self.mj.mjtObj.mjOBJ_GEOM, c.geom2),
                                               distance_m=float(c.dist), point_m=c.pos.tolist()) for c in self.data.contact]))
        return False

    @timed_stage("gripper")
    def set_gripper_width(self, width_m, effort, *, timeout_s):
        grasp = width_m < self.settings['preopen_m']-self.settings['grip_open_delta_m'] and not self.holding
        if grasp:
            self.active = int(np.argmin(np.linalg.norm(self.positions()-np.array(self.tool_pose().position_m), axis=1)))
        if self.holding and width_m > self.gripper_width_m()+self.settings['grip_open_delta_m']:
            self.events.append(dict(action='release', apple_index=self.active, center_m=self.positions()[self.active].tolist()))
            self.holding = False
            self.apple_in_tcp = None
        gap = closing_gap(width_m, grasp=grasp, assumptions=self.a)
        half = (gap-self.a['closed_contact_gap_m'])/2
        if not 0 <= half <= self.a['finger_stroke_m']:
            raise ValueError('width outside assumed stroke')
        for name in self.fingers:
            self.data.ctrl[self.joints.act[name]] = half
        self.step(self.a['gripper_settle_s'])
        if grasp:
            self.apple_in_tcp = relative_pose(self.tool_pose(), Pose3D(tuple(self.positions()[self.active]), (0,0,0,1)))
            self.holding = True
            self.events.append(dict(action='grasp', apple_index=self.active))
        self.events.append(dict(action='gripper', width_m=width_m, effort=effort, virtual_time_s=float(self.data.time)))
        return True

    @timed_stage("color_capture")
    def capture_in_hand(self, apple_id, view_index, *, timeout_s):
        """Analytic sphere RGB-D using ACTUAL physical pose; ray tests exclude fingers.

        Skin pattern is fixed in the apple body frame, never repainted per view.
        """
        settle = self.optimization.get('inspection_physics_settle_s', 0.)
        if settle:
            with stage_timer(self, 'inspection_physics_settle'):
                self.step(settle)
        index = self.active
        body = self.ids[index]
        c = self.positions()[index]
        radius = self.cfg['simulation']['apples'][index]['radius_m']
        T = np.array(self.cfg['workcell']['T_base_camera'], float)
        h, w = self.cfg['simulation']['image_size']
        intr = tuple(self.cfg['simulation']['intrinsics'])
        v, u = np.indices((h,w))
        rays = np.stack(((u-intr[2])/intr[0], (v-intr[3])/intr[1], np.ones_like(u)), axis=-1)
        dirs = rays @ T[:3,:3].T
        origin = T[:3,3]
        delta = origin-c
        aa = (dirs*dirs).sum(-1)
        bb = dirs@delta
        disc = bb**2-aa*(delta@delta-radius**2)
        z = (-bb-np.sqrt(np.maximum(disc,0)))/aa
        mask = (disc>=0)&(z>0)
        # Visible sphere samples only; reject rays blocked by robot/fingers/boxes.
        gid = np.empty(1, np.int32)
        for y,x in zip(*np.nonzero(mask)):
            direction = dirs[y,x]/np.linalg.norm(dirs[y,x])
            distance = self.mj.mj_ray(self.model, self.data, origin, direction, None, 1, -1, gid)
            if distance < 0 or self.model.geom_bodyid[gid[0]] != body:
                mask[y,x] = False
        normals = (origin+z[...,None]*dirs-c)/radius
        rotation = np.array(self.data.xmat[body]).reshape(3,3)
        normal_body = normals@rotation
        red = normal_body[...,1] > 1-2*self.cfg['simulation']['apples'][index]['red_fraction']
        color = np.full((h,w,3),150,np.uint8)
        color[mask&red]=(38,30,175)
        color[mask&~red]=(75,190,205)
        depth = np.where(mask,z,0).astype(np.float32)
        self.scene_id += 1
        snapshot = SceneSnapshot(self.scene_id, Source.IN_HAND, CaptureMode.FULL, color, depth,
            intr, float(self.data.time), apple_id=apple_id, view_index=view_index, tool_pose=pose_to_matrix(self.tool_pose()))
        R = T[:3,:3].T@rotation
        self.views.append(dict(apple=index+1, perception_apple_id=apple_id, view_index=view_index, center_m=c.tolist(),
                               visible_pixels=int(mask.sum()), time_s=float(self.data.time)))
        self.events.append(dict(action='inspect_capture', apple_index=index, view_index=view_index,
                                virtual_time_s=float(self.data.time)))
        if self.capture_callback:
            self.capture_callback(index, view_index, color, depth)
        return InHandCapture(snapshot, mask, R)


class CentralInspectionAdapter(PiperTrajectoryMotionAdapter):
    """Simulation-only routing through the center, then existing view executor."""
    @timed_stage("pose_motion")
    def _move_poses(self, poses, **kwargs):
        try:
            return super()._move_poses(poses, **kwargs)
        except Exception as exc:
            self._transport.motion_errors.append(dict(error=str(exc),
                current_joints_rad=list(self._transport.joint_state()),
                target_poses=[dict(position_m=list(p.position_m), orientation_xyzw=list(p.orientation_xyzw)) for p in poses]))
            raise

    def _high_transfer(self, target_xy, *, speed_scale, timeout_s):
        """Follow a radial arc, avoiding the near-base horizontal wrist singularity."""
        current = self._transport.tool_pose()
        settings = self._transport.settings
        height = settings['transfer_height_m']
        tilt = settings['tilt_rad']
        start_xy = np.array(current.position_m[:2])
        end_xy = np.asarray(target_xy)
        a0 = math.atan2(start_xy[1], start_xy[0])
        a1 = math.atan2(end_xy[1], end_xy[0])
        radius = settings['transfer_arc_radius_m']
        poses = [Pose3D((*map(float, start_xy), height), radial_orientation(start_xy, tilt))]
        count = max(1, math.ceil(abs(a1-a0)/settings['transfer_arc_step_rad']))
        for angle in np.linspace(a0,a1,count+1):
            xy = [radius*math.cos(angle), radius*math.sin(angle)]
            poses.append(Pose3D((*xy,height), radial_orientation(xy,tilt)))
        poses.append(Pose3D((*map(float,end_xy),height),radial_orientation(end_xy,tilt)))
        if self._transport.optimization.get('choose_transfer_route'):
            from aple.hal.simulation_optimization import execute_shorter_transfer
            direct = (poses[0], poses[-1])
            execute_shorter_transfer(self, (('direct', direct), ('radial_arc', tuple(poses))),
                speed_scale=speed_scale, timeout_s=timeout_s)
            return
        self._move_poses(tuple(poses), speed_scale=speed_scale, deadline=self._deadline(timeout_s))

    def execute_pick(self, apple, profile, candidate, targets, **kwargs):
        self._transport.set_gripper_width(profile.open_width_m, kwargs['effort'], timeout_s=kwargs['timeout_s'])
        with stage_timer(self, 'empty_transfer'):
            self._high_transfer(targets.pregrasp_pose.position_m[:2], speed_scale=kwargs['speed_scale'], timeout_s=kwargs['timeout_s'])
        return super().execute_pick(apple, profile, candidate, targets, **kwargs)

    def inspect_views(self, profile, inspection_plan, **kwargs):
        transport = self._transport
        settings = transport.settings
        # Move at the transport plane, then center the actual held-apple target.
        horizontal = _axis_angle((0,1,0), settings['tilt_rad'])
        target_apple = Pose3D(tuple(settings['inspection_center_m']), _multiply(horizontal, transport.apple_in_tcp.orientation_xyzw))
        central_tcp = compose(target_apple, inverse(transport.apple_in_tcp))
        with stage_timer(self, 'center_transfer'):
            self._high_transfer(central_tcp.position_m[:2], speed_scale=self._cfg['safety']['speed_scale_max'], timeout_s=kwargs['timeout_s'])
            self._move_poses((central_tcp,), speed_scale=self._cfg['safety']['speed_scale_max'], deadline=self._deadline(kwargs['timeout_s']))
        poses = inspection_poses(settings, transport.apple_in_tcp)
        # Offline synthetic capture has no camera pipeline to wait for. Advance
        # real physics in capture_in_hand instead of sleeping in real time.
        settle = 0. if transport.optimization else settings['settle_time_s']
        plan = InspectionPlan(poses, settle)
        transport.rotating = True
        try:
            with stage_timer(self, 'rotation_inspection'):
                return super().inspect_views(profile, plan, **kwargs)
        finally:
            transport.rotating = False

    @timed_stage("box_transfer")
    def execute_place_phase_a(self, apple, profile, placement, targets, **kwargs):
        transport = self._transport
        current = transport.tool_pose()
        aligned = Pose3D(current.position_m, _axis_angle((0,1,0), transport.settings['tilt_rad']))
        self._move_poses((aligned,), speed_scale=kwargs['speed_scale'], deadline=self._deadline(kwargs['timeout_s']))
        self._high_transfer(targets.hover_pose.position_m[:2], speed_scale=kwargs['speed_scale'], timeout_s=kwargs['timeout_s'])
        tilt = math.radians(transport.a['tool_down_rpy_deg'][1])
        if np.linalg.norm(placement.place_center_m[:2]) > transport.settings['far_place_radius_m']:
            tilt = transport.settings['far_place_tilt_rad']
        down = radial_orientation(placement.place_center_m, tilt)
        apple_pose = Pose3D(placement.place_center_m, _multiply(down, targets.apple_in_tcp.orientation_xyzw))
        place = compose(apple_pose, inverse(targets.apple_in_tcp))
        hover = Pose3D((place.position_m[0], place.position_m[1], place.position_m[2]+placement.hover_z_m-placement.release_z_m), down)
        self._sim_place_targets = PlaceTargets(place, hover, targets.apple_in_tcp)
        if transport.optimization.get('joint_place_approach', False):
            # A straight Cartesian rotation can cross unreachable wrist poses.
            # Plan the full joint segment first, including existing safety checks.
            from aple.hal.simulation_optimization import native_route_clear
            from aple.execution.trajectory import ToolPose, TrajectoryPlanningError
            from aple.schema import FailureCode
            trajectory = self._planner.plan((ToolPose(hover.position_m, hover.orientation_xyzw),), transport.joint_state())
            if not native_route_clear(transport, trajectory.points_rad):
                raise TrajectoryPlanningError(FailureCode.COLLISION, 'joint place approach contacts scene obstacles')
            deadline = self._deadline(kwargs['timeout_s'])
            for joints in trajectory.points_rad:
                if not transport.move_joint_goal(joints, speed_scale=kwargs['speed_scale'], timeout_s=self._remaining(deadline)):
                    raise TimeoutError('joint place approach did not arrive')
        return super().execute_place_phase_a(apple, profile, placement, self._sim_place_targets, **kwargs)

    @timed_stage("landing_release_retreat")
    def execute_phase_b(self, apple, profile, plan, targets, landing_plan, **kwargs):
        from aple.execution.soft_landing import build_soft_landing_plan
        landing = build_soft_landing_plan(self._sim_place_targets,
                    final_descent_m=kwargs['final_descent_m'], slow_speed_scale=kwargs['slow_speed_scale'],
                    approach_speed_scale=landing_plan.steps[0].speed_scale)
        self._transport.landing_segments.append(dict(apple=apple.apple_id,
            slow_distance_m=landing.slow_descent_start_z_m-self._sim_place_targets.place_pose.position_m[2],
            slow_speed_scale=kwargs['slow_speed_scale']))
        return super().execute_phase_b(apple, profile, plan, self._sim_place_targets, landing, **kwargs)


def radial_orientation(position_m, tilt_rad):
    return _multiply(_axis_angle((0,0,1), math.atan2(position_m[1], position_m[0])),
                     _axis_angle((0,1,0), tilt_rad))


def inspection_poses(settings, apple_in_tcp):
    """Use the shared inspection geometry; vertical pitch is an experiment option."""
    from aple.execution.inspect_skill import build_inspection_plan
    mode = settings.get('optimization', {}).get('inspection_rotation_mode', 'tool_roll')
    axis = (0,1,0) if mode == 'vertical_pitch' else (0,0,1)
    return build_inspection_plan(apple_in_tcp, tuple(settings['inspection_center_m']),
        tilt_rad=settings['tilt_rad'], rotation_angles_rad=settings['rotation_angles_rad'],
        settle_time_s=settings.get('settle_time_s', 0.0), rotation_axis=axis).target_tool_poses


def radial_grasps(apples, cfg):
    """Simulation-only radial wrist posture, with neighbor/bin/workspace clearance."""
    from aple.perception.grasp_planner import GraspCandidate
    candidates = []
    for apple in apples:
        yaw = math.atan2(apple.center_m[1], apple.center_m[0])
        params = cfg['grasp_planner']
        width = cfg['simulation']['in_hand_scene']['preopen_m']
        footprint = math.hypot(params['finger_thickness_m']/2, params['finger_width_m']/2)
        reach = width/2 + params['finger_thickness_m']/2
        axis = np.array([-math.sin(yaw), math.cos(yaw)])
        clearances = []
        workspace = np.asarray(cfg['safety']['workspace_aabb_m'])[:2]
        for sign in (-1,1):
            finger = np.array(apple.center_m[:2])+sign*reach*axis
            clearances.extend(finger-workspace[:,0]-footprint)
            clearances.extend(workspace[:,1]-finger-footprint)
            for other in apples:
                if other.apple_id != apple.apple_id:
                    clearances.append(np.linalg.norm(finger-np.array(other.center_m[:2]))-other.diameter_m/2-footprint)
            for box in cfg['workcell']['boxes'].values():
                half = np.array(box['grid'][::-1])*box['cell_pitch_m']/2+.002
                center = np.array(box['origin_m'][:2])+half-.002
                clearances.append(np.linalg.norm(np.maximum(abs(finger-center)-half,0))-footprint)
        clearance = float(min(clearances))
        pose = Pose3D(apple.center_m, radial_orientation(apple.center_m, math.radians(params['tool_down_rpy_deg'][1])))
        candidates.append(GraspCandidate(apple.scene_id, apple.apple_id, pose, yaw, (0.,0.,1.),
            params['approach_height_m'], max(0.,clearance), clearance>=params['min_clearance_m'], max(0.,clearance)))
    return candidates
