"""Optional offline-only route and computation improvements, not hardware tuning."""
import copy
import math
from pathlib import Path

import numpy as np
import yaml

from aple.execution.goal_builder import Pose3D
from aple.execution.trajectory import ToolPose
from aple.hal.simulation_timing import stage_timer


class SimulationIKRetry:
    """Retry numerical IK from nearby seeds; never changes physical joint state.

    The trajectory planner still checks limits and every connecting segment.
    No fallback is used when the original seed already produces a solution.
    """
    def __init__(self, solver, cfg, delta_rad):
        if cfg['piper_web']['enabled'] or 'in_hand_scene' not in cfg.get('simulation', {}):
            raise ValueError('IK retries require a disabled-hardware in-hand simulation')
        if isinstance(delta_rad, bool) or not math.isfinite(delta_rad) or not 0 < delta_rad <= .3:
            raise ValueError('IK seed perturbation must be within (0, 0.3] rad')
        self.solver, self.delta = solver, delta_rad
        self.limits = np.asarray(cfg['safety']['joint_limits_rad'], float)
        self.retries = []
        self.failures = []

    def solve(self, pose, seed_rad):
        solution = self.solver.solve(pose, seed_rad)
        if solution is not None:
            return solution
        for delta in (-self.delta, self.delta):
            seed = np.array(seed_rad, float)
            seed[0] = np.clip(seed[0]+delta, *self.limits[0])
            solution = self.solver.solve(pose, tuple(seed))
            if solution is not None and np.all(np.asarray(solution) >= self.limits[:,0]) and np.all(np.asarray(solution) <= self.limits[:,1]):
                self.retries.append(dict(seed_delta_rad=delta, target_position_m=list(pose.position_m)))
                return solution
        self.failures.append(dict(target_position_m=list(pose.position_m),
                                  target_orientation_xyzw=list(pose.orientation_xyzw),
                                  seed_rad=list(seed_rad)))
        return None


def optimization_config(cfg, path):
    if 'in_hand_scene' not in cfg.get('simulation', {}) or cfg['piper_web']['enabled']:
        raise ValueError('optimization requires a disabled-hardware in-hand simulation config')
    settings = yaml.safe_load(Path(path).read_text())['optimization']
    if 'joint_place_approach' in settings and not isinstance(settings['joint_place_approach'], bool):
        raise ValueError('joint_place_approach must be boolean')
    if 'ik_seed_retry_rad' in settings:
        value = settings['ik_seed_retry_rad']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 < value <= .3:
            raise ValueError('IK seed perturbation must be within (0, 0.3] rad')
    for key in ('choose_transfer_route', 'native_slip_fk', 'exact_final_descent'):
        if not isinstance(settings[key], bool):
            raise ValueError(f'{key} must be boolean')
    for key in ('inspection_physics_settle_s', 'empty_joint_velocity_rad_s', 'landing_speed_scale'):
        value = settings[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'invalid {key}')
    if settings['landing_speed_scale'] > 1:
        raise ValueError('landing_speed_scale must be <= 1')
    mode = settings.get('inspection_rotation_mode', 'tool_roll')
    if mode not in ('tool_roll', 'vertical_pitch'):
        raise ValueError('unknown inspection_rotation_mode')
    if mode == 'vertical_pitch':
        angles = settings.get('vertical_angles_rad', [])
        if len(angles) != 4 or not np.isfinite(angles).all():
            raise ValueError('vertical inspection requires four finite angles')
        radius = settings.get('opaque_cup_radius_m')
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or not math.isfinite(radius) or radius <= 0:
            raise ValueError('vertical inspection requires a positive opaque_cup_radius_m')
    out = copy.deepcopy(cfg)
    scene = out['simulation']['in_hand_scene']
    for key in ('loaded_joint_velocity_rad_s','rotation_joint_velocity_rad_s'):
        if key in settings:
            value=settings[key]
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError(f'invalid {key}')
            scene[key]=value
    if scene['rotation_joint_velocity_rad_s']>scene['loaded_joint_velocity_rad_s']:
        raise ValueError('rotation speed must not exceed loaded speed')
    if settings['empty_joint_velocity_rad_s'] <= scene['loaded_joint_velocity_rad_s']:
        raise ValueError('empty speed must exceed loaded speed')
    scene['optimization'] = settings
    if 'preopen_m' in settings:
        width=settings['preopen_m']
        limits=out['piper_web']['gripper_width_range_m']
        largest=max(2*a['radius_m'] for a in out['simulation']['apples'])
        if isinstance(width,bool) or not isinstance(width,(int,float)) or not math.isfinite(width) or not max(limits[0],largest)<width<limits[1]:
            raise ValueError('preopen_m must clear all apples and stay inside the virtual range')
        scene['preopen_m']=width
        for size in out['size_bins']['bins']:
            size['open_width_mm']=width*1000
    if 'studio_poll_interval_s' in settings:
        poll=settings['studio_poll_interval_s']
        if isinstance(poll,bool) or not isinstance(poll,(int,float)) or not math.isfinite(poll) or poll<.01:
            raise ValueError('studio_poll_interval_s must be finite and at least 0.01 s')
        out['piper_web']['poll_interval_s']=poll
    if mode == 'vertical_pitch':
        scene['rotation_angles_rad'] = list(settings['vertical_angles_rad'])
        out['inspection']['views']['rotation_angles_rad'] = list(settings['vertical_angles_rad'])
    scene['empty_joint_velocity_rad_s'] = settings['empty_joint_velocity_rad_s']
    out['execution']['soft_landing']['slow_speed_scale'] = settings['landing_speed_scale']
    return out


def planned_points(adapter, poses):
    """Use the existing IK, limits and segment validator without moving the robot."""
    current = adapter._transport.joint_state()
    start_pose = adapter._transport.tool_pose()
    result = []
    for pose in poses:
        count = max(1, math.ceil(math.dist(start_pose.position_m, pose.position_m)/adapter._cartesian_step_m))
        a, b = np.array(start_pose.orientation_xyzw), np.array(pose.orientation_xyzw)
        if np.dot(a, b) < 0:
            b = -b
        for index in range(1, count+1):
            t = index/count
            orientation = (1-t)*a+t*b
            orientation /= np.linalg.norm(orientation)
            position = tuple((1-t)*x+t*y for x,y in zip(start_pose.position_m, pose.position_m))
            trajectory = adapter._planner.plan((ToolPose(position, tuple(orientation)),), current)
            result.extend(trajectory.points_rad)
            current = trajectory.points_rad[-1]
        start_pose = pose
    return result


def native_route_clear(transport, points):
    """Contact check on a disposable state; never alters physical scene/apple poses."""
    probe = transport.mj.MjData(transport.model)
    probe.qpos[:] = transport.data.qpos
    excluded = transport.ids[transport.active] if transport.holding else None
    # Check robot against bins and unheld fruit. The existing segment validator
    # separately checks joint limits, workspace, floor and configured obstacles.
    external = set(transport.ids) | {
        transport.mj.mj_name2id(transport.model, transport.mj.mjtObj.mjOBJ_BODY, name)
        for name in ('aple_upper', 'aple_middle', 'aple_reject')}
    external.discard(excluded)
    robot_root = transport.mj.mj_name2id(transport.model, transport.mj.mjtObj.mjOBJ_BODY, 'piper_base')
    if robot_root < 0 or any(body < 0 for body in external):
        raise ValueError('Unsupported native robot/bin body names')
    robot = set()
    for body in range(1, transport.model.nbody):
        ancestor = body
        while ancestor and ancestor != robot_root:
            ancestor = int(transport.model.body_parentid[ancestor])
        if ancestor == robot_root:
            robot.add(body)
    for joints in points:
        for name, value in zip(transport.arm, joints):
            probe.qpos[transport.joints.qadr[name]] = value
        transport.mj.mj_forward(transport.model, probe)
        for contact in probe.contact:
            a = int(transport.model.geom_bodyid[contact.geom1])
            b = int(transport.model.geom_bodyid[contact.geom2])
            if contact.dist < 0 and ((a in robot and b in external) or (b in robot and a in external)):
                return False
    return True


def execute_shorter_transfer(adapter, candidates, *, speed_scale, timeout_s):
    transport = adapter._transport
    choices, rejected = [], []
    speed = (transport.settings['loaded_joint_velocity_rad_s'] if transport.holding
             else transport.settings['empty_joint_velocity_rad_s'])*speed_scale
    deadline = adapter._deadline(timeout_s)
    with stage_timer(adapter, 'route_planning'):
        for name, poses in candidates:
            try:
                points = planned_points(adapter, poses)
                if not native_route_clear(transport, points):
                    raise ValueError('robot intersects a bin or unheld apple')
                current = np.array(transport.joint_state())
                cost = 0.
                for point in points:
                    cost += max(.02, float(np.max(abs(np.array(point)-current)))/speed)
                    current = np.array(point)
                choices.append((cost, name, points))
            except (ValueError, RuntimeError) as exc:
                rejected.append(dict(route=name, reason=str(exc)))
    if not choices:
        raise RuntimeError(f'No valid transfer candidate: {rejected}')
    cost, name, points = min(choices, key=lambda item:item[0])
    transport.route_choices.append(dict(route=name, estimated_motion_s=cost,
        candidate_costs_s={name:cost for cost,name,_ in choices}, rejected=rejected,
        holding=transport.holding, virtual_time_s=float(transport.data.time)))
    for point in points:
        if not transport.move_joint_goal(point, speed_scale=speed_scale, timeout_s=adapter._remaining(deadline)):
            raise TimeoutError('Prechecked transfer joint goal did not arrive')
