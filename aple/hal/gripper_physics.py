"""Assumed finger adapter model for Studio MuJoCo, with physical apple motion.

All added motion follows MuJoCo contacts. No welds, teleporting apples or ideal
attachment rules. This model is an assumption, not a reconstruction of the CAD.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import yaml


def optical_ring_dimensions(assumptions):
    """Measured diameters are config-only mm; model geometry uses m."""
    outer = assumptions.get('opaque_cup_outer_diameter_mm')
    inner = assumptions.get('opaque_cup_inner_diameter_mm')
    if outer is None and inner is None:
        return None
    for value in (outer, inner):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError('ring diameters must be positive finite numbers')
    if inner >= outer:
        raise ValueError('ring inner diameter must be smaller than outer diameter')
    return outer / 2000., inner / 2000.


def add_optical_ring(spec, body, name, pos, outer, inner, curve=None, sign=1):
    """Separate convex sectors preserve a real aperture in MuJoCo ray tests.

    A single convex mesh would fill the hole. Thickness remains an assumption.
    """
    import mujoco
    sectors = 96
    inner_vertex = inner / math.cos(math.pi / sectors)
    faces = [0,2,1,0,3,2,4,5,6,4,6,7]
    for edge in range(4):
        nxt = (edge+1) % 4
        faces.extend([edge,nxt,nxt+4,edge,nxt+4,edge+4])
    for index in range(sectors):
        angles = np.array([index,index+1]) * 2*math.pi/sectors
        xy = [[r*math.cos(a),r*math.sin(a)] for r,a in
              [(inner_vertex,angles[0]),(outer,angles[0]),(outer,angles[1]),(inner_vertex,angles[1])]]
        vertices = [[x,y,z+(sign*(curve-math.sqrt(curve**2-x*x-y*y)) if curve else 0.)]
                    for z in (-.0015,.0015) for x,y in xy]
        mesh = f'aple_ring_{name}_{index}'
        spec.add_mesh(name=mesh,uservert=np.array(vertices).reshape(-1).tolist(),userface=faces)
        body.add_geom(name=mesh,type=mujoco.mjtGeom.mjGEOM_MESH,meshname=mesh,
            pos=pos,quat=[math.sqrt(.5),math.sqrt(.5),0,0],
            rgba=[1,1,1,1],mass=0,contype=0,conaffinity=0)


def load_assumptions(path):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))['assumed_gripper']
    optical_ring_dimensions(cfg)
    if not isinstance(cfg.get('concave_ring',False),bool):
        raise ValueError('concave_ring must be boolean')
    ring=optical_ring_dimensions(cfg)
    if ring and cfg.get('concave_ring') and cfg['cup_radius_m']<=ring[0]:
        raise ValueError('cup curvature radius must exceed ring outer radius')
    clearance=cfg.get('ring_pad_clearance_m',.0005)
    if isinstance(clearance,bool) or not isinstance(clearance,(int,float)) or not math.isfinite(clearance) or clearance<0:
        raise ValueError('ring_pad_clearance_m must be finite and nonnegative')
    opaque = cfg.get('opaque_cup_radius_m')
    if opaque is not None and (isinstance(opaque, bool) or not isinstance(opaque, (int, float)) or not math.isfinite(opaque) or opaque <= 0):
        raise ValueError('opaque_cup_radius_m must be positive and finite')
    if 'top_camera_fovy_deg' in cfg:
        fovy = cfg['top_camera_fovy_deg']
        if isinstance(fovy,bool) or not isinstance(fovy,(int,float)) or not math.isfinite(fovy) or not 0 < fovy < 180:
            raise ValueError('invalid top_camera_fovy_deg')
    if 'top_camera_align_xy' in cfg and not isinstance(cfg['top_camera_align_xy'],bool):
        raise ValueError('top_camera_align_xy must be boolean')
    for key in ('closed_contact_gap_m', 'finger_stroke_m', 'cup_radius_m', 'pad_radius_m',
                'pad_mass_kg', 'shaft_radius_m', 'shaft_mass_kg', 'finger_servo_kp_n_m',
                'finger_force_limit_n', 'closing_compression_m', 'gripper_settle_s',
                'final_settle_s', 'apple_mass_kg', 'joint_arrival_tolerance_rad',
                'max_goal_settle_s', 'lift_threshold_m', 'destination_xy_tolerance_m',
                'lift_height_m', 'hover_margin_m', 'cartesian_step_m', 'opening_margin_m',
                'finger_envelope_thickness_m', 'finger_envelope_width_m'):
        value = cfg[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'assumed_gripper.{key} must be positive and finite')
    for key, length in [('flange_tcp_m', 3), ('pad_friction', 3), ('apple_friction', 3), ('pad_solref', 2), ('tool_down_rpy_deg', 3), ('tray_size_m', 2)]:
        value = cfg[key]
        if len(value) != length or not np.isfinite(value).all():
            raise ValueError(f'invalid assumed_gripper.{key}')
    if cfg['flange_tcp_m'][:2] != [0, 0]:
        raise ValueError('assumed model currently supports axial +Z TCP only')
    offsets = np.asarray(cfg['cup_grid_offsets_m'], dtype=float)
    if offsets.ndim != 1 or not len(offsets) or not np.isfinite(offsets).all():
        raise ValueError('invalid cup grid offsets')
    if math.sqrt(2) * max(abs(offsets)) >= cfg['cup_radius_m']:
        raise ValueError('cup grid must fit inside assumed spherical curvature')
    if abs(cfg['finger_stroke_m'] - .034) > 1e-9:
        raise ValueError('must match Studio stock finger stroke of 0.034 m')
    return cfg


def contact_gap_range(assumptions):
    lo = assumptions['closed_contact_gap_m']
    return lo, lo + 2 * assumptions['finger_stroke_m']


def build_assumed_model(scene, assumptions):
    import mujoco
    from piper_sim import scene_spec
    spec = scene_spec.compose(scene)
    # The central-inspection scene owns its virtual camera location. Studio's
    # scene schema otherwise permits only top-camera height, with fixed XY.
    for obj in scene.get('objects', []):
        if obj['id'] == 'aple_inspection_center':
            for camera in spec.cameras:
                if camera.name == 'top':
                    camera.pos[:2] = obj['pos'][:2]
                    if 'top_camera_fovy_deg' in assumptions:
                        camera.fovy = assumptions['top_camera_fovy_deg']
                    if assumptions.get('top_camera_align_xy'):
                        camera.alt.type = mujoco.mjtOrientation.mjORIENTATION_XYAXES
                        camera.alt.xyaxes = [1,0,0,0,1,0]
    for sign, name in ((1, 'finger_l'), (-1, 'finger_r')):
        body = spec.body(name)
        if body is None:
            raise ValueError(f'Studio model lacks {name}')
        # Official model: gripper_base z=.035, finger body z=.030,
        # finger body y=±.008. Confirm before extending it.
        base = spec.body('gripper_base')
        if not np.allclose(body.pos, [0, sign * .008, .03]) or not np.allclose(base.pos, [0, 0, .035]):
            raise ValueError('unsupported Studio finger mounting geometry')
        body.gravcomp = 1
        offset = assumptions['closed_contact_gap_m'] / 2 - .008
        tcp_z = assumptions['flange_tcp_m'][2] - .065
        radius = assumptions['pad_radius_m']
        curve = assumptions['cup_radius_m']
        ring = optical_ring_dimensions(assumptions)
        concave=assumptions.get('concave_ring',False)
        if ring is not None:
            add_optical_ring(spec,body,name,[0,sign*(offset+.0015),tcp_z],*ring,curve=curve if concave else None,sign=sign)
        elif assumptions.get('opaque_cup_radius_m'):
            # Optical silhouette only. Do not pretend a convex cylinder models
            # the real concave shell's collision mechanics or mass.
            body.add_geom(name=f'aple_opaque_{name}', type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                pos=[0, sign*(offset+.0015), tcp_z],
                quat=[math.sqrt(.5), math.sqrt(.5), 0, 0],
                size=[assumptions['opaque_cup_radius_m'], .0015, 0],
                rgba=[1,1,1,1], mass=0, contype=0, conaffinity=0)
        if ring is not None and concave:
            outer,inner=ring
            contact_radius=inner+radius+assumptions.get('ring_pad_clearance_m',.0005)
            if contact_radius+radius>=outer:
                raise ValueError('contact pads do not fit outside the ring aperture')
            sag=math.sqrt(curve**2-contact_radius**2)-curve
            pad_points=[(contact_radius*math.cos(a),contact_radius*math.sin(a),sag)
                        for a in np.linspace(0,2*math.pi,12,endpoint=False)]
            pad_mass=assumptions['pad_mass_kg']*len(assumptions['cup_grid_offsets_m'])**2/12
        else:
            pad_points=[(x,z,math.sqrt(curve**2-x**2-z**2)-curve)
                        for x in assumptions['cup_grid_offsets_m'] for z in assumptions['cup_grid_offsets_m']]
            pad_mass=assumptions['pad_mass_kg']
        for i,(x,z,sag) in enumerate(pad_points):
            body.add_geom(name=f'aple_pad_{name}_{i}', type=mujoco.mjtGeom.mjGEOM_SPHERE,
                pos=[x, sign * (offset + sag + radius), tcp_z + z], size=[radius, 0, 0],
                rgba=[1, 1, 1, 1], mass=pad_mass, condim=6,
                friction=assumptions['pad_friction'], solref=assumptions['pad_solref'])
        shaft_end=tcp_z-ring[0]-assumptions['shaft_radius_m'] if ring is not None and concave else tcp_z
        body.add_geom(name=f'aple_shaft_{name}', type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[0, sign * (offset + .012), .02, 0, sign * (offset + .012), shaft_end],
            size=[assumptions['shaft_radius_m'], 0, 0], rgba=[1, 1, 1, 1],
            mass=assumptions['shaft_mass_kg'])
    camera_position = np.asarray(assumptions['side_camera_position_m'], dtype=float)
    camera_target = np.asarray(assumptions['side_camera_target_m'], dtype=float)
    z_axis = camera_position - camera_target
    x_axis = np.cross([0., 0., 1.], z_axis)
    if (camera_position.shape != (3,) or camera_target.shape != (3,)
            or not np.isfinite(camera_position).all() or not np.isfinite(camera_target).all()
            or np.linalg.norm(x_axis) < 1e-9):
        raise ValueError('side camera must have finite, distinct position and target with a horizontal viewing direction')
    x_axis /= np.linalg.norm(x_axis)
    z_axis /= np.linalg.norm(z_axis)
    y_axis = np.cross(z_axis, x_axis)
    fovy = assumptions['side_camera_fovy_deg']
    if isinstance(fovy, bool) or not math.isfinite(fovy) or not 1 < fovy < 179:
        raise ValueError('side_camera_fovy_deg must be between 1 and 179 degrees')
    spec.worldbody.add_camera(name='side', pos=camera_position,
                             xyaxes=np.concatenate([x_axis, y_axis]), fovy=fovy)
    model = spec.compile()
    for name in ('gripper_l', 'gripper_r'):
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        model.actuator_gainprm[aid, 0] = assumptions['finger_servo_kp_n_m']
        model.actuator_biasprm[aid, 1] = -assumptions['finger_servo_kp_n_m']
        model.actuator_forcelimited[aid] = True
        model.actuator_forcerange[aid] = [-assumptions['finger_force_limit_n'], assumptions['finger_force_limit_n']]
    return model


def physics_scene(cfg, assumptions):
    from aple.hal.studio_simulation import scene_from_config
    scene = scene_from_config(cfg)
    scene['name'] = 'APLE assumed gripper physics'
    for obj in scene['objects']:
        if obj['shape'] == 'sphere':
            index = int(obj['id'].removeprefix('aple_apple_')) - 1
            obj['mass'] = cfg['simulation']['apples'][index].get('mass_kg', assumptions['apple_mass_kg'])
            obj['friction'] = list(assumptions['apple_friction'])
    inspection = cfg['simulation'].get('in_hand_scene')
    if inspection:
        scene['cameras']['top']['z'] = inspection['camera_height_m']
        scene['objects'].append(dict(id='aple_inspection_center', label='INSPECTION CENTER',
            shape='cylinder', size=[inspection['inspection_clear_radius_m'], .0005],
            pos=[*inspection['inspection_center_m'][:2], .0005], movable=False,
            rgba=[.15,.55,.95,.4]))
    return scene


def closing_gap(width_m, *, grasp, assumptions):
    if grasp:
        return max(assumptions['closed_contact_gap_m'], width_m - assumptions['closing_compression_m'])
    return width_m


def run_physics(cfg, trace, assumptions, *, sample_callback=None, with_attachment=True):
    count = len(cfg['simulation']['apples'])
    releases = [e for e in trace['events'] if e.get('action') == 'release']
    if (len(trace['apples']) != count or len(releases) != count
            or sorted(e.get('apple_index', -1) for e in releases) != list(range(count))):
        raise ValueError('physical test requires a complete planned cycle with one release per configured apple')
    import mujoco
    from piper_sim.scene import JointMap, ARM_JOINTS, FINGERS
    scene = physics_scene(cfg, assumptions)
    if with_attachment:
        model = build_assumed_model(scene, assumptions)
    else:
        from piper_sim.scene import load_model
        model = load_model(scene)
    data = mujoco.MjData(model)
    joints = JointMap(model)
    initial = cfg['simulation']['initial_joint_rad']
    for name, q in zip(ARM_JOINTS, initial):
        data.qpos[joints.qadr[name]] = q
        data.ctrl[joints.act[name]] = q
    for name in FINGERS:
        data.qpos[joints.qadr[name]] = assumptions['finger_stroke_m']
        data.ctrl[joints.act[name]] = assumptions['finger_stroke_m']
    mujoco.mj_forward(model, data)
    apple_names = [f'aple_apple_{i+1}' for i in range(len(cfg['simulation']['apples']))]
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in apple_names]
    initial_positions = np.array([data.xpos[i].copy() for i in ids])
    peak = initial_positions[:, 2].copy()
    transport_errors = []
    close_next = False
    previous_time = 0.0
    steps = 0
    def step(n):
        nonlocal steps
        for _ in range(max(1, n)):
            mujoco.mj_step(model, data)
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise RuntimeError('MuJoCo state became non-finite')
            peak[:] = np.maximum(peak, [data.xpos[i, 2] for i in ids])
            steps += 1
            if sample_callback and steps % 25 == 0:
                sample_callback(model, data)

    dt = model.opt.timestep
    step(round(.3 / dt))
    for event in trace['events']:
        action = event['action']
        if action == 'grasp':
            close_next = True
        if action == 'move':
            target = np.asarray(event['joints_rad'])
            now = float(event['virtual_time_s'])
            duration = max(.05, now - previous_time)
            previous_time = now
            start = np.array([data.ctrl[joints.act[n]] for n in ARM_JOINTS])
            count = max(1, math.ceil(duration / dt))
            for i in range(1, count + 1):
                goal = start + (target - start) * i / count
                for name, value in zip(ARM_JOINTS, goal):
                    data.ctrl[joints.act[name]] = value
                step(1)
            for _ in range(math.ceil(assumptions['max_goal_settle_s'] / dt)):
                actual = np.array([data.qpos[joints.qadr[n]] for n in ARM_JOINTS])
                if np.max(abs(actual - target)) <= assumptions['joint_arrival_tolerance_rad']:
                    break
                step(1)
            else:
                transport_errors.append({'goal_rad': target.tolist(), 'actual_rad': actual.tolist()})
                break
        elif action == 'gripper':
            width = float(event['width_m'])
            gap = closing_gap(width, grasp=close_next, assumptions=assumptions)
            half = (gap - assumptions['closed_contact_gap_m']) / 2
            if not -1e-9 <= half <= assumptions['finger_stroke_m'] + 1e-9:
                raise ValueError('requested width outside assumed gripper contact range')
            half = max(0, min(half, assumptions['finger_stroke_m']))
            for name in FINGERS:
                data.ctrl[joints.act[name]] = half
            step(round(assumptions['gripper_settle_s'] / dt))
            close_next = False
    step(round(assumptions['final_settle_s'] / dt))
    final_positions = np.array([data.xpos[i].copy() for i in ids])
    outcomes = evaluate_outcomes(cfg, trace, assumptions, initial_positions, peak, final_positions)
    return dict(mode='assumed_gripper_contact_physics' if with_attachment else 'stock_finger_negative_control',
                assumptions=assumptions, simulated_time_s=float(data.time),
                transport_errors=transport_errors, apples=outcomes,
                success=not transport_errors and all(o['success'] for o in outcomes))


def evaluate_outcomes(cfg, trace, assumptions, initial_positions, peak, final_positions):
    """Report transport into the destination bin separately from center accuracy."""
    outcomes = []
    for i, target in enumerate(trace['apples']):
        expected = np.asarray(target['center_m'])
        final = np.asarray(final_positions[i])
        xy_error = float(np.linalg.norm(final[:2] - expected[:2]))
        lifted = bool(peak[i] - initial_positions[i, 2] >= assumptions['lift_threshold_m'])
        center_accurate = bool(xy_error <= assumptions['destination_xy_tolerance_m'])
        radius = cfg['simulation']['apples'][i]['radius_m']
        box = min(cfg['workcell']['boxes'].values(), key=lambda b: math.dist(
            expected[:2], [b['origin_m'][j] + b['cell_pitch_m'] * b['grid'][1-j] / 2 for j in (0, 1)]))
        center = np.array([box['origin_m'][j] + box['cell_pitch_m'] * box['grid'][1-j] / 2 for j in (0, 1)])
        half_size = np.asarray(box['grid'][::-1]) * box['cell_pitch_m'] / 2
        # Studio bin walls are lower than an apple's center. Check the sphere's
        # actual cross-section at the wall top, not just its center point.
        wall_top = box['origin_m'][2] + box['wall_height_m']
        dz = max(0., final[2] - wall_top)
        section_radius = math.sqrt(max(0., radius**2 - dz**2))
        settled = bool(abs(final[2] - expected[2]) < .015)
        in_bin = bool(settled and (np.abs(final[:2] - center) + section_radius <= half_size).all())
        outcomes.append(dict(apple=i+1, lifted=lifted, placed_in_destination_bin=in_bin,
                             center_accurate=center_accurate, peak_height_m=float(peak[i]),
                             final_center_m=final.tolist(), destination_xy_error_m=xy_error,
                             success=lifted and in_bin))
    return outcomes


def assumed_config(cfg, assumptions):
    """Change only an in-memory virtual config; preserve measured files."""
    import copy
    out = copy.deepcopy(cfg)
    out['workcell']['tcp_offset_m'] = list(assumptions['flange_tcp_m'])
    out['piper_web']['gripper_width_range_m'] = list(contact_gap_range(assumptions))
    out['grasp_planner'].update(tool_down_rpy_deg=assumptions['tool_down_rpy_deg'],
                              approach_height_m=assumptions['lift_height_m'],
                              finger_thickness_m=assumptions['finger_envelope_thickness_m'],
                              finger_width_m=assumptions['finger_envelope_width_m'])
    out['workcell']['tray']['size_m'] = list(assumptions['tray_size_m'])
    out['workcell']['placement']['margin_m'] = assumptions['hover_margin_m']
    out['execution']['trajectory']['max_cartesian_step_m'] = assumptions['cartesian_step_m']
    out['execution']['gripper']['opening_margin_m'] = assumptions['opening_margin_m']
    return out
