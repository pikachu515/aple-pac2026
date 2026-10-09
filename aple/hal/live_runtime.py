"""Compose the real Studio pipeline without opening CAN/camera devices.

Construction performs only configuration/kinematics work. Robot commands happen
when the caller explicitly runs the returned scan/execute callbacks.
"""
from dataclasses import dataclass, replace
from pathlib import Path
import math
import numpy as np

from aple.execution.gripper_geometry import apply_attachment_geometry
from aple.execution.timed_motion import MotionLimits, positive
from aple.execution.trajectory import JointTrajectoryPlanner, build_piper_kinematics
from aple.hal.calibration import validate_T
from aple.orchestration.recovery import is_lighting_fault


def missing_for_live_execution(config):
    """Concrete missing configuration, rather than a permanent placeholder guard."""
    errors = []
    try:
        cfg = apply_attachment_geometry(config)
    except (ValueError, KeyError, TypeError) as exc:
        return [str(exc)]
    def check(label, call):
        try:
            call()
        except (ValueError, KeyError, TypeError) as exc:
            errors.append(f'{label}: {exc}')
    def required(value, label):
        if value is None:
            raise ValueError(f'{label} must be measured/configured')
    def vector(values, count):
        if not isinstance(values, (list, tuple)) or len(values) != count or not np.isfinite(values).all():
            raise ValueError(f'{count} finite measured values required')
    work, safety = cfg['workcell'], cfg['safety']
    check('workcell.T_base_camera', lambda: validate_T(work.get('T_base_camera')))
    check('workcell.tcp_offset_m', lambda: vector(work.get('tcp_offset_m'), 3))
    check('safety.joint_limits_rad', lambda: JointTrajectoryPlanner._validate_limits(safety.get('joint_limits_rad')))
    check('safety.workspace_aabb_m', lambda: bounds(safety.get('workspace_aabb_m')))
    check('execution.motion', lambda: MotionLimits.from_config(cfg['execution'].get('motion', {})))
    check('capture_joint_tolerance_rad', lambda: positive(cfg['execution'].get('motion', {}).get('capture_joint_tolerance_rad'), 'capture_joint_tolerance_rad'))
    for key in ('closing_compression_m', 'contact_settle_s', 'contact_width_tolerance_m', 'contact_stability_m'):
        check('execution.gripper.'+key, lambda key=key: positive(cfg['execution']['gripper'].get(key), key))
    if cfg['execution']['gripper'].get('control_mode') != 'fixed_effort':
        errors.append('live Studio runtime currently requires fixed_effort; calibrated force needs a measured feedback backend')
    if cfg['execution']['default_backend'] != 'scripted':
        errors.append('live runtime requires the scripted backend; ACT adapter is not connected')
    settings = safety.get('segment_validation', {})
    for key in ('sample_step_rad', 'floor_z_m', 'clearance_m', 'wall_thickness_m',
                'obstacle_aabbs_m', 'self_collision_pairs', 'payload_self_links', 'support_contact_tolerance_m'):
        check('safety.segment_validation.'+key, lambda key=key: required(settings.get(key), key))
    for key in ('floor_z_m', 'clearance_m', 'support_contact_tolerance_m'):
        check('safety.segment_validation.'+key,
              lambda key=key: finite_scalar(settings.get(key), nonnegative=key != 'floor_z_m'))
    check('safety.segment_validation.obstacle_aabbs_m', lambda: obstacle_bounds(settings.get('obstacle_aabbs_m')))
    check('safety.segment_validation.self_collision_pairs', lambda: link_names(settings.get('self_collision_pairs'), pairs=True))
    check('safety.segment_validation.payload_self_links', lambda: link_names(settings.get('payload_self_links')))
    if cfg['inspection']['views'].get('center_m') is not None:
        check('inspection.views.center_m', lambda: vector(cfg['inspection']['views']['center_m'], 3))
    for key in ('sample_step_rad', 'wall_thickness_m'):
        check('safety.segment_validation.'+key, lambda key=key: positive(settings.get(key), key))
    from aple.perception.workcell_detect import auto_enabled
    if not auto_enabled(cfg):                          # auto_detect 면 장면마다 카메라로 찾는다 (예비값은 선택)
        check('workcell.tray.center_m', lambda: vector(work['tray'].get('center_m'), 3))
        for name, box in work['boxes'].items():
            check('workcell.boxes.'+name+'.origin_m', lambda box=box: vector(box.get('origin_m'), 3))
    for name in ('home', 'scan'):
        check('workcell.poses.'+name, lambda name=name: vector(work['poses'].get(name+'_joint_rad'), 6))
    web = cfg['piper_web']
    if web.get('enabled') is not True or web.get('robotd_safety_confirmed') is not True:
        errors.append('piper_web.enabled / robotd_safety_confirmed must be explicitly configured')
    if not isinstance(web.get('iface'), str) or not web['iface'] or 'PLACEHOLDER' in web['iface'] or web['iface'].startswith('sim_'):
        errors.append('piper_web.iface must name the measured hardware follower')
    check('piper_web.gripper_width_range_m', lambda: width_range(web.get('gripper_width_range_m')))
    card = cfg['color'].get('card_json')
    if not card or not Path(card).is_file():
        errors.append('color.card_json: measured color calibration file required')
    return errors


def finite_scalar(value, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('finite measured scalar required')
    if nonnegative and value < 0:
        raise ValueError('must be nonnegative')


def obstacle_bounds(values):
    if not isinstance(values, (list, tuple)):
        raise ValueError('measured obstacle AABB list required')
    for value in values:
        bounds(value)


def link_names(values, *, pairs=False):
    if not isinstance(values, (list, tuple)):
        raise ValueError('reviewed link list required')
    for value in values:
        names = value if pairs and isinstance(value, (list, tuple)) else (value,)
        if (pairs and len(names) != 2) or any(not isinstance(n, str) or not n for n in names):
            raise ValueError('link names or two-link pairs required')
        if pairs and names[0] == names[1]:
            raise ValueError('self-collision pair must use distinct links')


def bounds(value):
    b = np.asarray(value, dtype=float)
    if b.shape != (3, 2) or not np.isfinite(b).all() or (b[:, 0] >= b[:, 1]).any():
        raise ValueError('three increasing finite workspace bounds required')


def width_range(value):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('measured closed/open contact widths required')
    a, b = map(float, value)
    if not math.isfinite(a) or not math.isfinite(b) or not 0 <= a < b:
        raise ValueError('invalid measured contact widths')


@dataclass
class LiveRuntime:
    studio: object
    adapter: object
    pipeline: object
    runner: object
    capture: object


def build_live_runtime(config, capture, light_for):
    from aple.hal.piper_studio import PiperStudio
    from aple.hal.piper_motion import build_piper_motion_adapter
    from aple.execution.segment_validator import build_piper_segment_validator
    from aple.execution.holding_verifier import VisualHoldingVerifier
    from aple.execution.pick_place_pipeline import PickPlacePipeline
    from aple.inspection.in_hand_provider import InHandCaptureProvider
    from aple.orchestration.scene_runner import SceneRunner
    errors = missing_for_live_execution(config)
    if errors:
        raise ValueError('live execution configuration incomplete:\n'+'\n'.join(errors))
    if not callable(light_for):
        raise ValueError('live execution requires a calibrated lighting status provider')
    cfg = apply_attachment_geometry(config)
    studio = PiperStudio(cfg, snapshot_provider=capture)
    ik, fk = build_piper_kinematics(cfg)
    guard = build_piper_segment_validator(cfg, fk)
    def safe_snapshot():
        snap = capture()
        light = light_for(snap)
        if light is None or is_lighting_fault(light):
            raise RuntimeError('lighting cannot be verified; robot execution halted')
        return snap
    raw_provider = InHandCaptureProvider.from_config(cfg, safe_snapshot, fk)
    class StationaryCapture:
        def __init__(self):
            self._tool_pose_fn = fk
        def begin(self, apple, apple_in_tcp):
            raw_provider.begin(apple, apple_in_tcp)
        def __call__(self, apple_id, view_index, joints, *, timeout_s):
            tolerance = cfg['execution']['motion']['capture_joint_tolerance_rad']
            before = studio.joint_state()
            if max(abs(a-b) for a, b in zip(joints, before)) > tolerance:
                return None
            result = raw_provider(apple_id, view_index, before, timeout_s=timeout_s)
            after = studio.joint_state()
            return result if max(abs(a-b) for a, b in zip(after, before)) <= tolerance else None
    provider = StationaryCapture()
    verifier = VisualHoldingVerifier(cfg, provider, studio.joint_state)
    adapter = build_piper_motion_adapter(studio, cfg, ik, guard,
        tool_pose_provider=fk, in_hand_capture_provider=provider, holding_verifier=verifier)
    pipeline = PickPlacePipeline(cfg, adapter, inspection_begin=provider.begin)
    runner = None
    def scan_capture():
        if runner is not None and runner._execution_halted:
            raise RuntimeError('previous execution failed; refusing further scan motion')
        if not pipeline._executor.move_named('scan').success:
            raise RuntimeError('scan pose could not be reached safely')
        return safe_snapshot()
    observed_scene = {}
    def before_execute(item, apples, boxes):
        observed_scene.update(apples=tuple(apples), boxes=tuple(boxes))
        guard.clear_payload()
        guard.update_scene(apples, boxes, target_apple_id=item.apple.apple_id)
    runner = SceneRunner.from_config(cfg, pipeline, light_for=light_for,
        capture_for_retry=scan_capture, capture_for_verification=scan_capture,
        before_execute=before_execute)
    from aple.orchestration.mission_controller import MissionController
    from aple.schema import Phase
    def prepare_recovery(item):
        # APPROACH/DESCEND failures precede close in the scripted adapter.
        # Later phases can still hold an apple: never open or move to scan here.
        if item.execution is None or item.execution.phase_reached not in (Phase.APPROACH, Phase.DESCEND):
            return False
        if cfg['execution']['default_backend'] != 'scripted':
            return False
        joints = studio.joint_state()
        if len(joints) != 6 or not np.isfinite(joints).all():
            return False
        guard.clear_payload()
        guard.update_scene(observed_scene['apples'], observed_scene['boxes'], target_apple_id=None)
        return True
    def observe_drop():
        # Read-only observation; the arm has not yet been cleared out of view.
        return replace(safe_snapshot(), tray_clear=False)
    mission = MissionController(cfg, runner, scan_capture,
        prepare_recovery=prepare_recovery, observe_drop=observe_drop)
    return LiveRuntime(studio, adapter, pipeline, mission, scan_capture)
