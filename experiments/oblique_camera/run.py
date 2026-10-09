"""Contact-physics central rotation/color inspection. Simulation only, no hardware."""
import argparse
import json
from pathlib import Path
import sys
import traceback
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output', type=Path, default=ROOT/'logs/oblique-camera-contact')
    parser.add_argument('--scene-config', type=Path, default=ROOT/'configs/simulation/in_hand_scene.yaml')
    parser.add_argument('--video', action='store_true', help='save side-view MP4 under logs (never commit)')
    parser.add_argument('--quick', action='store_true', help='execute one apple, no video/capture files; retain all scene obstacles')
    parser.add_argument('--apple-count', type=int, choices=(1, 2, 3), help='execute physical apples 1..N, preserving the full seeded scene')
    parser.add_argument('--save-captures', action='store_true', help='save inspection RGB/depth files (off by default)')
    parser.add_argument('--optimize', action='store_true', help='compare optional simulation-only route/speed/compute candidate')
    parser.add_argument('--optimization-config', type=Path, default=ROOT/'configs/simulation/in_hand_vertical.yaml')
    args = parser.parse_args(argv)
    if args.quick and (args.video or args.save_captures or args.apple_count not in (None, 1)):
        parser.error('--quick requires one apple and no video/capture file saving')
    apple_count = args.apple_count or (1 if args.quick else 3)
    started = time.perf_counter()
    transport = None
    from aple.hal.simulation_timing import SimulationTiming
    timing = SimulationTiming(lambda: float(transport.data.time) if transport is not None else 0.)
    from aple.config import load_config
    from aple.hal.simulation import simulation_config, KinematicWorld
    from aple.hal.gripper_physics import load_assumptions, assumed_config, evaluate_outcomes, physics_scene
    from aple.hal.in_hand_simulation import scene_config, CentralInspectionAdapter, radial_grasps
    from experiments.oblique_camera.comparison import CameraComparisonTransport as ContactTransport
    from aple.execution.trajectory import build_piper_kinematics
    from aple.execution.segment_validator import build_piper_segment_validator
    from aple.execution.pick_place_pipeline import PickPlacePipeline
    from aple.orchestration.scene_runner import SceneRunner
    from aple.inspection.coloration import params_from_config
    from aple.schema import TaskAction

    a = load_assumptions(ROOT/'configs/simulation/gripper_physics.yaml')
    a.update(yaml.safe_load(args.scene_config.read_text())['in_hand_scene'].get('gripper_overrides', {}))
    cfg = scene_config(assumed_config(simulation_config(load_config(), ROOT/'configs/simulation/piper.yaml'), a), args.scene_config, seed=args.seed)
    if args.optimize:
        from aple.hal.simulation_optimization import optimization_config
        cfg = optimization_config(cfg, args.optimization_config)
        opaque_radius = cfg['simulation']['in_hand_scene']['optimization'].get('opaque_cup_radius_m')
        if opaque_radius is not None:
            a['opaque_cup_radius_m'] = opaque_radius
    # Longer deadlines for an intentionally slow, loaded virtual arm.
    for key in cfg['execution']['timeouts']:
        cfg['execution']['timeouts'][key] = 300.
    base = args.output/f'seed-{cfg["simulation"]["in_hand_scene"]["seed"]}'
    output = base
    i = 2
    while output.exists():
        output = base.with_name(f'{base.name}-{i}')
        i += 1
    output.mkdir(parents=True)
    (output/'virtual-config.yaml').write_text(yaml.safe_dump(cfg))
    (output/'assumptions.yaml').write_text(yaml.safe_dump(dict(assumed_gripper=a)))
    (output/'scene.json').write_text(json.dumps(physics_scene(cfg,a),indent=2))
    print(f'가상 중앙 검사·무작위 장면: {output}', flush=True)
    ik, fk = build_piper_kinematics(cfg)
    validator = build_piper_segment_validator(cfg, fk)
    video_writer = renderer = transport = None
    def render_frame(model, data):
        nonlocal renderer, video_writer
        import mujoco
        import cv2
        if renderer is None:
            renderer = mujoco.Renderer(model, height=480, width=640)
            video_writer = cv2.VideoWriter(str(output/'side.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), cfg['simulation']['in_hand_scene']['video_fps'], (640,480))
            if not video_writer.isOpened():
                raise RuntimeError('Cannot open MP4 writer')
        renderer.update_scene(data, camera='side')
        picture = renderer.render()[..., ::-1].copy()
        state = 'ROTATE / INSPECT' if transport and transport.rotating else 'CARRY' if transport and transport.holding else 'EMPTY (FAST)'
        cv2.putText(picture, f'{data.time:.1f}s  {state}', (12,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255,255,255), 1, cv2.LINE_AA)
        video_writer.write(picture)
    def frame(model, data):
        with timing.measure('video_render_write'):
            render_frame(model, data)

    def save_capture(apple, view, color, depth):
        import cv2
        directory = output/'captures'
        directory.mkdir(exist_ok=True)
        cv2.imwrite(str(directory/f'apple-{apple+1}-view-{view}.png'), color)
        np.save(directory/f'apple-{apple+1}-view-{view}-depth.npy', depth)
    def capture_frame(apple, view, color, depth):
        with timing.measure('capture_file_write'):
            save_capture(apple, view, color, depth)

    with timing.measure('physics_initialization'):
        transport = ContactTransport(cfg, a, fk, sample_callback=frame if args.video else None,
                                     capture_callback=capture_frame if args.save_captures else None)
    transport.timing = timing
    adapter = CentralInspectionAdapter(transport, cfg, ik, validator)
    pipeline = PickPlacePipeline(cfg, adapter, inspection_begin=transport.begin)
    runner = SceneRunner.from_config(cfg, pipeline)
    runner._grasps = lambda apples: radial_grasps(apples, cfg)
    synthetic = KinematicWorld(cfg, fk, validator)
    report = dict(mode='simulation_only_central_in_hand_contact', assumptions=a,
                  settings=cfg['simulation']['in_hand_scene'], success=False, errors=[], apples=[],
                  tested_apple_count=apple_count, scene_apple_count=len(cfg['simulation']['apples']),
                  full_scene_complete=False, video_saved=args.video, captures_saved=args.save_captures)
    targets = [dict(center_m=apple['center_m']) for apple in cfg['simulation']['apples']]
    try:
        with timing.measure('scene_capture_plan'):
            plan = runner.plan(synthetic.capture())
        if plan.work is None:
            raise RuntimeError(f'No work: {plan}')
        if len(plan.work.items) != len(targets) or any(item.task.action != TaskAction.PICK_INSPECT_PLACE for item in plan.work.items):
            raise RuntimeError('Every configured apple must have a valid inspection pick plan')
        for item in plan.work.items:
            physical_index = int(np.argmin([np.linalg.norm(np.array(a['center_m'])-np.array(item.apple.center_m)) for a in cfg['simulation']['apples']]))
            if physical_index >= apple_count:
                continue
            timing.apple = physical_index+1
            print(f'사과 {item.apple.apple_id}: 집기 → 중앙 이동 → 회전·촬영 → 재판정 → 적재', flush=True)
            with timing.measure('pick'):
                held = pipeline.pick_for_inspection(item)
            if not held.pick.success:
                raise RuntimeError(f'Pick failed: {held.pick.failure_code.name}')
            with timing.measure('inspection'):
                held = pipeline.inspect_held_apple(held)
            if not held.inspection.success:
                raise RuntimeError(f'Inspection failed: {held.inspection.failure_code.name}')
            with timing.measure('regrade'):
                held = pipeline.regrade_held_apple(held, params_from_config(cfg))
            print(f'  재판정: {held.regraded_grade.grade.name}, 착색 {held.regraded_grade.ratio:.3f}, 관측 {held.regraded_grade.coverage:.3f}', flush=True)
            with timing.measure('place'):
                placed = pipeline.place_regraded_apple(held, held.regraded_grade, plan.boxes)
            if placed.placement is not None:
                targets[physical_index]['center_m'] = list(placed.placement.place_center_m)
            grade = held.regraded_grade
            report['apples'].append(dict(apple=physical_index+1, perception_apple_id=item.apple.apple_id, diameter_m=item.apple.diameter_m,
                mass_kg=cfg['simulation']['apples'][physical_index]['mass_kg'],
                captured_views=held.inspection.captured_views, grade=grade.grade.name,
                reject_reason=grade.reject_reason.name, color_ratio=grade.ratio,
                ratio_lo=grade.ratio_lo, ratio_hi=grade.ratio_hi, coverage=grade.coverage,
                execution_success=bool(placed.execution and placed.execution.success),
                failure_code=placed.execution.failure_code.name if placed.execution else None,
                phase=placed.execution.phase_reached.name if placed.execution else None))
            if not placed.execution or not placed.execution.success:
                raise RuntimeError('Placement failed')
        timing.apple = None
        with timing.measure('final_settle'):
            transport.step(a['final_settle_s'])
        outcomes = evaluate_outcomes(cfg, dict(apples=targets), a, transport.initial, transport.peak, transport.positions())
        report['outcomes'] = outcomes[:apple_count]
        report['success'] = len(report['apples']) == apple_count and all(o['success'] for o in outcomes[:apple_count])
        report['full_scene_complete'] = report['success'] and apple_count == len(targets)
    except Exception as exc:
        report['errors'].append(str(exc))
        traceback.print_exc()
    finally:
        report.update(slips=transport.slips, views=transport.views, motion_errors=transport.motion_errors,
                      route_choices=transport.route_choices,
                      camera_comparison=transport.camera_report(),
                      landing_segments=transport.landing_segments,
                      simulated_time_s=float(transport.data.time))
        (output/'trace.json').write_text(json.dumps(dict(events=transport.events, apples=targets), indent=2))
        if video_writer is not None:
            video_writer.release()
        if renderer is not None:
            renderer.close()
        report['wall_time_s'] = time.perf_counter()-started
        report['timing'] = timing.summary()
        (output/'report.json').write_text(json.dumps(report, indent=2))
        lines = ['stage,calls,failures,wall_time_s,simulated_time_s,exclusive_wall_time_s,exclusive_simulated_time_s']
        for stage, values in report['timing']['stages'].items():
            lines.append(','.join([stage]+[str(values[key]) for key in lines[0].split(',')[1:]]))
        (output/'timings.csv').write_text('\n'.join(lines)+'\n')
    print(f"성공: {report['success']} | 사과 {len(report['apples'])}/{apple_count} | 실제 실행 {report['wall_time_s']:.1f}s | 물리 시간 {report['simulated_time_s']:.1f}s", flush=True)
    for stage in ('scene_capture_plan', 'pick', 'empty_transfer', 'inspection', 'center_transfer', 'rotation_inspection', 'color_capture', 'regrade', 'place', 'box_transfer', 'landing_release_retreat', 'final_settle'):
        values = report['timing']['stages'].get(stage)
        if values:
            print(f"  {stage}: 실제 {values['wall_time_s']:.2f}s / 물리 {values['simulated_time_s']:.2f}s", flush=True)
    print(f'보고서: {output}/report.json · timings.csv', flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
