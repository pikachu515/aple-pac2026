"""Single apple pick → box → return timing trial. MuJoCo simulation only, no hardware.

Examples:
  python tools/run_pick_place_trial.py --seed 1
  python tools/run_pick_place_trial.py --seed 1 --repeat 5
  python tools/run_pick_place_trial.py --seed 1 --live      # side view at real time
  python tools/run_pick_place_trial.py --apple 0.42 -0.10 --box 0.30 0.30
  python tools/run_pick_place_trial.py --seed 1 --repeat 3 \\
      --compare simulation.in_hand_scene.loaded_joint_velocity_rad_s=2,4,8.8
"""
import argparse
import copy
import json
from pathlib import Path
import statistics
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OPEN_WINDOWS = []
SHOWN = (('pick', '집기'), ('empty_transfer', '빈손이송'), ('box_transfer', '상자이송'),
         ('landing_release_retreat', '착지·놓기'), ('return_home', '복귀'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/simulation/pick_place_trial.yaml')
    parser.add_argument('--seed', type=int, help='box layout seed; omitted = new random layout')
    parser.add_argument('--repeat', type=int, default=1, help='run seeds seed, seed+1, ... (different box positions)')
    parser.add_argument('--apple', type=float, nargs=2, metavar=('X', 'Y'), help='assumed apple center (m)')
    parser.add_argument('--box', type=float, nargs=2, metavar=('X', 'Y'), help='fix the target box center (m)')
    parser.add_argument('--target', choices=('UPPER', 'MIDDLE', 'REJECT'))
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                        help='override a virtual-config key; trial.KEY edits the trial file, '
                             'assumed_gripper.KEY the physics assumptions')
    parser.add_argument('--compare', metavar='KEY=V1,V2,...', help='repeat the same seeds for each value')
    parser.add_argument('--video', action='store_true', help='save side.mp4 per run (slower)')
    parser.add_argument('--live', action='store_true', help='show a side-view window paced to real time')
    parser.add_argument('--live-rate', type=float, default=1.0, help='window playback rate (1 = real time)')
    parser.add_argument('--output', type=Path, default=ROOT/'logs/pick-place-trial')
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error('--repeat must be at least 1')
    if args.live and args.video:
        parser.error('--live and --video cannot be combined')
    if not args.live_rate > 0:
        parser.error('--live-rate must be positive')

    started = time.perf_counter()
    from aple.config import load_config
    from aple.execution.trajectory import build_piper_kinematics
    from aple.hal.gripper_physics import assumed_config, load_assumptions
    from aple.hal.pick_place_trial import (apply_overrides, load_trial, load_trial_dict, run_trial,
                                           split_overrides, trial_config)
    from aple.hal.simulation import simulation_config

    trial = load_trial(args.config)
    if args.apple:
        trial['apple']['center_m'] = list(args.apple)
    if args.box:
        trial['box']['center_m'] = list(args.box)
    if args.target:
        trial['target_box'] = args.target
    assumptions = load_assumptions(ROOT/'configs/simulation/gripper_physics.yaml')
    assumptions.update(yaml.safe_load((ROOT/trial['base_scene']).read_text())['in_hand_scene'].get('gripper_overrides', {}))
    base = assumed_config(simulation_config(load_config(), ROOT/'configs/simulation/piper.yaml'), assumptions)
    ik = fk = None

    first_seed = args.seed if args.seed is not None else int(time.time()) % 100000
    seeds = [first_seed+i for i in range(args.repeat)]
    conditions = [None]
    if args.compare:
        key, _, values = args.compare.partition('=')
        conditions = [f'{key}={value}' for value in values.split(',') if value]
        if not key or not conditions:
            parser.error('--compare needs KEY=V1,V2,...')
    stamp = time.strftime('%Y%m%d-%H%M%S')
    output = args.output/stamp
    suffix = 2
    while output.exists():
        output = args.output/f'{stamp}-{suffix}'
        suffix += 1
    output.mkdir(parents=True)
    print(f'설정 {args.config.relative_to(ROOT) if args.config.is_relative_to(ROOT) else args.config} → {output}', flush=True)

    rows = []
    for condition in conditions:
        if condition:
            print(f'\n조건 {condition}', flush=True)
        for seed in seeds:
            trial_sets, config_sets = split_overrides(list(trial.get('set') or [])+args.set+([condition] if condition else []))
            case = copy.deepcopy(trial)
            apply_overrides(case, None, trial_sets)
            case = load_trial_dict(case)
            cfg = trial_config(base, case, ROOT/case['base_scene'], seed=seed)
            if case.get('speed_profile'):
                from aple.hal.simulation_optimization import optimization_config
                cfg = optimization_config(cfg, ROOT/case['speed_profile'])
            for key in cfg['execution']['timeouts']:
                cfg['execution']['timeouts'][key] = 300.
            a = copy.deepcopy(assumptions)
            apply_overrides(cfg, a, config_sets)
            if ik is None:
                ik, fk = build_piper_kinematics(cfg)
            run_dir = output/(f'{len(rows)+1:02d}-seed-{seed}')
            run_dir.mkdir()
            if args.video or args.live:
                cfg['simulation']['in_hand_scene']['video_fps'] = 30
            frame, close_video = (_video(run_dir, cfg) if args.video else
                                  _live(args.live_rate, last=len(rows)+1 == len(seeds)*len(conditions)) if args.live
                                  else (None, lambda: None))
            try:
                report, transport = run_trial(cfg, a, ik=ik, fk=fk, sample_callback=frame)
            finally:
                close_video()
            report.update(condition=condition, seed=seed, assumptions=a,
                          full_scene_complete=report['success'], tested_apple_count=1,
                          apples=[dict(apple=1)] if report['success'] else [])
            (run_dir/'virtual-config.yaml').write_text(yaml.safe_dump(cfg))
            (run_dir/'assumptions.yaml').write_text(yaml.safe_dump(dict(assumed_gripper=a)))
            (run_dir/'trace.json').write_text(json.dumps(dict(events=transport.events,
                apples=[dict(center_m=report['placement']['place_center_m'])]), indent=2))
            (run_dir/'report.json').write_text(json.dumps(report, indent=2))
            rows.append(report)
            _print_run(report, cfg)

    _summary(rows, conditions, output)
    print(f'전체 실제 경과 {time.perf_counter()-started:.1f}s · 결과 {output}', flush=True)
    for viewer in OPEN_WINDOWS:
        print('창을 닫으면 종료합니다.', flush=True)
        while viewer.is_running():
            time.sleep(.1)
        viewer.close()
    return 0 if all(r['success'] for r in rows) else 1


def _print_run(report, cfg):
    box = cfg['simulation']['pick_place_trial']['box_centers_m'][report['trial']['target_box']]
    stages = report['stages']
    parts = ' '.join(f"{label} {stages[name]['simulated_time_s']:.2f}" for name, label in SHOWN if name in stages)
    status = f"사이클 {report['cycle_time_s']:.2f}s" if report['success'] else f"실패: {report['errors'][-1]}"
    print(f"  seed {report['seed']:>6} 상자({box[0]:+.3f},{box[1]:+.3f}) {status} | {parts} | 계산 {report['wall_time_s']:.1f}s", flush=True)


def _summary(rows, conditions, output):
    lines = ['condition,seed,success,cycle_time_s,'+','.join(n for n, _ in SHOWN)+',wall_time_s,error']
    for r in rows:
        stage = [f"{r['stages'][n]['simulated_time_s']:.3f}" if n in r['stages'] else '' for n, _ in SHOWN]
        lines.append(','.join([r['condition'] or '', str(r['seed']), str(r['success']),
            f"{r.get('cycle_time_s', float('nan')):.3f}", *stage, f"{r['wall_time_s']:.2f}",
            '"'+(r['errors'][-1].replace('"', "'") if r['errors'] else '')+'"']))
    (output/'runs.csv').write_text('\n'.join(lines)+'\n')
    print('\n요약 (사이클 = 집기 시작 ~ 시작 자세 복귀, 물리 시간)', flush=True)
    for condition in conditions:
        done = [r['cycle_time_s'] for r in rows if r['condition'] == condition and r['success']]
        total = sum(r['condition'] == condition for r in rows)
        name = condition or '기본'
        if done:
            print(f'  {name}: 성공 {len(done)}/{total} · 평균 {statistics.mean(done):.2f}s'
                  f' · 최소 {min(done):.2f}s · 최대 {max(done):.2f}s', flush=True)
        else:
            print(f'  {name}: 성공 0/{total}', flush=True)


def _live(rate, *, last):
    """MuJoCo window on the side camera; sleeps so simulated time follows the wall clock."""
    import mujoco
    import mujoco.viewer
    state = {}

    def frame(model, data):
        if not state:
            state['viewer'] = mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False)
            state['viewer'].cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            state['viewer'].cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'side')
            state['start'] = (time.perf_counter(), data.time)
        if not state['viewer'].is_running():
            raise KeyboardInterrupt('live window closed')
        wall0, sim0 = state['start']
        ahead = (data.time-sim0)/rate - (time.perf_counter()-wall0)
        if ahead > 0:
            time.sleep(ahead)
        state['viewer'].sync()

    def close():
        if state:
            if last:
                OPEN_WINDOWS.append(state['viewer'])
            else:
                time.sleep(1.)
                state['viewer'].close()
    return frame, close


def _video(run_dir, cfg):
    import cv2
    import mujoco
    state = {}

    def frame(model, data):
        if not state:
            state['renderer'] = mujoco.Renderer(model, height=480, width=640)
            state['writer'] = cv2.VideoWriter(str(run_dir/'side.mp4'), cv2.VideoWriter_fourcc(*'mp4v'),
                                              cfg['simulation']['in_hand_scene']['video_fps'], (640, 480))
        state['renderer'].update_scene(data, camera='side')
        picture = state['renderer'].render()[..., ::-1].copy()
        cv2.putText(picture, f'{data.time:.1f}s', (12, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1, cv2.LINE_AA)
        state['writer'].write(picture)

    def close():
        if state:
            state['writer'].release()
            state['renderer'].close()
    return frame, close


if __name__ == '__main__':
    raise SystemExit(main())
