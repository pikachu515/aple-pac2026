"""Colour sort of central apples (red → left box, yellow → right box). Simulation only.

Planning time on this computer is counted as the arm holding still, so the simulated
time equals what a real arm driven by the same code would take. The run is then
played back in a side-view window at real-time speed.

  python tools/run_color_sort.py                 # new random layout and colours
  python tools/run_color_sort.py --seed 3
  python tools/run_color_sort.py --seed 3 --rate 0.5      # slow-motion playback
  python tools/run_color_sort.py --seed 3 --no-window     # numbers only
  python tools/run_color_sort.py --seed 3 --fast          # exclude planning time (motion only)
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAMES = {'red': '빨강', 'yellow': '노랑'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/simulation/color_sort_trial.yaml')
    parser.add_argument('--seed', type=int, help='apple layout/colour seed; omitted = new random')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--no-window', action='store_true', help='skip the playback window')
    parser.add_argument('--fast', action='store_true', help='do not add planning time as holding time')
    parser.add_argument('--rate', type=float, default=1.0, help='window playback rate (1 = real time)')
    parser.add_argument('--output', type=Path, default=ROOT/'logs/color-sort')
    args = parser.parse_args(argv)

    from aple.config import load_config
    from aple.hal.color_sort_trial import (FrameRecorder, RealtimeClock, load_sort, run_sort, sort_config,
                                          use_hardware_motion)
    from aple.hal.gripper_physics import assumed_config, load_assumptions
    from aple.hal.pick_place_trial import apply_overrides, split_overrides
    from aple.hal.simulation import simulation_config
    from aple.hal.simulation_optimization import optimization_config

    sort = load_sort(args.config)
    sets = list(sort.get('set') or [])+args.set
    trial_sets, config_sets = split_overrides(sets)
    apply_overrides(sort, None, trial_sets)
    a = load_assumptions(ROOT/'configs/simulation/gripper_physics.yaml')
    a.update(yaml.safe_load((ROOT/sort['base_scene']).read_text())['in_hand_scene'].get('gripper_overrides', {}))
    base = assumed_config(simulation_config(load_config(), ROOT/'configs/simulation/piper.yaml'), a)
    seed = args.seed if args.seed is not None else int(time.time()) % 100000
    cfg = sort_config(base, sort, ROOT/sort['base_scene'], seed=seed)
    if sort.get('speed_profile'):
        cfg = optimization_config(cfg, ROOT/sort['speed_profile'])
    cfg = use_hardware_motion(cfg, load_config())  # same path settings as the real arm
    for key in cfg['execution']['timeouts']:
        cfg['execution']['timeouts'][key] = 300.
    a = copy.deepcopy(a)
    apply_overrides(cfg, a, config_sets)
    cfg['simulation']['in_hand_scene']['video_fps'] = 30

    colors = cfg['simulation']['color_sort']['colors']
    print(f'seed {seed}: ' + ', '.join(f'사과{i+1} {NAMES[c]} {tuple(round(v, 3) for v in ap["center_m"][:2])}'
          for i, (c, ap) in enumerate(zip(colors, cfg['simulation']['apples']))), flush=True)
    clock = None if args.fast else RealtimeClock()
    recorder = None if args.no_window else FrameRecorder(
        lambda sim, wall: print(f'재생 끝: 시뮬레이션 {sim:.2f}s 를 실제 {wall:.2f}s 동안 보여 줌', flush=True))
    report, transport = run_sort(cfg, a, clock=clock, sample_callback=recorder)

    output = args.output/f'seed-{seed}'
    suffix = 2
    while output.exists():
        output = args.output/f'seed-{seed}-{suffix}'
        suffix += 1
    output.mkdir(parents=True)
    report.update(seed=seed, assumptions=a, full_scene_complete=report['success'],
                  tested_apple_count=len(colors))
    (output/'virtual-config.yaml').write_text(yaml.safe_dump(cfg))
    (output/'assumptions.yaml').write_text(yaml.safe_dump(dict(assumed_gripper=a)))
    (output/'trace.json').write_text(json.dumps(dict(events=transport.events, apples=report['trace_apples']), indent=2))
    (output/'report.json').write_text(json.dumps(report, indent=2))

    for item in report['apples']:
        side = '왼쪽' if item['color'] == 'red' else '오른쪽'
        print(f"  사과{item['apple']} {NAMES[item['color']]} → {side} 상자 칸 {item['cell_index']}  {item['time_s']:.2f}s", flush=True)
    if report['success']:
        line = f"성공 | 전체 {report['total_time_s']:.2f}s (시작 → 사과 {len(colors)}개 분류 → 시작)"
        if clock:
            line += f" | 그중 경로 계산 대기 {report['planning_wait_s']:.2f}s"
        print(line, flush=True)
    else:
        print(f"실패: {report['errors'][-1]}", flush=True)
    print(f'결과 {output}', flush=True)
    if recorder:
        print(f'옆 카메라 창에서 실제 시간 속도(x{args.rate:g})로 재생합니다. 창을 닫으면 종료합니다.', flush=True)
        recorder.play(transport.model, rate=args.rate)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
