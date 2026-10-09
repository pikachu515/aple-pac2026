"""Which apple / box placement gives the shortest pick → place → return cycle? Simulation only.

Runs every (apple position, box position) pair on a polar grid around the base with the
pick-place trial settings, in parallel, and ranks layouts by simulated cycle time.

  python tools/search_pick_place_layout.py                  # default grid, all CPU cores
  python tools/search_pick_place_layout.py --radii 0.28 0.36 0.44 --angles -45 0 45
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_WORKER = {}


def _init(config_path, sets):
    import yaml
    from aple.config import load_config
    from aple.execution.trajectory import build_piper_kinematics
    from aple.hal.gripper_physics import assumed_config, load_assumptions
    from aple.hal.pick_place_trial import load_trial
    from aple.hal.simulation import simulation_config
    os.chdir(ROOT)
    trial = load_trial(config_path)
    a = load_assumptions(ROOT/'configs/simulation/gripper_physics.yaml')
    a.update(yaml.safe_load((ROOT/trial['base_scene']).read_text())['in_hand_scene'].get('gripper_overrides', {}))
    base = assumed_config(simulation_config(load_config(), ROOT/'configs/simulation/piper.yaml'), a)
    _WORKER.update(trial=trial, a=a, base=base, sets=list(trial.get('set') or [])+sets,
                   kin=build_piper_kinematics(base))


def _run(job):
    import copy
    from aple.hal.pick_place_trial import apply_overrides, run_trial, split_overrides, trial_config
    from aple.hal.simulation_optimization import optimization_config
    w = _WORKER
    trial = copy.deepcopy(w['trial'])
    trial['apple']['center_m'] = list(job['apple'])
    trial['box']['center_m'] = list(job['box'])
    trial['other_boxes'] = 'park'
    row = dict(job, success=False, cycle_time_s=None, transfer_s=None, error='')
    try:
        trial_sets, config_sets = split_overrides(w['sets'])
        apply_overrides(trial, None, trial_sets)
        cfg = trial_config(w['base'], trial, ROOT/trial['base_scene'], seed=0)
        if trial.get('speed_profile'):
            cfg = optimization_config(cfg, ROOT/trial['speed_profile'])
        for key in cfg['execution']['timeouts']:
            cfg['execution']['timeouts'][key] = 300.
        a = copy.deepcopy(w['a'])
        apply_overrides(cfg, a, config_sets)
        report, _ = run_trial(cfg, a, ik=w['kin'][0], fk=w['kin'][1])
        stages = report['stages']
        row.update(success=report['success'], error=report['errors'][-1] if report['errors'] else '')
        if report['success']:
            row.update(cycle_time_s=report['cycle_time_s'],
                       transfer_s=stages['box_transfer']['simulated_time_s'],
                       pick_s=stages['pick']['simulated_time_s'],
                       landing_s=stages['landing_release_retreat']['simulated_time_s'],
                       return_s=stages['return_home']['simulated_time_s'])
    except Exception as exc:  # infeasible layout (overlap, IK, clearance) is a result, not a crash
        row['error'] = str(exc)
    return row


def polar(r, deg):
    return (round(r*math.cos(math.radians(deg)), 4), round(r*math.sin(math.radians(deg)), 4))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/simulation/pick_place_trial.yaml')
    parser.add_argument('--radii', type=float, nargs='+', default=[0.26, 0.33, 0.40, 0.47], help='distance from base (m)')
    parser.add_argument('--angles', type=float, nargs='+', default=[-60, -30, 0, 30, 60], help='bearing, deg (+ = left, +Y)')
    parser.add_argument('--min-distance', type=float, default=0.15, help='minimum apple-box center distance (m)')
    parser.add_argument('--set', action='append', default=[], metavar='KEY=VALUE')
    parser.add_argument('--workers', type=int, default=os.cpu_count())
    parser.add_argument('--output', type=Path, default=ROOT/'logs/pick-place-layout')
    args = parser.parse_args(argv)

    points = [(r, d) for r in args.radii for d in args.angles]
    jobs = []
    for ar, ad in points:
        for br, bd in points:
            apple, box = polar(ar, ad), polar(br, bd)
            if math.dist(apple, box) >= args.min_distance:
                jobs.append(dict(apple_r=ar, apple_deg=ad, box_r=br, box_deg=bd, apple=apple, box=box))
    output = args.output/time.strftime('%Y%m%d-%H%M%S')
    output.mkdir(parents=True)
    print(f'배치 {len(jobs)}개, 작업자 {args.workers} → {output}', flush=True)
    started = time.perf_counter()
    rows = []
    keys = ['apple_r', 'apple_deg', 'box_r', 'box_deg', 'success', 'cycle_time_s', 'pick_s',
            'transfer_s', 'landing_s', 'return_s', 'error']
    # Rows are written as they finish so an interrupted search keeps its results.
    with open(output/'layouts.csv', 'w', newline='') as f, \
            ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.config, args.set)) as pool:
        writer = csv.DictWriter(f, keys, extrasaction='ignore')
        writer.writeheader()
        for i, row in enumerate(pool.map(_run, jobs), 1):
            rows.append(row)
            writer.writerow(row)
            f.flush()
            if i % 20 == 0 or i == len(jobs):
                print(f'  {i}/{len(jobs)} ({time.perf_counter()-started:.0f}s)', flush=True)
    report(rows, output)
    return 0


def report(rows, output):
    ok = sorted((r for r in rows if r['success']), key=lambda r: r['cycle_time_s'])
    print(f'\n성공 {len(ok)}/{len(rows)}  (사이클 = 시작 자세 → 사과 → 상자 → 시작 자세, 물리 시간)')
    print('\n가장 빠른 배치 10개')
    for r in ok[:10]:
        print(f"  사과 r={r['apple_r']:.2f} {r['apple_deg']:+4.0f}°  상자 r={r['box_r']:.2f} {r['box_deg']:+4.0f}°"
              f"  사이클 {r['cycle_time_s']:.2f}s (집기 {r['pick_s']:.2f} 이송 {r['transfer_s']:.2f}"
              f" 놓기 {r['landing_s']:.2f} 복귀 {r['return_s']:.2f})")

    def group(name, select):
        values = [r['cycle_time_s'] for r in ok if select(r)]
        moves = [r['transfer_s'] for r in ok if select(r)]
        if values:
            print(f'  {name:28s} {len(values):3d}개  사이클 평균 {statistics.mean(values):.2f}s'
                  f'  상자이송 평균 {statistics.mean(moves):.2f}s')
    print('\n옮기는 방향 (사과 → 상자)')
    group('옆 (같은 거리, 각도만 다름)', lambda r: r['apple_r'] == r['box_r'])
    group('앞뒤 (같은 각도, 거리만 다름)', lambda r: r['apple_deg'] == r['box_deg'])
    group('대각선 (둘 다 다름)', lambda r: r['apple_r'] != r['box_r'] and r['apple_deg'] != r['box_deg'])
    print('\n사과 위치별 (그 사과 위치에서 가능한 모든 상자 위치의 평균 / 최솟값)')
    for key in sorted({(r['apple_r'], r['apple_deg']) for r in ok}):
        values = [r['cycle_time_s'] for r in ok if (r['apple_r'], r['apple_deg']) == key]
        print(f'  r={key[0]:.2f} {key[1]:+4.0f}°  평균 {statistics.mean(values):.2f}s  최소 {min(values):.2f}s')
    print('\n상자 위치별')
    for key in sorted({(r['box_r'], r['box_deg']) for r in ok}):
        values = [r['cycle_time_s'] for r in ok if (r['box_r'], r['box_deg']) == key]
        print(f'  r={key[0]:.2f} {key[1]:+4.0f}°  평균 {statistics.mean(values):.2f}s  최소 {min(values):.2f}s')
    print(f'\n전체 결과 {output}/layouts.csv')


if __name__ == '__main__':
    raise SystemExit(main())
