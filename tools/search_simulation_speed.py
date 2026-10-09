"""Descending, reproducible contact-simulation speed trials; never commands hardware."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def candidate(base, multiplier):
    settings = dict(base)
    for key in ('empty_joint_velocity_rad_s', 'loaded_joint_velocity_rad_s',
                'rotation_joint_velocity_rad_s'):
        settings[key] *= multiplier
    # Keep contact descent at the already-tested absolute joint speed.
    settings['landing_speed_scale'] = (
        base['landing_speed_scale'] * base['loaded_joint_velocity_rad_s']
        / settings['loaded_joint_velocity_rad_s'])
    return settings


def complete_success(report, exit_code):
    """A partial run, missing captures or any slip cannot pass the speed gate."""
    apples = report.get('apples', [])
    return (exit_code == 0 and report.get('success') is True
            and report.get('full_scene_complete') is True and len(apples) == 3
            and all(a.get('captured_views') == 4 and a.get('execution_success') is True
                    for a in apples) and report.get('slips') == []
            and report.get('errors') == [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default='/home/seo/.venvs/piper-daemons/bin/python3')
    parser.add_argument('--multipliers', nargs='+', type=float, default=[8, 6, 4, 3.5, 3, 2.5, 2, 1.5, 1.25, 1])
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 7, 13, 99, 2026])
    parser.add_argument('--scene-config', type=Path, default=ROOT/'configs/simulation/in_hand_speed_search_scene.yaml')
    args = parser.parse_args()
    if any(not 0 < m <= 32 for m in args.multipliers):
        parser.error('multipliers must be positive and at most 32')
    if len(set(args.seeds)) != len(args.seeds):
        parser.error('use distinct seeds')
    root = ROOT / 'logs/speed-search' / time.strftime('%Y%m%d-%H%M%S')
    root.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load((ROOT/'configs/simulation/in_hand_fast.yaml').read_text())['optimization']
    base['ik_seed_retry_rad'] = .15
    base['joint_place_approach'] = True
    summary = dict(scope='simulation_only', seeds=args.seeds, apples_per_seed=3,
                   candidate_family='common speed multiplier; unchanged contact descent',
                   trials=[], selected=None)
    print(f'Speed search: {root}', flush=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', MUJOCO_GL='egl')
    for multiplier in sorted(set(args.multipliers), reverse=True):
        settings = candidate(base, multiplier)
        directory = root/f'multiplier-{multiplier:g}'
        directory.mkdir()
        config = directory/'optimization.yaml'
        config.write_text(yaml.safe_dump(dict(optimization=settings)))
        trial = dict(multiplier=multiplier, settings=settings, runs=[], all_passed=False)
        summary['trials'].append(trial)
        for seed in args.seeds:
            print(f'Try multiplier={multiplier:g}, seed={seed}, all 3 apples', flush=True)
            with (directory/f'seed-{seed}.log').open('w') as log:
                result = subprocess.run([
                    args.python, str(ROOT/'tools/run_in_hand_simulation.py'),
                    '--seed', str(seed), '--optimize', '--optimization-config', str(config),
                    '--scene-config', str(args.scene_config),
                    '--output', str(directory)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            path = directory/f'seed-{seed}/report.json'
            report = json.loads(path.read_text()) if path.exists() else {}
            passed = complete_success(report, result.returncode)
            run = dict(seed=seed, passed=passed, exit_code=result.returncode,
                       report=str(path.relative_to(ROOT)), errors=report.get('errors', ['Initialization failed; inspect seed log']),
                       simulated_time_s=report.get('simulated_time_s'),
                       processed_apples=len(report.get('apples', [])))
            trial['runs'].append(run)
            (root/'summary.json').write_text(json.dumps(summary, indent=2))
            print(json.dumps(run), flush=True)
            if not passed:
                break
        trial['all_passed'] = len(trial['runs']) == len(args.seeds) and all(r['passed'] for r in trial['runs'])
        if trial['all_passed']:
            summary['selected'] = dict(multiplier=multiplier, config=str(config.relative_to(ROOT)),
                                       observed_successful_apples=3*len(args.seeds))
        (root/'summary.json').write_text(json.dumps(summary, indent=2))
        if not report:
            # Speed changes cannot repair a layout that fails before motion.
            break
        if trial['all_passed']:
            break
    print(json.dumps(summary['selected']), flush=True)
    return 0 if summary['selected'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
