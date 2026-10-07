"""Plan and physically move apples in Studio's assumed-gripper simd."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aple.config import load_config
from aple.hal.simulation import simulation_config
from aple.hal.gripper_physics import load_assumptions, assumed_config, physics_scene, closing_gap, evaluate_outcomes
from aple.hal.studio_simulation import StudioSimulationReplay
from aple.hal.piper_studio import PiperWebError
from tools.run_cycle import main as run_cycle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1')
    parser.add_argument('--assumptions', type=Path, default=ROOT/'configs/simulation/gripper_physics.yaml')
    parser.add_argument('--output', type=Path, default=ROOT/'logs/studio-gripper-physics')
    args = parser.parse_args(argv)
    a = load_assumptions(args.assumptions)
    cfg = assumed_config(simulation_config(load_config(), ROOT/'configs/simulation/piper.yaml'), a)
    args.output.mkdir(parents=True, exist_ok=True)
    overlay = args.output/'virtual-config.yaml'
    overlay.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    trace_path = args.output/'planned-trace.json'
    rows = run_cycle(['--simulate', '--simulation-config', str(overlay), '--trace', str(trace_path)])
    trace = json.loads(trace_path.read_text())
    count = len(cfg['simulation']['apples'])
    releases = [e['apple_index'] for e in trace['events'] if e['action'] == 'release']
    if (len(rows) != count or not all(row.success for row in rows)
            or len(trace['apples']) != count or sorted(releases) != list(range(count))):
        raise ValueError('Planning did not complete all apples; refusing Studio motion')
    sim = StudioSimulationReplay(cfg, base_url=args.url)
    digest = hashlib.sha256(json.dumps(a, sort_keys=True).encode()).hexdigest()

    def positions():
        objects = sim._request('GET', '/api/sim/scenes/live/objects')['objects']
        apples = {o['id']: o for o in objects if o['id'].startswith('aple_apple_')}
        if len(apples) != len(trace['apples']) or any(o.get('aple_assumptions_sha256') != digest for o in apples.values()):
            raise PiperWebError('Studio must run studio_assumed_simd.py with these exact assumptions')
        return np.asarray([apples[f'aple_apple_{i+1}']['pos'] for i in range(len(apples))])

    sim.prepare(physics_scene(cfg, a), backup_path=args.output/'previous-scene.json')
    sim.connect_virtual_cameras(('sim:top', 'sim:front', 'sim:side'))
    initial = positions()
    peak = initial[:, 2].copy()
    # A grasped apple prevents the fingers reaching the commanded compression.
    # This tolerance applies to this sim-only client, never to the hardware HAL.
    sim.client._arrival_tolerance_rad = .005
    targets = []
    close_next = False
    for event in trace['events']:
        if event['action'] == 'grasp':
            close_next = True
        elif event['action'] == 'move':
            targets.append((sim.client._normalize_joint_target(tuple(event['joints_rad'])), False))
        elif event['action'] == 'gripper':
            gap = closing_gap(event['width_m'], grasp=close_next, assumptions=a)
            targets.append(({'gripper': sim.client._normalize_gripper_width(gap)}, close_next))
            close_next = False
    report = dict(mode='studio_assumed_gripper_contact_physics', assumptions=a,
                  goals_reached=0, success=False, transport_errors=[])
    print(f'물리 파지 시험: {args.url}/scene — 가상 팔만 움직입니다.', flush=True)
    try:
        with sim.session() as keepalive:
            for i, (target, contact_close) in enumerate(targets, 1):
                # Force-limited closing may stop against a fruit before the
                # position target. Its success is determined by lifting below.
                # Opening must reach its target before approaching/releasing.
                sim.client._gripper_arrival_tolerance = 100 if contact_close else 2
                sim._send(target, keepalive, 15)
                time.sleep(a['gripper_settle_s'] if 'gripper' in target else .03)
                peak[:] = np.maximum(peak, positions()[:, 2])
                report['goals_reached'] = i
                if i % 20 == 0:
                    print(f'물리 시험 {i}/{len(targets)}', flush=True)
            time.sleep(a['final_settle_s'])
    except Exception as exc:
        report['transport_errors'].append(str(exc))
    finally:
        try:
            final = positions()
            report['apples'] = evaluate_outcomes(cfg, trace, a, initial, peak, final)
            report['success'] = not report['transport_errors'] and all(o['success'] for o in report['apples'])
        finally:
            (args.output/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report['success'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
