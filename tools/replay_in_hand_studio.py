"""Replay a completed central-inspection physics run on Studio's virtual arm only."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def initial_virtual_reader(reader, keepalive, *, max_wait_s=1.):
    """Wait briefly for a reset virtual gripper's encoder to enter valid bounds.

    No clamping/fabrication: the normal decoder must return a valid state.
    Used only for the first opening command of a guarded simulated session.
    """
    from aple.hal.piper_studio import PiperWebError
    def read():
        deadline = time.monotonic()+max_wait_s
        while True:
            try:
                return reader()
            except PiperWebError as exc:
                if str(exc) != 'Piper Web returned a gripper encoder value outside calibration' or time.monotonic() >= deadline:
                    raise
                keepalive.check()
                time.sleep(.02)
    return read


def open_virtual_gripper(sim, width_m, keepalive, *, timeout_s=15):
    """Bounded transient encoder retries during guarded simulated opening.

    Contact-loaded intermediate opening targets can stall, so retain the full
    validated opening goal. Accept only actual in-range measurements.
    """
    original=sim.client._read_raw_state
    sim.client._read_raw_state=initial_virtual_reader(original,keepalive)
    try:
        sim.client._read_raw_state()
        sim._send({'gripper':sim.client._normalize_gripper_width(width_m)},keepalive,timeout_s)
    finally:
        sim.client._read_raw_state=original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='completed logs/in-hand-simulation/seed-N directory')
    parser.add_argument('--url', default='http://127.0.0.1')
    parser.add_argument('--allow-partial', action='store_true', help='explicitly replay only the apples proven successful by a partial benchmark')
    args = parser.parse_args(argv)
    from aple.hal.gripper_physics import physics_scene, closing_gap, evaluate_outcomes
    from aple.hal.studio_simulation import StudioSimulationReplay
    from aple.hal.piper_studio import PiperWebError
    offline = json.loads((args.run/'report.json').read_text())
    if offline.get('success') is not True:
        raise ValueError('Only a fully successful offline contact run may be replayed')
    if offline.get('full_scene_complete') is False and not args.allow_partial:
        raise ValueError('A partial apple benchmark is not a full-scene web replay; run all apples first')
    cfg = yaml.safe_load((args.run/'virtual-config.yaml').read_text())
    trace = json.loads((args.run/'trace.json').read_text())
    tested = list(range(1,len(trace['apples'])+1))
    if offline.get('full_scene_complete') is False:
        tested = [item['apple'] for item in offline.get('apples', [])]
        if (not tested or len(tested) != offline.get('tested_apple_count') or len(set(tested)) != len(tested)
                or any(isinstance(i,bool) or not isinstance(i,int) or not 1 <= i <= len(trace['apples']) for i in tested)):
            raise ValueError('Invalid tested apple IDs in partial report')
    a = offline['assumptions']
    digest = hashlib.sha256(json.dumps(a,sort_keys=True).encode()).hexdigest()
    sim = StudioSimulationReplay(cfg,base_url=args.url)
    # Validate every arm target before scene changes or motion ownership.
    commands = []
    grasp = False
    for event in trace['events']:
        if event['action'] == 'grasp':
            grasp = True
        elif event['action'] == 'move':
            sim.client._normalize_joint_target(tuple(event['joints_rad']))
            speed = event['joint_velocity_rad_s']
            if not math.isfinite(speed) or speed <= 0:
                raise ValueError('Invalid replay speed')
            commands.append(('move',event['joints_rad'],speed))
        elif event['action'] == 'gripper':
            gap = closing_gap(event['width_m'],grasp=grasp,assumptions=a)
            sim.client._normalize_gripper_width(gap)
            commands.append(('gripper',gap,grasp))
            grasp = False
        elif event['action'] == 'inspect_capture':
            commands.append(('capture',event['view_index'],event['apple_index']))
    existing = sim._request('GET','/api/sim/scenes/live/objects')['objects']
    if not existing or any(o.get('aple_assumptions_sha256') != digest for o in existing):
        raise PiperWebError(f'Run studio_assumed_simd.py --assumptions {args.run / "assumptions.yaml"} first; current model assumptions differ')
    output = args.run/'studio'
    suffix = 2
    while output.exists():
        output = args.run/f'studio-{suffix}'
        suffix += 1
    output.mkdir()
    report_path = output/'report.json'
    sim.prepare(physics_scene(cfg,a),backup_path=output/'previous-scene.json')
    sim.connect_virtual_cameras(('sim:top','sim:front','sim:side'))
    def positions():
        values = {o['id']:o for o in sim._request('GET','/api/sim/scenes/live/objects')['objects']}
        fruit = [values[f'aple_apple_{i+1}'] for i in range(len(trace['apples']))]
        if any(o.get('aple_assumptions_sha256') != digest for o in fruit):
            raise PiperWebError('Virtual model changed during replay')
        return np.array([o['pos'] for o in fruit])
    initial = positions()
    peak = initial[:,2].copy()
    # Use the same measured-joint criterion as the offline contact test;
    # heavier fruit creates servo load error even after settling.
    sim.client._arrival_tolerance_rad = a['joint_arrival_tolerance_rad']
    report = dict(mode='studio_contact_replay_of_offline_color_inspection',success=False,errors=[],goals_reached=0,
                  tested_apples=tested, full_scene_complete=False,
                  offline_color_results=offline['apples'],inspections_replayed=0,
                  arrival_tolerance_rad=sim.client._arrival_tolerance_rad)
    try:
        with sim.session() as keepalive:
            sim.client._gripper_arrival_tolerance = 2
            original_reader = sim.client._read_raw_state
            sim.client._read_raw_state = initial_virtual_reader(original_reader,keepalive)
            try:
                open_virtual_gripper(sim,cfg['simulation']['in_hand_scene']['preopen_m'],keepalive)
                sim._send(sim.client._normalize_joint_target(tuple(cfg['simulation']['initial_joint_rad'])),keepalive,15)
            finally:
                sim.client._read_raw_state = original_reader
            for index,(kind,value,extra) in enumerate(commands):
                if kind == 'move':
                    start = np.array(sim.client.joint_state())
                    goal = np.array(value)
                    duration = max(.02,float(np.max(abs(goal-start)))/extra)
                    count = max(1,math.ceil(duration/.10))
                    for t in range(1,count+1):
                        begun = time.monotonic()
                        q = start+(goal-start)*t/count
                        sim._send(sim.client._normalize_joint_target(tuple(q)),keepalive,15)
                        time.sleep(max(0.,duration/count-(time.monotonic()-begun)))
                elif kind == 'gripper':
                    sim.client._gripper_arrival_tolerance = 100 if extra else 2
                    if extra:
                        sim._send({'gripper':sim.client._normalize_gripper_width(value)},keepalive,15)
                    else:
                        open_virtual_gripper(sim,value,keepalive)
                    time.sleep(a['gripper_settle_s'])
                else:
                    time.sleep(cfg['inspection']['views']['settle_time_s'])
                    report['inspections_replayed'] += 1
                peak[:] = np.maximum(peak,positions()[:,2])
                report['goals_reached'] = index+1
                if index%30 == 0:
                    print(f'Studio virtual replay {index+1}/{len(commands)}',flush=True)
            time.sleep(a['final_settle_s'])
    except Exception as exc:
        report['errors'].append(str(exc))
    finally:
        report['outcomes'] = [o for o in evaluate_outcomes(cfg,trace,a,initial,peak,positions()) if o['apple'] in tested]
        report['success'] = not report['errors'] and all(o['success'] for o in report['outcomes'])
        report['full_scene_complete'] = report['success'] and len(tested)==len(trace['apples'])
        report_path.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return 0 if report['success'] else 1


if __name__=='__main__':
    raise SystemExit(main())
