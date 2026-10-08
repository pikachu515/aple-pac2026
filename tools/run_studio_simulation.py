"""Plan with APLE and replay the trajectory in Studio's sim_follower1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aple.config import load_config
from aple.hal.simulation import simulation_config
from aple.hal.studio_simulation import StudioSimulationReplay, scene_from_config
from aple.hal.piper_studio import PiperWebError
from tools.run_cycle import main as run_cycle


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1')
    parser.add_argument('--prepare-only', action='store_true', help='load scene without moving arm')
    parser.add_argument('--step-delay', type=float, default=.1, help='minimum pause between replay goals, seconds')
    parser.add_argument('--output', type=Path, default=ROOT / 'logs/studio-simulation')
    args = parser.parse_args(argv)
    cfg = simulation_config(load_config(), ROOT / 'configs/simulation/piper.yaml')
    args.output.mkdir(parents=True, exist_ok=True)
    trace = args.output / 'planned-trace.json'
    # Compute and validate the complete APLE cycle before contacting Studio.
    if not args.prepare_only:
        run_cycle(['--simulate', '--trace', str(trace)])
    client = StudioSimulationReplay(cfg, base_url=args.url)
    sid = client.prepare(scene_from_config(cfg), backup_path=args.output / 'previous-scene.json')
    client.connect_virtual_cameras()
    report = dict(mode='planned_trajectory_replay', scene_id=sid, iface='sim_follower1',
                  physical_grasp_verified=False, goals_reached=0, status='prepared')
    print(f'Studio 장면 준비 완료: {sid}', flush=True)
    print(f'브라우저에서 {args.url.rstrip("/")}/scene 을 열어 가상 카메라를 보세요.', flush=True)
    print('계획 궤적 미리보기입니다. Studio 기본 핑거는 가상 TCP와 다르며 물리 파지 성공을 판정하지 않습니다.', flush=True)
    try:
        if not args.prepare_only:
            data = json.loads(trace.read_text())
            report['status'] = 'running'
            def progress(i, n):
                report['goals_reached'] = i
                print(f'Studio replay {i}/{n}', flush=True)
            report['goals_reached'] = client.replay(data['events'], minimum_step_s=args.step_delay,
                on_progress=progress)
            report['status'] = 'completed'
            print('Studio 궤적 재생 완료. 실제 파지 검증은 별도로 필요합니다.', flush=True)
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = str(exc)
        raise
    finally:
        (args.output / 'studio-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    try:
        main()
    except (PiperWebError, ValueError) as exc:
        raise SystemExit(f'Studio 시뮬레이션 실패: {exc}') from exc
