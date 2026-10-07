"""Offline MuJoCo contact test of assumed adapters; no robot/Studio connection."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aple.config import load_config
from aple.hal.simulation import simulation_config
from aple.hal.gripper_physics import load_assumptions, run_physics, assumed_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, default=ROOT/'logs/studio-gripper-physics/planned-trace.json')
    parser.add_argument('--assumptions', type=Path, default=ROOT/'configs/simulation/gripper_physics.yaml')
    parser.add_argument('--report', type=Path, default=ROOT/'logs/gripper-physics/report.json')
    args=parser.parse_args()
    cfg=simulation_config(load_config(), ROOT/'configs/simulation/piper.yaml')
    assumptions=load_assumptions(args.assumptions)
    cfg=assumed_config(cfg, assumptions)
    report=run_physics(cfg, json.loads(args.trace.read_text()), assumptions)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0 if report['success'] else 1


if __name__=='__main__':
    raise SystemExit(main())
