#!/usr/bin/env python3
"""Report mounted adapter configuration without contacting robot devices."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aple.config import load_config
from aple.execution.gripper_geometry import apply_attachment_geometry, reference_distances_m


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-dir', type=Path)
    parser.add_argument('--check', action='store_true', help='require complete, active mounted geometry')
    args = parser.parse_args()
    cfg = load_config(args.config_dir) if args.config_dir else load_config()
    for label, meters in reference_distances_m(cfg).items():
        print(f'{label}: {meters * 1000:g} mm ({meters:g} m), photo point-to-point distance')
    if not cfg.get('gripper_attachment', {}).get('installed', False):
        print('PENDING: attachment inactive; assembled TCP/contact width/envelope remain to be specified.')
        return 1 if args.check else 0
    try:
        active = apply_attachment_geometry(cfg)
    except ValueError as exc:
        print(f'INVALID: {exc}', file=sys.stderr)
        return 1
    print(f"TCP offset [m]: {active['workcell']['tcp_offset_m']}")
    print(f"Contact width [m]: {active['piper_web']['gripper_width_range_m']}")
    print('Configuration complete; hardware calibration and clearance still require verification.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
