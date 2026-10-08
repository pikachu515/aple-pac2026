"""Launch the installed Studio simd with an assumed finger contact model.

Only this process is extended. Installed Studio files are never modified.
"""
from pathlib import Path
import argparse
import hashlib
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assumptions', type=Path, default=ROOT/'configs/simulation/gripper_physics.yaml')
    parser.add_argument('--daemon', type=Path, default=Path.home()/'piper-web-deploy/current/daemons/simd.py')
    args = parser.parse_args()
    from aple.hal.gripper_physics import load_assumptions, build_assumed_model
    assumptions = load_assumptions(args.assumptions)
    digest = hashlib.sha256(json.dumps(assumptions, sort_keys=True).encode()).hexdigest()
    import piper_sim.world as world
    import piper_sim.bridge as bridge
    import piper_sim.cameras as cameras
    import mujoco
    world.load_model = lambda spec: build_assumed_model(spec, assumptions)
    cameras.CAMERAS = (*cameras.CAMERAS, 'side')
    original_objects = world.World.objects

    def objects(self):
        result = original_objects(self)
        with self._lock:
            valid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'aple_shaft_finger_l') >= 0
        if valid:
            for obj in result:
                obj['aple_assumptions_sha256'] = digest
        return result

    world.World.objects = objects
    # Studio 0.5.12 can observe a newly-created, not-yet-sized action segment.
    # Translate only this known transient failure to the bridge's retry type.
    original_reader = bridge.ActionReader

    class RetryingActionReader(original_reader):
        def __init__(self, *args, **kwargs):
            try:
                super().__init__(*args, **kwargs)
            except ValueError as exc:
                if str(exc) != 'cannot mmap an empty file':
                    raise
                raise bridge.ArmSegmentError(str(exc)) from exc

    bridge.ActionReader = RetryingActionReader
    runpy.run_path(str(args.daemon), run_name='__main__')


if __name__ == '__main__':
    main()
