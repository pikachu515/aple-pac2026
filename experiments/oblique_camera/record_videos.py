"""Record four calibrated camera angles from one shared contact-physics run."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import mujoco
import numpy as np

from experiments.oblique_camera import comparison, run


class AngleRecorder:
    def __init__(self, output):
        self.output = output
        self.renderer = None
        self.writers = {}
        self.frames = 0
        self.transport = None

    def frame(self, model, data):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(model, height=480, width=640)
            for tilt in comparison.SETTINGS['tilt_rad']:
                degrees = round(np.degrees(tilt))
                writer = cv2.VideoWriter(str(self.output/f'camera-{degrees:02d}deg.mp4'),
                    cv2.VideoWriter_fourcc(*'mp4v'), 3., (640, 480))
                self.writers[degrees] = writer
                if not writer.isOpened():
                    raise RuntimeError(f'Cannot write angle {degrees} video')
        camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'top')
        saved_position = data.cam_xpos[camera].copy()
        saved_rotation = data.cam_xmat[camera].copy()
        try:
            for tilt in comparison.SETTINGS['tilt_rad']:
                degrees = round(np.degrees(tilt))
                transform = comparison.camera_transform(tilt)
                # Only renderer camera state changes. Joint/fruit state, time,
                # contacts, and camera configuration used by M1/M2 stay intact.
                data.cam_xpos[camera] = transform[:3, 3]
                data.cam_xmat[camera] = (transform[:3, :3] @ np.diag([1., -1., -1.])).reshape(9)
                self.renderer.update_scene(data, camera='top')
                image = self.renderer.render()[..., ::-1].copy()
                state = 'INSPECT' if self.transport.rotating else 'CARRY' if self.transport.holding else 'EMPTY'
                cv2.putText(image, f'{degrees} deg from vertical | {data.time:.1f}s | {state}',
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(image, f'{degrees} deg from vertical | {data.time:.1f}s | {state}',
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
                self.writers[degrees].write(image)
        finally:
            data.cam_xpos[camera] = saved_position
            data.cam_xmat[camera] = saved_rotation
        self.frames += 1
        if self.frames % 30 == 0:
            print(f'4개 각도 녹화: 가상 시간 {data.time:.1f}s', flush=True)

    def close(self):
        for writer in self.writers.values():
            writer.release()
        if self.renderer is not None:
            self.renderer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    base = ROOT/'logs/oblique-camera-videos'/f'seed-{args.seed}'
    output = base
    counter = 2
    while output.exists():
        output = base.with_name(f'{base.name}-{counter}')
        counter += 1
    output.mkdir(parents=True)
    recorder = AngleRecorder(output)
    original = comparison.CameraComparisonTransport

    class RecordingTransport(original):
        def __init__(self, cfg, assumptions, fk, **kwargs):
            recorder.transport = self
            cfg['simulation']['in_hand_scene']['video_fps'] = 3.
            kwargs['sample_callback'] = recorder.frame
            super().__init__(cfg, assumptions, fk, **kwargs)

    comparison.CameraComparisonTransport = RecordingTransport
    print(f'각도별 영상 폴더: {output}', flush=True)
    try:
        result = run.main(['--quick', '--seed', str(args.seed), '--optimize',
            '--scene-config', str(ROOT/'configs/simulation/in_hand_full_floor.yaml'),
            '--output', str(output/'physical-run')])
    finally:
        comparison.CameraComparisonTransport = original
        recorder.close()
    reports = list((output/'physical-run').glob('seed-*/report.json'))
    physical = json.loads(reports[0].read_text()) if reports else {}
    metadata = dict(success=physical.get('success', False), frames_each=recorder.frames, fps=3.,
        duration_s=recorder.frames/3., angles_deg=sorted(recorder.writers),
        physical_report=str(reports[0]) if reports else None,
        description='Four simultaneous views of one simulated contact-physics execution; no hardware.')
    (output/'video-report.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2), flush=True)
    return result


if __name__ == '__main__':
    raise SystemExit(main())
