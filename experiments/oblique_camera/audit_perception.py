"""Run unchanged team M1 on rendered RGB-D, with boxes and robot occlusion."""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import cv2
import mujoco
import numpy as np

from aple.config import load_config
from aple.execution.trajectory import build_piper_kinematics
from aple.hal.gripper_physics import assumed_config, load_assumptions
from aple.hal.in_hand_simulation import ContactTransport, scene_config
from aple.hal.simulation import simulation_config
from aple.hal.simulation_optimization import optimization_config
from aple.perception import box_occupancy, detector, pose_estimator
from aple.schema import AppleDetection2D, CaptureMode, SceneSnapshot, Source
from experiments.oblique_camera.comparison import SETTINGS, camera_transform


def setup(seed):
    assumptions = load_assumptions(ROOT/'configs/simulation/gripper_physics.yaml')
    config = assumed_config(simulation_config(load_config(), ROOT/'configs/simulation/piper.yaml'), assumptions)
    config = scene_config(config, ROOT/'configs/simulation/in_hand_full_floor.yaml', seed=seed)
    config = optimization_config(config, ROOT/'configs/simulation/in_hand_vertical.yaml')
    assumptions.update(config['simulation']['in_hand_scene']['gripper_overrides'])
    assumptions['opaque_cup_radius_m'] = .04
    # This audit exercises the existing depth-only detector. It does not train
    # YOLO or claim the trained model generalizes to the alternate viewpoint.
    config['yolo']['weights'] = None
    _, fk = build_piper_kinematics(config)
    return config, ContactTransport(config, assumptions, fk)


def errors(states, truth):
    results = []
    for state in states:
        distances = np.linalg.norm(truth - state.center_m, axis=1)
        index = int(np.argmin(distances))
        results.append(dict(apple=index+1, center_error_m=float(distances[index]),
            diameter_m=state.diameter_m, confidence=state.confidence,
            top_surface_z_m=state.top_surface_z_m, depth_valid_ratio=state.depth_valid_ratio))
    return results


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    report = dict(mode='unchanged_M1_rendered_RGBD_camera_audit', seeds=SETTINGS['seeds'], cases=[])
    for seed in SETTINGS['seeds']:
        config, transport = setup(seed)
        model, data = transport.model, transport.data
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, 'top')
        h, w = config['simulation']['image_size']
        renderer = mujoco.Renderer(model, height=h, width=w)
        try:
            for tilt in SETTINGS['tilt_rad']:
                transform = camera_transform(tilt)
                config['workcell']['T_base_camera'] = transform.tolist()
                model.cam_pos[camera_id] = transform[:3, 3]
                mujoco.mju_mat2Quat(model.cam_quat[camera_id], (transform[:3, :3] @ np.diag([1., -1., -1.])).reshape(9))
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera='top')
                color = renderer.render()[..., ::-1].copy()
                renderer.enable_depth_rendering()
                depth = renderer.render().copy()
                renderer.disable_depth_rendering()
                renderer.enable_segmentation_rendering()
                segmentation = renderer.render().copy()
                renderer.disable_segmentation_rendering()
                depth[(depth <= 0) | (depth > SETTINGS['maximum_depth_m'])] = 0
                degrees = round(np.degrees(tilt))
                cv2.imwrite(str(output/f'seed-{seed}-tilt-{degrees}.png'), color)
                snapshot = SceneSnapshot(100+seed, Source.TOP, CaptureMode.FULL, color, depth,
                    tuple(config['simulation']['intrinsics']), 0.)
                truth = transport.positions()
                case = dict(seed=seed, tilt_deg=degrees, T_base_camera=transform.tolist())
                try:
                    detections = detector.detect(snapshot, config)
                    measured = pose_estimator.estimate(snapshot, detections, transform, None, config)
                    case.update(detected_count=len(detections), detected_estimates=errors(measured, truth))
                except (ValueError, RuntimeError) as exc:
                    case['detection_error'] = str(exc)
                # Separate visible-mask oracle from actual detection: diagnose
                # 3D fitting/calibration without passing this oracle into M1 detect.
                oracle = []
                for index, body in enumerate(transport.ids):
                    geom_ids = np.flatnonzero(model.geom_bodyid == body)
                    mask = np.isin(segmentation[..., 0], geom_ids) & (segmentation[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
                    if mask.any():
                        bbox = cv2.boundingRect(mask.astype(np.uint8))
                        oracle.append(AppleDetection2D(snapshot.scene_id, index+1, bbox, 1., mask))
                estimates = pose_estimator.estimate(snapshot, oracle, transform, None, config)
                case.update(visible_apple_pixels=[int(d.mask.sum()) for d in oracle],
                    oracle_mask_estimates=errors(estimates, truth))
                missing = copy.deepcopy(snapshot)
                for detection in oracle:
                    missing.depth_m[detection.mask] = 0
                fallback = pose_estimator.estimate(missing, oracle, transform, None, config)
                case['missing_depth_fallback_estimates'] = errors(fallback, truth)
                boxes = box_occupancy.estimate(snapshot, transform, config)
                case['empty_boxes'] = [dict(box=b.box_id.name, valid=b.valid,
                    false_occupied_cells=int(sum(b.occupied)), heights_m=b.heights_m) for b in boxes]
                report['cases'].append(case)
                (output/'report.json').write_text(json.dumps(report, indent=2))
                print(f"seed {seed}, tilt {degrees}: detections={case.get('detected_count')}, "
                    f"boxes={[(b['box'],b['false_occupied_cells']) for b in case['empty_boxes']]}", flush=True)
        finally:
            renderer.close()
    return report


if __name__ == '__main__':
    destination = ROOT/'logs/oblique-camera-perception'
    counter = 2
    while destination.exists():
        destination = ROOT/f'logs/oblique-camera-perception-{counter}'
        counter += 1
    run(destination)
