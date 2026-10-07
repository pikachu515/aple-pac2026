"""Photo units, mounted coordinate conversion and added tool collisions."""
import copy
from types import SimpleNamespace

import numpy as np
import pytest

from aple.config import load_config
from aple.execution.gripper_geometry import apply_attachment_geometry, reference_distances_m
from aple.execution.segment_validator import PiperSegmentValidator
from aple.execution.goal_builder import Pose3D


def mounted_config():
    # Fictional mounted measurements for tests, not values for the user's tool.
    cfg = load_config()
    cfg['gripper_attachment']['installed'] = True
    cfg['gripper_attachment']['assembled'] = {
        'tcp_axes_match_flange': True, 'tcp_offset_mm': [1, 2, 160],
        'contact_width_range_mm': [20, 120], 'finger_thickness_mm': 10,
        'finger_width_mm': 30, 'grasp_height_offset_mm': -5,
        'collision_spheres_flange_mm': [{'center_mm': [0, 0, 160], 'radius_mm': 20}],
    }
    return cfg


def test_photo_distances_are_mm_and_do_not_override_runtime():
    cfg = load_config()
    assert reference_distances_m(cfg) == pytest.approx({'D1': .048144, 'D2': .048464, 'D3': .07176, 'D4': .07976})
    assert apply_attachment_geometry(cfg) is cfg
    assert cfg['gripper_attachment']['reference']['nominal_base_opening_range_mm'] == [0, 100]


def test_mounted_geometry_converts_once_without_mutating_source():
    cfg = mounted_config()
    before = copy.deepcopy(cfg)
    active = apply_attachment_geometry(cfg)
    assert cfg == before
    assert active['workcell']['tcp_offset_m'] == pytest.approx([.001, .002, .16])
    assert active['piper_web']['gripper_width_range_m'] == pytest.approx([.02, .12])
    assert active['grasp_planner']['finger_thickness_m'] == .01
    assert active['grasp_planner']['finger_width_m'] == .03
    assert active['size_bins']['gripper_range_mm'] == [30, 100]
    assert all(b['grasp_height_offset_mm'] == -5 for b in active['size_bins']['bins'])
    assert apply_attachment_geometry(active) == active


@pytest.mark.parametrize('key,value', [
    ('tcp_axes_match_flange', False), ('tcp_offset_mm', None),
    ('contact_width_range_mm', [100, 10]), ('finger_thickness_mm', -1),
    ('finger_width_mm', float('nan')), ('grasp_height_offset_mm', True),
    ('collision_spheres_flange_mm', []),
])
def test_active_unknown_or_invalid_geometry_fails_closed(key, value):
    cfg = mounted_config()
    cfg['gripper_attachment']['assembled'][key] = value
    with pytest.raises(ValueError):
        apply_attachment_geometry(cfg)


def test_added_adapter_collision_is_transformed_from_flange():
    cfg = mounted_config()
    cfg['safety'] = {'joint_limits_rad': [[-1, 1]] * 6,
                     'workspace_aabb_m': [[-2, 2], [-2, 2], [0, 2]],
                     'segment_validation': {'sample_step_rad': .05, 'floor_z_m': 0,
                                            'clearance_m': 0, 'obstacle_aabbs_m': []}}
    def links(q):
        transforms = np.tile(np.eye(4), (len(q), 2, 1, 1))
        transforms[:, 1, 2, 3] = .5
        return transforms
    def validator():
        return PiperSegmentValidator(cfg, fk=lambda q: Pose3D((0, 0, .66), (0, 0, 0, 1)),
            link_transforms=links, geometry=SimpleNamespace(names=('base', 'link6'), tip_index=1,
                radius=.01, by_link=[(1, np.array([[0., 0, 0]]))]))
    q = [0] * 6
    assert validator().is_segment_safe(q, q)
    # Original flange mesh at z=.5 is clear; mounted tool at z=.66 hits this box.
    cfg['safety']['segment_validation']['obstacle_aabbs_m'] = [[[-.01, .01], [-.01, .01], [.65, .67]]]
    assert not validator().is_segment_safe(q, q)
    cfg['gripper_attachment']['installed'] = False
    assert validator().is_segment_safe(q, q)
    cfg['gripper_attachment']['installed'] = True
    cfg['safety']['segment_validation']['obstacle_aabbs_m'] = []
    cfg['gripper_attachment']['assembled']['collision_spheres_flange_mm'][0]['center_mm'][2] = -500
    assert not validator().is_segment_safe(q, q)


def test_mounted_geometry_reaches_hal_and_tcp_consumers_without_devices():
    from aple.hal.piper_studio import PiperStudio
    from aple.execution.trajectory import PiperTCPTransform
    from aple.perception.grasp_planner import grasp_params
    cfg = mounted_config()
    studio = PiperStudio(cfg)
    assert studio._normalize_gripper_width(.02) == pytest.approx(0)
    assert studio._normalize_gripper_width(.07) == pytest.approx(50)
    assert studio._normalize_gripper_width(.12) == pytest.approx(100)
    with pytest.raises(ValueError, match='outside'):
        studio._normalize_gripper_width(.121)
    tcp = PiperTCPTransform.from_config(cfg)
    assert tcp.matrix()[:3, 3] == pytest.approx([.001, .002, .16])
    params = grasp_params(apply_attachment_geometry(cfg))
    assert params['finger_thickness_m'] == .01
    assert params['finger_width_m'] == .03
