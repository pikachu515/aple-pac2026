"""Reject unsafe intermediate states even when both endpoints are safe."""
from types import SimpleNamespace

import numpy as np
import pytest

from aple.execution.goal_builder import Pose3D
from aple.execution.segment_validator import PiperSegmentValidator
from tests.test_piper_motion import make_adapter


def validator(*, obstacles=(), floor=0, workspace=None, fk=None, links=None):
    cfg = {'safety': {'joint_limits_rad': [[-1, 1]] * 6,
                     'workspace_aabb_m': workspace or [[-2, 2], [-2, 2], [0, 2]],
                     'segment_validation': {'sample_step_rad': 0.05, 'floor_z_m': floor,
                                            'clearance_m': 0, 'obstacle_aabbs_m': list(obstacles)}}}
    def transforms(q):
        tf = np.tile(np.eye(4), (len(q), 2, 1, 1))
        tf[:, 1, 0, 3] = q[:, 0]
        tf[:, 1, 2, 3] = 0.5
        return tf
    return PiperSegmentValidator(cfg, fk=fk or (lambda q: Pose3D((q[0], 0, .5), (0, 0, 0, 1))),
                                 link_transforms=links or transforms,
                                 geometry=SimpleNamespace(names=('base', 'tool'), radius=.01,
                                                          by_link=[(1, np.array([[0., 0, 0]]))]))


def q(x):
    return (x, 0, 0, 0, 0, 0)


def test_obstacle_in_middle_of_segment_blocks_but_endpoints_are_safe():
    v = validator(obstacles=[[[-.02, .02], [-.02, .02], [.48, .52]]])
    assert v.is_segment_safe(q(-.5), q(-.5))
    assert v.is_segment_safe(q(.5), q(.5))
    assert not v.is_segment_safe(q(-.5), q(.5))


def test_nonlinear_tcp_path_leaving_workspace_is_blocked():
    v = validator(workspace=[[-2, 2], [-2, 2], [0, .8]],
                  fk=lambda q: Pose3D((q[0], 0, 1 - 2 * q[0] ** 2), (0, 0, 0, 1)))
    assert v.is_segment_safe(q(-.5), q(-.5))
    assert not v.is_segment_safe(q(-.5), q(.5))


def test_floor_checks_geometry_radius_and_joint_limits():
    v = validator(floor=.495)
    assert not v.is_segment_safe(q(0), q(0))
    assert not validator().is_segment_safe(q(0), q(1.01))
    assert not validator().is_segment_safe(q(0), q(1e100))


def test_malformed_geometry_does_not_pass():
    with pytest.raises(RuntimeError, match='transforms'):
        validator(links=lambda q: np.zeros((1, 4, 4))).is_segment_safe(q(0), q(.1))
    assert not validator(fk=lambda q: Pose3D((float('nan'), 0, .5), (0, 0, 0, 1))).is_segment_safe(q(0), q(0))


def test_named_moves_are_subdivided_and_checked_before_publishing():
    adapter, transport, guard = make_adapter()
    assert adapter.move_named('scan', q(.4), speed_scale=.2, timeout_s=1)
    moves = [c for c in transport.calls if c[0] == 'move']
    assert len(moves) == 4 and len(guard.calls) == 4
    adapter, transport, _ = make_adapter(safe=False)
    assert not adapter.move_named('scan', q(.4), speed_scale=.2, timeout_s=1)
    assert transport.calls == []
