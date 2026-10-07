"""TCP/flange contract tests with an analytic math runtime; no robot devices."""

import math
from types import SimpleNamespace

import numpy as np
import pytest

from aple.execution.trajectory import (
    PiperKinematicsFKProvider, PiperKinematicsIKSolver, PiperTCPTransform,
    ToolPose, build_piper_kinematics,
)
from aple.inspection.in_hand_provider import pose_to_matrix


class AnalyticRuntime:
    """Synthetic xyz + X rotation model, with the actual Piper batch API shape."""
    def __init__(self):
        self.geom = SimpleNamespace(tip="link6", tip_index=1)
        self.last_q = None

    def geometry(self):
        return self.geom

    def link_transforms(self, q, geom):
        assert geom is self.geom
        self.last_q = q.copy()
        T = np.eye(4)
        c, s = math.cos(q[0, 3]), math.sin(q[0, 3])
        T[:3, :3] = ((1, 0, 0), (0, c, -s), (0, s, c))
        T[:3, 3] = q[0, :3]
        return np.stack((np.eye(4), T))[None, :]

    def ik(self, target, seed, *, geom):
        assert geom is self.geom
        return {"ok": True, "q": (*target[:3, 3], math.atan2(target[2, 1], target[2, 2]), 0., 0.)}


@pytest.mark.parametrize("offset", [(0., 0., 0.), (0.015, -0.02, 0.12)])
@pytest.mark.parametrize("angle", [0., math.pi / 2, math.pi, -math.pi / 2])
def test_tcp_ik_fk_round_trip_with_offset_and_rotation(offset, angle):
    runtime = AnalyticRuntime()
    ik, fk = build_piper_kinematics({"workcell": {"tcp_offset_m": offset}}, kinematics_module=runtime)
    pose = ToolPose((0.3, -0.2, 0.4), (math.sin(angle / 2), 0., 0., math.cos(angle / 2)))

    joints = ik.solve(pose, (0.,) * 6)
    actual = fk(joints)

    assert ik.tcp_transform is fk.tcp_transform
    np.testing.assert_allclose(pose_to_matrix(actual), pose_to_matrix(pose), atol=1e-12)
    np.testing.assert_allclose(runtime.last_q, np.asarray(joints)[None, :])


def test_ik_rotates_offset_in_flange_frame_before_subtracting():
    seen = []
    ik = PiperKinematicsIKSolver(
        lambda target, seed: seen.append(target) or {"ok": True, "q": (0.,) * 6},
        tcp_transform=PiperTCPTransform((0.1, 0.02, 0.03)),
    )
    ik.solve(ToolPose((0.3, -0.2, 0.4), (0., 0., 2**-0.5, 2**-0.5)), (0.,) * 6)

    np.testing.assert_allclose(seen[0][:3, 3], (0.32, -0.3, 0.37), atol=1e-12)


@pytest.mark.parametrize("offset", [None, [], [0., 0.], [0., 0., float("nan")], [0., 0., float("inf")]])
def test_missing_or_invalid_offset_fails_before_loading_piper(offset):
    with pytest.raises(ValueError, match="tcp_offset_m"):
        build_piper_kinematics({"workcell": {"tcp_offset_m": offset}})


def test_direct_ik_constructor_does_not_silently_assume_zero_offset():
    with pytest.raises(ValueError, match="tcp_offset_m"):
        PiperKinematicsIKSolver(lambda target, seed: {"ok": True, "q": (0.,) * 6})


@pytest.mark.parametrize("R", [np.diag((1., -1., -1.)), np.diag((-1., 1., -1.)), np.diag((-1., -1., 1.))])
def test_fk_quaternion_handles_half_turns_on_each_axis(R):
    flange = np.eye(4)
    flange[:3, :3] = R
    fk = PiperKinematicsFKProvider(PiperTCPTransform((0., 0., 0.1)),
                                  link_transforms=lambda q: flange[None, None, :], tip_index=0)
    np.testing.assert_allclose(pose_to_matrix(fk((0.,) * 6)),
                               flange @ fk.tcp_transform.matrix(), atol=1e-12)


@pytest.mark.parametrize("flange", [np.full((4, 4), np.nan), np.zeros((4, 4)), np.diag((-1., 1., 1., 1.))])
def test_fk_rejects_bad_geometry_transform(flange):
    fk = PiperKinematicsFKProvider(PiperTCPTransform((0., 0., 0.1)),
                                  link_transforms=lambda q: flange[None, None, :], tip_index=0)
    with pytest.raises(RuntimeError, match="invalid rigid"):
        fk((0.,) * 6)


def test_fk_rejects_unbatched_result_and_bad_joint_units_shape():
    fk = PiperKinematicsFKProvider(PiperTCPTransform((0., 0., 0.1)),
                                  link_transforms=lambda q: np.eye(4), tip_index=0)
    with pytest.raises(RuntimeError, match="links"):
        fk((0.,) * 6)
    with pytest.raises(ValueError, match="six"):
        fk((0.,) * 5)


@pytest.mark.parametrize("configured_offset,fk_offset,message", [
    (None, (0., 0., 0.1), "tcp_offset_m"),
    ((0., 0., 0.2), (0., 0., 0.1), "match workcell"),
    ((0., 0., 0.1), (0., 0., 0.2), "same measured"),
])
def test_motion_builder_blocks_missing_or_mismatched_offsets(configured_offset, fk_offset, message):
    from aple.hal.piper_motion import build_piper_motion_adapter

    runtime = AnalyticRuntime()
    ik, _ = build_piper_kinematics({"workcell": {"tcp_offset_m": (0., 0., 0.1)}}, kinematics_module=runtime)
    fk = PiperKinematicsFKProvider(PiperTCPTransform(fk_offset),
                                  link_transforms=lambda q: runtime.link_transforms(q, runtime.geom), tip_index=1)
    with pytest.raises(ValueError, match=message):
        build_piper_motion_adapter(None, {"workcell": {"tcp_offset_m": configured_offset}},
                                   ik, None, tool_pose_provider=fk)


def test_motion_builder_wires_the_same_tcp_fk_callable():
    from aple.config import load_config
    from aple.hal.piper_motion import build_piper_motion_adapter
    from tests.test_trajectory import SafeSegments

    cfg = load_config()
    cfg["workcell"]["tcp_offset_m"] = [0., 0., 0.1]
    cfg["safety"]["joint_limits_rad"] = [(-math.pi, math.pi)] * 6
    ik, fk = build_piper_kinematics(cfg, kinematics_module=AnalyticRuntime())
    studio = SimpleNamespace(joint_state=lambda: (0.,) * 6)
    adapter = build_piper_motion_adapter(studio, cfg, ik, SafeSegments(), tool_pose_provider=fk)

    assert adapter._transport._tool_pose_provider is fk
    np.testing.assert_allclose(adapter._transport.tool_pose().position_m, (0., 0., 0.1))
