"""Calibrated gripper feedback and slip detection tests."""

import pytest

from aple.execution.gripper import (
    GraspStateCode,
    GripperForceController,
    GripperTelemetry,
)
from aple.schema import GraspProfile


@pytest.fixture
def cfg():
    return {
        "execution": {"gripper": {"slip_width_m": 0.002}},
        "size_bins": {"force": {"f_bruise_n": 30.0}},
    }


def profile():
    return GraspProfile(1, 2, 1, 0.2, 0.09, 0.0, 10.0, 1000, True)


def test_controller_estimates_force_and_enters_holding(cfg):
    controller = GripperForceController(cfg)

    feedback = controller.observe(profile(), GripperTelemetry(0.06, 0.4, 12.0))

    assert feedback.force_est_n == pytest.approx(12.0)
    assert feedback.state == GraspStateCode.HOLDING
    assert not feedback.slip_flag


def test_controller_latches_slip_on_width_increase(cfg):
    controller = GripperForceController(cfg)
    controller.observe(profile(), GripperTelemetry(0.06, 0.4, 12.0))

    feedback = controller.observe(profile(), GripperTelemetry(0.063, 0.4, 12.0))
    later = controller.observe(profile(), GripperTelemetry(0.06, 0.4, 12.0))

    assert feedback.state == GraspStateCode.SLIP
    assert later.state == GraspStateCode.SLIP
    assert later.slip_flag


def test_controller_reports_slip_when_force_drops_below_target(cfg):
    controller = GripperForceController(cfg)
    controller.observe(profile(), GripperTelemetry(0.06, 0.4, 12.0))

    feedback = controller.observe(profile(), GripperTelemetry(0.06, 0.2, 8.0))

    assert feedback.force_est_n == pytest.approx(8.0)
    assert feedback.state == GraspStateCode.SLIP


def test_controller_rejects_non_finite_force_feedback(cfg):
    with pytest.raises(ValueError, match="finite"):
        GripperForceController(cfg).observe(
            profile(), GripperTelemetry(0.06, 0.4, float("nan"))
        )


def test_controller_refuses_force_at_bruise_limit(cfg):
    with pytest.raises(ValueError, match="bruise limit"):
        GripperForceController(cfg).observe(
            profile(), GripperTelemetry(0.06, 1.0, 30.0)
        )
