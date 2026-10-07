"""tools/rviz_view.py — ROS 없이 확인할 수 있는 부분 (장면 구성·요약·회전 변환)."""

from __future__ import annotations

import numpy as np

from aple.hal.calibration import rpy_to_matrix
from tools import rviz_view


def test_demo_view_matches_truth():
    view = rviz_view.main(["--demo", "--no-ros"])
    assert len(view.apples) == 5 and len(view.boxes) == 2 and len(view.calib) == 8
    for a in view.apples:
        c, r = min(view.truth, key=lambda t: np.linalg.norm(np.subtract(t[0], a.center_m)))
        assert np.linalg.norm(np.subtract(c, a.center_m)) < 0.003 and abs(a.diameter_m - 2 * r) < 0.003
    upper = next(b for b in view.boxes if b.box_id.name == "UPPER")
    assert upper.occupied == [True, False, False, False, True, False]
    assert max(c["err"] for c in view.calib) < 0.004
    assert len(view.cloud_cam) == len(view.cloud_bgr) > 10000


def test_quaternion_from_rotation():
    for angles in [(0, 0, 0), (np.pi, 0.05, -np.pi / 2), (0.3, -1.2, 2.5), (np.pi, 0, 0)]:
        R = rpy_to_matrix(*angles)
        x, y, z, w = rviz_view._quat(R)
        q = np.array([w, x, y, z])
        assert np.isclose(np.linalg.norm(q), 1.0)
        w, x, y, z = q
        R2 = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                       [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                       [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        assert np.allclose(R, R2, atol=1e-9)
