"""Fresh depth evidence for a lifted apple, independent of command acknowledgments."""
import time
import numpy as np
from aple.inspection.inhand import apple_center_cam, check_slip


class VisualHoldingVerifier:
    def __init__(self, cfg, capture_provider, joint_state):
        self.cfg = cfg
        self.provider = capture_provider
        self.joint_state = joint_state

    def __call__(self, apple, apple_in_tcp, *, timeout_s):
        self.provider.begin(apple, apple_in_tcp)
        deadline = time.monotonic() + timeout_s
        capture = self.provider(apple.apple_id, 0, self.joint_state(), timeout_s=timeout_s)
        if capture is None or time.monotonic() > deadline:
            return False
        snap = capture.snapshot
        center = apple_center_cam(np.asarray(self.cfg['workcell']['T_base_camera']),
                                  np.asarray(snap.tool_pose), apple_in_tcp.position_m)
        check = check_slip(snap.depth_m, capture.mask, snap.intrinsics, center, apple.diameter_m/2,
                          **self.cfg['inspection']['slip_check'])
        return bool(check.ok)
