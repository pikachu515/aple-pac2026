"""Piper Web v0.5.9 adapter tests using a local fake HTTP server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest

from aple.execution.goal_builder import Pose3D
from aple.hal.piper_studio import PiperStudio, PiperStudioMotionTransport, PiperWebError
from aple.inspection.in_hand_capture import InHandCapture
from aple.schema import (
    ArmCommand,
    CaptureMode,
    LightState,
    LightingStatus,
    SceneSnapshot,
    Source,
)


class FakePiperWeb:
    def __init__(self):
        self.calls = []
        self.gripper_torque = {"iface": "follower0", "effort_nm": 1.0}
        self.gripper_torque_status = 200
        self.torque_after_goal = None
        self.auth_enabled = True
        self.login_ok = True
        self.goal_status = 200
        self.estop_bus_available = True
        self.estop_heartbeat_status = 200
        self.estop_heartbeat_count = 0
        self.estop_heartbeat_fail_after = None
        self.estop_timeout_ms = 2500
        self.estop_heartbeat_interval_ms = 500
        self.auto_reach_goals = False
        self.camera_preview = b"\xff\xd8test-jpeg\xff\xd9"
        self.camera_light = {"top": {"brightness": 120.0}}
        self.raw = {
            "joint1": 0,
            "joint2": 90000,
            "joint3": -85000,
            "joint4": 0,
            "joint5": 0,
            "joint6": 0,
            "gripper": 0,
        }
        self._lock = threading.Lock()

    def handler(self):
        state = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def _respond(self, code, body, *, cookie=None):
                payload = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                if cookie:
                    self.send_header("Set-Cookie", cookie)
                self.end_headers()
                self.wfile.write(payload)

            def _body(self):
                length = int(self.headers.get("Content-Length", "0"))
                return json.loads(self.rfile.read(length) or b"{}")

            def do_GET(self):
                with state._lock:
                    state.calls.append(("GET", self.path, None))
                    if self.path == "/api/auth/status":
                        self._respond(200, {
                            "enabled": state.auth_enabled,
                            "authenticated": False,
                        })
                    elif self.path == "/api/robots/gripper-effort?iface=follower0":
                        self._respond(state.gripper_torque_status, state.gripper_torque)
                    elif self.path == "/api/estop/status":
                        self._respond(200, {
                            "bus_available": state.estop_bus_available,
                            "timeout_ms": state.estop_timeout_ms,
                            "heartbeat_interval_ms": state.estop_heartbeat_interval_ms,
                        })
                    elif self.path == "/api/robots/joints/raw/follower0":
                        if state.auth_enabled and self.headers.get("Cookie") != "piper_session=ok":
                            self._respond(401, {"detail": "login required"})
                        else:
                            self._respond(200, dict(state.raw))
                    elif self.path == "/api/cameras/top%3Acamera/preview":
                        self.send_response(200)
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(state.camera_preview)))
                        self.end_headers()
                        self.wfile.write(state.camera_preview)
                    elif self.path == "/api/cameras/light":
                        self._respond(200, state.camera_light)
                    else:
                        self._respond(404, {"detail": "not found"})

            def do_POST(self):
                body = self._body()
                with state._lock:
                    state.calls.append(("POST", self.path, body))
                    if self.path == "/api/auth/login":
                        if body.get("password") == "test-password" and state.login_ok:
                            self._respond(200, {"authenticated": True}, cookie="piper_session=ok; Path=/")
                        else:
                            self._respond(401, {"detail": "invalid password"})
                    elif self.path == "/api/estop/heartbeat":
                        state.estop_heartbeat_count += 1
                        if (
                            state.estop_heartbeat_status != 200
                            or (
                                state.estop_heartbeat_fail_after is not None
                                and state.estop_heartbeat_count > state.estop_heartbeat_fail_after
                            )
                        ):
                            self._respond(state.estop_heartbeat_status, {"detail": "heartbeat failed"})
                        else:
                            self._respond(200, {"status": "ok"})
                    elif state.auth_enabled and self.headers.get("Cookie") != "piper_session=ok":
                        self._respond(401, {"detail": "login required"})
                    elif self.path == "/api/robots/jog/start":
                        self._respond(200, {
                            "status": "started",
                            "running": True,
                            "iface": body.get("iface"),
                        })
                    elif self.path == "/api/robots/jog/goal":
                        if state.goal_status != 200:
                            self._respond(state.goal_status, {"detail": "goal rejected"})
                        else:
                            if state.auto_reach_goals:
                                limits = {
                                    "joint1": (-150, 150),
                                    "joint2": (0, 180),
                                    "joint3": (-170, 0),
                                    "joint4": (-100, 100),
                                    "joint5": (-65, 65),
                                    "joint6": (-120, 120),
                                }
                                for name, (lower, upper) in limits.items():
                                    normalized = body["values"][name]
                                    degrees = lower + (normalized + 100) * (upper - lower) / 200
                                    state.raw[name] = round(degrees * 1000)
                            if "gripper" in body["values"]:
                                state.raw["gripper"] = round(body["values"]["gripper"] * 680.0)
                            if state.torque_after_goal is not None:
                                state.gripper_torque["effort_nm"] = state.torque_after_goal
                            self._respond(200, {"status": "ok", "goal": body["values"]})
                    elif self.path == "/api/robots/jog/stop":
                        self._respond(200, {"status": "stopped"})
                    else:
                        self._respond(404, {"detail": "not found"})

        return Handler


class FakeMotionStudio:
    def __init__(self):
        self.calls = []
        self.joints = (0.0,) * 6

    def joint_state(self):
        return self.joints

    def send_joint_goal(self, position_rad, timeout_s):
        self.calls.append(("goal", position_rad, timeout_s))
        self.joints = position_rad

    def send_joint_trajectory(self, waypoints, timeout_s):
        self.calls.append(("trajectory", waypoints, timeout_s))
        self.joints = waypoints[-1]

    def send_timed_joint_trajectory(self, goals, limits, *, speed_scale, timeout_s, gripper_width_watch=None):
        self.calls.append(("timed", goals, timeout_s))
        self.joints = goals[-1].position_rad

    def send(self, command, *, timeout_s=None):
        self.calls.append(("send", command, timeout_s))
        self.joints = command.position_rad


@pytest.fixture
def piper_web(monkeypatch):
    fake = FakePiperWeb()
    server = ThreadingHTTPServer(("127.0.0.1", 0), fake.handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("PIPER_WEB_PASSWORD", "test-password")
    cfg = {
        "execution": {"gripper": {"fixed_effort": 1000}},
        "piper_web": {
            "enabled": True,
            "base_url": f"http://127.0.0.1:{server.server_port}",
            "iface": "follower0",
            "request_timeout_s": 0.5,
            "goal_timeout_s": 0.5,
            "poll_interval_s": 0.01,
            "arrival_tolerance_rad": 0.001,
            "gripper_arrival_tolerance": 1.0,
            "camera_id": "top:camera",
            "robotd_safety_confirmed": True,
            "joint_limits_deg": {
                "joint1": [-150, 150],
                "joint2": [0, 180],
                "joint3": [-170, 0],
                "joint4": [-100, 100],
                "joint5": [-65, 65],
                "joint6": [-120, 120],
            },
            "gripper_width_range_m": [0.0, 0.08],
        },
    }
    try:
        yield fake, PiperStudio(cfg)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def test_joint_state_reads_v059_raw_millidegrees(piper_web):
    fake, adapter = piper_web

    result = adapter.joint_state()

    assert result == pytest.approx((0.0, 1.5707963, -1.4835299, 0.0, 0.0, 0.0))
    assert fake.calls[0][:2] == ("GET", "/api/auth/status")
    assert fake.calls[1][:2] == ("POST", "/api/auth/login")


def test_gripper_width_reads_the_raw_encoder_through_the_measured_range(piper_web):
    fake, adapter = piper_web
    fake.raw["gripper"] = 34000                      # 68000 = 열림 100 %

    assert adapter.gripper_width_m() == pytest.approx(0.04)


def test_gripper_width_requires_the_measured_range(piper_web):
    _, adapter = piper_web
    adapter.cfg["gripper_width_range_m"] = None

    with pytest.raises(PiperWebError, match="gripper_width_range_m"):
        adapter.gripper_width_m()


def test_camera_preview_returns_the_jpeg_bytes_without_scene_claims(piper_web):
    fake, adapter = piper_web

    jpeg = adapter.camera_preview_jpeg()

    assert jpeg == fake.camera_preview
    assert ("GET", "/api/cameras/top%3Acamera/preview", None) in fake.calls


def test_camera_preview_requires_a_configured_camera_id(piper_web):
    _, adapter = piper_web
    adapter.cfg["camera_id"] = None

    with pytest.raises(PiperWebError, match="piper_web.camera_id"):
        adapter.camera_preview_jpeg()


def test_camera_light_metrics_returns_raw_payload(piper_web):
    fake, adapter = piper_web

    assert adapter.camera_light_metrics() == fake.camera_light
    assert ("GET", "/api/cameras/light", None) in fake.calls


def test_snapshot_uses_an_injected_aligned_frame_provider(piper_web):
    _, adapter = piper_web
    expected = SceneSnapshot(
        1,
        Source.TOP,
        CaptureMode.FULL,
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2), dtype=np.float32),
        (1.0, 1.0, 0.5, 0.5),
        1.0,
    )
    configured = PiperStudio(
        {"piper_web": dict(adapter.cfg)},
        snapshot_provider=lambda: expected,
    )

    assert configured.snapshot() is expected


def test_snapshot_fails_explicitly_when_no_depth_provider_is_configured(piper_web):
    _, adapter = piper_web

    with pytest.raises(PiperWebError, match="aligned depth and camera intrinsics"):
        adapter.snapshot()


def test_lighting_status_requires_a_calibration_aware_provider(piper_web):
    _, adapter = piper_web
    expected = LightingStatus(LightState.OK, 0.5, True)
    configured = PiperStudio(
        {"piper_web": dict(adapter.cfg)},
        lighting_status_provider=lambda: expected,
    )

    assert configured.lighting_status() is expected


def test_lighting_status_does_not_coerce_raw_metrics(piper_web):
    _, adapter = piper_web

    with pytest.raises(PiperWebError, match="calibration-aware"):
        adapter.lighting_status()


def test_motion_transport_lower_speed_takes_more_time():
    from aple.execution.timed_motion import MotionLimits
    limits = MotionLimits((1.,)*6, (2.,)*6, .02, .1, .1)
    durations = []
    for scale in (1., .5):
        studio = FakeMotionStudio()
        transport = PiperStudioMotionTransport(studio, max_joint_step_rad=.2, motion_limits=limits)
        transport.move_joint_goal((.6, 0, 0, 0, 0, 0), speed_scale=scale, timeout_s=5.)
        kind, goals, _ = studio.calls[0]
        assert kind == 'timed'
        assert goals[-1].position_rad == pytest.approx((.6, 0, 0, 0, 0, 0))
        durations.append(goals[-1].time_s)
    assert durations[1] > durations[0]


def test_motion_transport_without_measured_limits_refuses_before_command():
    studio = FakeMotionStudio()
    transport = PiperStudioMotionTransport(studio, max_joint_step_rad=.2)
    with pytest.raises(PiperWebError, match='measured motion limits'):
        transport.move_joint_goal((.1, 0, 0, 0, 0, 0), speed_scale=1., timeout_s=1.)
    assert studio.calls == []


class _CaptureProvider:
    def __init__(self, tool_pose_fn):
        self._tool_pose_fn = tool_pose_fn

    def __call__(self, *args, **kwargs):
        return None


def _fk_warnings(monkeypatch, provider, fk):
    from aple.hal import piper_studio

    warnings = []
    monkeypatch.setattr(piper_studio._LOG, "warning", lambda msg, *a, **k: warnings.append(msg))
    PiperStudioMotionTransport(
        FakeMotionStudio(), max_joint_step_rad=0.2,
        tool_pose_provider=fk, in_hand_capture_provider=provider,
    )
    return warnings


def test_motion_transport_warns_when_capture_and_tool_pose_use_different_fk(monkeypatch):
    def fk_a(joints):
        return None

    def fk_b(joints):
        return None

    assert _fk_warnings(monkeypatch, _CaptureProvider(fk_a), fk_b)


def test_motion_transport_is_quiet_when_both_share_the_same_fk(monkeypatch):
    def fk(joints):
        return None

    assert not _fk_warnings(monkeypatch, _CaptureProvider(fk), fk)


def test_motion_transport_uses_current_joint_state_for_gripper_commands():
    studio = FakeMotionStudio()
    transport = PiperStudioMotionTransport(studio, max_joint_step_rad=0.2)

    transport.set_gripper_width(0.04, 1000, timeout_s=1.0)

    kind, command, timeout_s = studio.calls[0]
    assert kind == "send"
    assert command.position_rad == pytest.approx((0.0,) * 6)
    assert command.gripper_width_m == pytest.approx(0.04)
    assert command.gripper_effort == 1000
    assert timeout_s == pytest.approx(1.0)


def test_send_uses_caller_timeout_for_gripper_command(piper_web, monkeypatch):
    _, adapter = piper_web
    calls = []
    monkeypatch.setattr(
        adapter,
        "_move_to_normalized",
        lambda target, timeout_s: calls.append((target, timeout_s)),
    )

    adapter.send(ArmCommand((0.0,) * 6, 0.04, 1000, "scripted"), timeout_s=0.1)

    assert calls[0][1] == pytest.approx(0.1)


def test_motion_transport_tool_pose_and_capture_require_providers():
    transport = PiperStudioMotionTransport(FakeMotionStudio(), max_joint_step_rad=0.2)

    with pytest.raises(PiperWebError, match="tool_pose_provider"):
        transport.tool_pose()
    with pytest.raises(PiperWebError, match="in_hand_capture_provider"):
        transport.capture_in_hand(1, 0, timeout_s=1.0)


def test_motion_transport_passes_measured_joints_to_providers():
    studio = FakeMotionStudio()
    studio.joints = (0.1,) * 6
    pose = Pose3D((0.1, 0.2, 0.3), (0.0, 0.0, 0.0, 1.0))
    snapshot = SceneSnapshot(
        1, Source.IN_HAND, CaptureMode.FULL,
        np.zeros((2, 2, 3), dtype=np.uint8),
        np.ones((2, 2), dtype=np.float32),
        (1.0, 1.0, 0.0, 0.0), 0.0,
    )
    capture = InHandCapture(snapshot, np.ones((2, 2), dtype=bool), np.eye(3))
    seen = {}
    transport = PiperStudioMotionTransport(
        studio,
        max_joint_step_rad=0.2,
        tool_pose_provider=lambda joints: (seen.update(pose_joints=joints) or pose),
        in_hand_capture_provider=lambda apple, view, joints, timeout_s: (
            seen.update(
                capture_args=(apple, view, joints, timeout_s)
            ) or capture
        ),
    )

    assert transport.tool_pose() == pose
    assert seen["pose_joints"] == (0.1,) * 6
    assert transport.capture_in_hand(7, 2, timeout_s=1.0) == capture
    assert seen["capture_args"] == (7, 2, (0.1,) * 6, 1.0)


def test_motion_transport_rejects_invalid_provider_payloads():
    transport = PiperStudioMotionTransport(
        FakeMotionStudio(),
        max_joint_step_rad=0.2,
        tool_pose_provider=lambda joints: ("not", "a", "Pose3D"),
        in_hand_capture_provider=lambda apple, view, joints, timeout_s: {
            "not": "an InHandCapture"
        },
    )

    with pytest.raises(PiperWebError, match="must return a Pose3D"):
        transport.tool_pose()
    with pytest.raises(PiperWebError, match="must return InHandCapture or None"):
        transport.capture_in_hand(1, 0, timeout_s=1.0)


def test_motion_transport_rejects_non_positive_capture_timeout():
    transport = PiperStudioMotionTransport(
        FakeMotionStudio(),
        max_joint_step_rad=0.2,
        in_hand_capture_provider=lambda apple, view, joints, timeout_s: None,
    )

    with pytest.raises(ValueError, match="timeout_s"):
        transport.capture_in_hand(1, 0, timeout_s=0.0)


def test_joint_goal_opens_and_closes_v059_jog_session(piper_web):
    fake, adapter = piper_web

    adapter.send_joint_goal((0.0, 1.57079632679, -1.4835298642, 0.0, 0.0, 0.0), 0.5)

    goal = next(call for call in fake.calls if call[1] == "/api/robots/jog/goal")
    assert goal[2]["iface"] == "follower0"
    assert set(goal[2]["values"]) == {f"joint{index}" for index in range(1, 7)}
    assert all(value == pytest.approx(0.0, abs=1e-8) for value in goal[2]["values"].values())
    assert [call[1] for call in fake.calls if call[0] == "POST"][-2:] == [
        "/api/robots/jog/goal",
        "/api/robots/jog/stop",
    ]
    assert any(call[1] == "/api/estop/heartbeat" for call in fake.calls)
    assert fake.calls.index(("GET", "/api/estop/status", None)) < fake.calls.index(
        next(call for call in fake.calls if call[1] == "/api/robots/jog/start")
    )


def test_joint_trajectory_sends_acknowledged_waypoints_in_one_jog_session(piper_web):
    fake, adapter = piper_web
    fake.auto_reach_goals = True

    adapter.send_joint_trajectory(
        (
            (0.01, 1.57079632679, -1.4835298642, 0.0, 0.0, 0.0),
            (0.02, 1.57079632679, -1.4835298642, 0.0, 0.0, 0.0),
        ),
        timeout_s=0.5,
    )

    posts = [call[1] for call in fake.calls if call[0] == "POST"]
    assert posts.count("/api/robots/jog/start") == 1
    assert posts.count("/api/robots/jog/goal") == 2
    assert posts.count("/api/robots/jog/stop") == 1
    assert adapter.joint_state()[0] == pytest.approx(0.02, abs=0.001)


def test_joint_trajectory_validates_every_waypoint_before_starting(piper_web):
    fake, adapter = piper_web

    with pytest.raises(ValueError, match="joint1 target is outside"):
        adapter.send_joint_trajectory(
            (
                (0.01, 1.57079632679, -1.4835298642, 0.0, 0.0, 0.0),
                (10.0, 1.57079632679, -1.4835298642, 0.0, 0.0, 0.0),
            ),
            timeout_s=0.5,
        )

    assert not any(call[1] == "/api/robots/jog/start" for call in fake.calls)


def test_joint_trajectory_rejects_empty_waypoint_sequence(piper_web):
    fake, adapter = piper_web

    with pytest.raises(ValueError, match="at least one joint waypoint"):
        adapter.send_joint_trajectory((), timeout_s=0.5)

    assert not any(call[1] == "/api/robots/jog/start" for call in fake.calls)


def test_joint_goal_refuses_when_estop_bus_is_unavailable(piper_web):
    fake, adapter = piper_web
    fake.estop_bus_available = False

    with pytest.raises(PiperWebError, match="E-stop bus is unavailable"):
        adapter.send_joint_goal((0.0,) * 6, 0.5)

    assert not any(call[1] == "/api/robots/jog/start" for call in fake.calls)


def test_joint_goal_refuses_when_estop_heartbeat_fails(piper_web):
    fake, adapter = piper_web
    fake.estop_heartbeat_status = 503

    with pytest.raises(PiperWebError, match="HTTP 503"):
        adapter.send_joint_goal((0.0,) * 6, 0.5)

    assert not any(call[1] == "/api/robots/jog/start" for call in fake.calls)


def test_estop_heartbeat_is_maintained_during_goal_wait(piper_web):
    fake, adapter = piper_web
    fake.estop_timeout_ms = 300
    fake.estop_heartbeat_interval_ms = 60

    with pytest.raises(PiperWebError, match="Timed out waiting"):
        adapter.send_joint_goal((0.1, 1.57079632679, -1.4835298642, 0.0, 0.0, 0.0), 0.2)

    heartbeat_calls = [
        call for call in fake.calls
        if call[0] == "POST" and call[1] == "/api/estop/heartbeat"
    ]
    assert len(heartbeat_calls) >= 2
    assert any(call[1] == "/api/robots/jog/stop" for call in fake.calls)


def test_estop_heartbeat_failure_aborts_goal_and_stops_jog(piper_web):
    fake, adapter = piper_web
    fake.estop_timeout_ms = 600
    fake.estop_heartbeat_interval_ms = 60
    fake.estop_heartbeat_fail_after = 1

    with pytest.raises(PiperWebError, match="E-stop heartbeat failed"):
        adapter.send_joint_goal((0.1, 1.57079632679, -1.4835298642, 0.0, 0.0, 0.0), 0.5)

    assert any(call[1] == "/api/robots/jog/stop" for call in fake.calls)


def test_motion_stays_disabled_without_explicit_opt_in():
    cfg = {
        "piper_web": {
            "enabled": False,
            "base_url": "http://127.0.0.1:8000",
            "iface": None,
            "request_timeout_s": 1,
            "goal_timeout_s": 1,
            "poll_interval_s": 0.1,
            "arrival_tolerance_rad": 0.01,
            "gripper_arrival_tolerance": 1,
            "joint_limits_deg": {},
            "gripper_width_range_m": None,
        },
    }
    adapter = PiperStudio(cfg)

    with pytest.raises(PiperWebError, match="disabled"):
        adapter.send_joint_goal((0.0,) * 6, 1)


def test_motion_requires_operator_confirmation_of_robotd_safety(piper_web):
    fake, adapter = piper_web
    adapter.cfg["robotd_safety_confirmed"] = False

    with pytest.raises(PiperWebError, match="Confirm Piper Web robotd"):
        adapter.send_joint_goal((0.0,) * 6, 0.5)
    assert fake.calls == []


def test_robot_control_refuses_when_piper_web_authentication_is_disabled(piper_web):
    fake, adapter = piper_web
    fake.auth_enabled = False

    with pytest.raises(PiperWebError, match="login is disabled"):
        adapter.send_joint_goal((0.0,) * 6, 0.5)
    assert not any(call[1] == "/api/robots/jog/start" for call in fake.calls)


def test_read_only_access_cannot_cache_authorization_for_robot_control(piper_web):
    fake, adapter = piper_web
    fake.auth_enabled = False

    adapter.joint_state()
    with pytest.raises(PiperWebError, match="login is disabled"):
        adapter.send_joint_goal((0.0,) * 6, 0.5)
    assert not any(call[1] == "/api/robots/jog/start" for call in fake.calls)


def test_joint_goal_failure_still_stops_the_jog_session(piper_web):
    fake, adapter = piper_web
    fake.goal_status = 409

    with pytest.raises(PiperWebError, match="HTTP 409"):
        adapter.send_joint_goal((0.0,) * 6, 0.5)

    assert any(call[1] == "/api/robots/jog/stop" for call in fake.calls)


def test_combined_arm_command_requires_measured_gripper_range(piper_web):
    fake, adapter = piper_web
    adapter.cfg["gripper_width_range_m"] = None

    with pytest.raises(PiperWebError, match="measured closed/open widths"):
        adapter.send(ArmCommand((0.0,) * 6, 0.04, 1000, "scripted"))
    assert fake.calls == []


def test_combined_arm_command_sends_calibrated_gripper_position(piper_web):
    fake, adapter = piper_web

    adapter.send(ArmCommand((0.0, 1.57079632679, -1.4835298642, 0.0, 0.0, 0.0),
                            0.04, 1000, "scripted"))

    goal = next(call for call in fake.calls if call[1] == "/api/robots/jog/goal")
    assert goal[2]["values"]["gripper"] == pytest.approx(50.0)
    assert any(call[1] == "/api/robots/jog/stop" for call in fake.calls)


def test_combined_arm_command_rejects_effort_other_than_configured_value(piper_web):
    fake, adapter = piper_web

    with pytest.raises(PiperWebError, match="command effort must match"):
        adapter.send(ArmCommand((0.0,) * 6, 0.04, 900, "scripted"))
    assert fake.calls == []


@pytest.mark.parametrize('payload', [
    {}, {'iface': 'other', 'effort_nm': 1},
    {'iface': 'follower0', 'effort_nm': True},
    {'iface': 'follower0', 'effort_nm': '1.0'},
    {'iface': 'follower0', 'effort_nm': float('nan')},
    {'iface': 'follower0', 'effort_nm': -1},
    {'iface': 'follower0', 'effort_nm': 5.1},
    {'iface': 'follower0', 'effort_nm': 1, 'supported': False},
])
def test_invalid_torque_payload_blocks_motion(piper_web, payload):
    fake, adapter = piper_web
    fake.gripper_torque = payload
    with pytest.raises(PiperWebError):
        adapter.send_joint_goal((0,) * 6, .5)
    assert not any(c[1].startswith('/api/robots/jog/') for c in fake.calls)


@pytest.mark.parametrize('status', [404, 503])
def test_unavailable_torque_api_has_no_fixed_1000_fallback(piper_web, status):
    fake, adapter = piper_web
    fake.gripper_torque_status = status
    with pytest.raises(PiperWebError, match=f'HTTP {status}'):
        adapter.send_joint_goal((0,) * 6, .5)
    assert not any(c[1].startswith('/api/robots/jog/') for c in fake.calls)


def test_torque_mismatch_blocks_before_jog_start_without_setting_it(piper_web):
    fake, adapter = piper_web
    fake.gripper_torque['effort_nm'] = .7
    with pytest.raises(PiperWebError, match='Studio 0.7 N·m, expected 1 N·m'):
        adapter.send_joint_goal((0,) * 6, .5)
    assert not any(c[1].startswith('/api/robots/jog/') for c in fake.calls)
    assert not any(c[0] == 'POST' and 'gripper-effort' in c[1] for c in fake.calls)


def test_torque_change_during_wait_stops_jog(piper_web):
    fake, adapter = piper_web
    fake.torque_after_goal = .8
    with pytest.raises(PiperWebError, match='torque mismatch'):
        adapter.send_joint_goal((.1, 1.57079632679, -1.4835298642, 0, 0, 0), .5)
    assert sum(c[1] == '/api/robots/jog/goal' for c in fake.calls) == 1
    assert any(c[1] == '/api/robots/jog/stop' for c in fake.calls)


def test_torque_verification_uses_actual_sdk_millinewton_metre_rounding(piper_web):
    fake, adapter = piper_web
    fake.gripper_torque['effort_nm'] = 1.0004
    assert adapter.verify_gripper_torque() == 1.0004
    fake.gripper_torque['effort_nm'] = 1.0006
    with pytest.raises(PiperWebError, match='torque mismatch'):
        adapter.verify_gripper_torque()


def test_read_only_torque_check_never_starts_or_writes_robot(piper_web):
    fake, adapter = piper_web
    from tools.check_gripper_torque import main
    config = {'piper_web': {'enabled': False, 'iface': 'follower0'}}
    assert main(cfg=config, make_studio=lambda cfg: adapter) == 1
    assert config['piper_web']['enabled'] is False
    assert not any(c[1].startswith('/api/robots/jog/') or
                   (c[0] == 'POST' and 'gripper-effort' in c[1]) for c in fake.calls)


@pytest.mark.parametrize('expected', [None, True, 1.0, -1, 5001])
def test_invalid_expected_torque_refuses_without_sending(piper_web, expected):
    fake, adapter = piper_web
    adapter._expected_gripper_effort = expected
    with pytest.raises(PiperWebError, match='fixed_effort'):
        adapter.send_joint_goal((0,) * 6, .5)
    assert not any(c[1].startswith('/api/robots/jog/') for c in fake.calls)


def test_timed_studio_session_waits_and_always_cleans_up(piper_web):
    import time
    from aple.execution.timed_motion import MotionLimits, TimedJointGoal
    fake, studio = piper_web
    start = studio.joint_state()
    limits = MotionLimits((1.,)*6, (2.,)*6, .02, .1, .1)
    before = time.monotonic()
    studio.send_timed_joint_trajectory((TimedJointGoal(start, .02), TimedJointGoal(start, .04)),
                                      limits, speed_scale=1., timeout_s=.5)
    assert time.monotonic()-before >= .04
    paths = [path for method, path, body in fake.calls if method == 'POST']
    assert paths.count('/api/robots/jog/start') == 1
    assert paths.count('/api/robots/jog/goal') == 2
    assert paths.count('/api/robots/jog/stop') == 1


@pytest.mark.parametrize('feedback_delay_s, succeeds', [(.03, True), (.26, False)])
def test_timed_studio_feedback_crossing_goal_time_respects_total_deadline(
    piper_web, monkeypatch, feedback_delay_s, succeeds,
):
    from types import SimpleNamespace
    import aple.hal.piper_studio as studio_module
    from aple.execution.timed_motion import MotionLimits, TimedJointGoal

    _, studio = piper_web
    start = studio.joint_state()
    clock = [0.]
    calls = []

    def sleep(duration):
        clock[0] += duration

    def read_state():
        clock[0] += feedback_delay_s
        return start, 0.

    def request(method, path, body=None):
        calls.append(path)
        if path.endswith('/start'):
            return {'status': 'started', 'running': True, 'iface': studio._iface}
        return {'status': 'stopped' if path.endswith('/stop') else 'ok'}

    monkeypatch.setattr(studio_module, 'time', SimpleNamespace(
        monotonic=lambda: clock[0], sleep=sleep,
    ))
    monkeypatch.setattr(studio, '_ensure_authenticated', lambda **kw: None)
    monkeypatch.setattr(studio, 'verify_gripper_torque', lambda: 1.)
    monkeypatch.setattr(studio, '_start_estop_keepalive', lambda: SimpleNamespace(
        check=lambda: None, stop=lambda: None,
    ))
    monkeypatch.setattr(studio, '_read_raw_state', read_state)
    monkeypatch.setattr(studio, '_request', request)
    limits = MotionLimits((1.,)*6, (2.,)*6, .02, .1, .1)
    goals = (TimedJointGoal(start, .02), TimedJointGoal(start, .04))
    if succeeds:
        studio.send_timed_joint_trajectory(goals, limits, speed_scale=1., timeout_s=.5)
        assert calls.count('/api/robots/jog/goal') == 2
    else:
        with pytest.raises(PiperWebError, match='deadline'):
            studio.send_timed_joint_trajectory(goals, limits, speed_scale=1., timeout_s=.5)
        assert '/api/robots/jog/goal' not in calls
    assert calls.count('/api/robots/jog/stop') == 1


def test_timed_studio_overspeed_stops_jog(piper_web):
    from aple.execution.timed_motion import MotionLimits, TimedJointGoal
    fake, studio = piper_web
    fake.auto_reach_goals = True # instantaneous fake jumps should violate the physical-rate check
    start = studio.joint_state()
    target = (start[0]+.6, *start[1:])
    limits = MotionLimits((.1,)*6, (.2,)*6, .02, .1, .01)
    with pytest.raises(PiperWebError, match='velocity exceeded'):
        studio.send_timed_joint_trajectory((TimedJointGoal(target, .02), TimedJointGoal(target, .04)),
                                          limits, speed_scale=1., timeout_s=.5)
    assert any(method == 'POST' and path == '/api/robots/jog/stop' for method, path, body in fake.calls)


def test_timed_studio_grip_width_loss_stops_before_next_goal(piper_web):
    from aple.execution.timed_motion import MotionLimits, TimedJointGoal
    fake, studio = piper_web
    start = studio.joint_state()
    limits = MotionLimits((1.,)*6, (2.,)*6, .02, .1, .1)
    with pytest.raises(PiperWebError, match='gripper width changed'):
        studio.send_timed_joint_trajectory((TimedJointGoal(start, .02),), limits,
            speed_scale=1., timeout_s=.5, gripper_width_watch=(.06, .003))
    assert not any(path == '/api/robots/jog/goal' for method, path, body in fake.calls)
    assert any(path == '/api/robots/jog/stop' for method, path, body in fake.calls)


def test_timeout_during_setup_never_starts_jog(piper_web, monkeypatch):
    import time
    fake, studio = piper_web
    monkeypatch.setattr(studio, '_ensure_authenticated', lambda **kw:time.sleep(.06))
    with pytest.raises(PiperWebError, match='Timed out waiting'):
        studio.send_joint_goal((0, 1.57, -1.48, 0, 0, 0), timeout_s=.02)
    assert not any(path == '/api/robots/jog/start' for method, path, body in fake.calls)
