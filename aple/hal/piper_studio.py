"""Piper Web v0.5.9 HTTP adapter.

This uses Piper Web's version-specific authenticated robot jog endpoints. Jog
commands pass through Piper Web's robotd safety filter; this client never opens
the robot CAN device directly. It verifies the E-stop bus and keeps the
gateway heartbeat alive for the duration of each jog session.
"""

from __future__ import annotations

import json
import ipaddress
import logging
import math
import os
import threading
import time
from collections.abc import Callable
from http.cookiejar import CookieJar
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import HTTPCookieProcessor, Request, build_opener

from aple.execution.goal_builder import Pose3D
from aple.inspection.in_hand_capture import InHandCapture
from aple.schema import ArmCommand, LightingStatus, SceneSnapshot

_LOG = logging.getLogger(__name__)

JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
PIPER_WEB_FIXED_GRIPPER_EFFORT = 1000


class PiperWebError(RuntimeError):
    """A Piper Web request failed or the requested operation is unsupported."""


SnapshotProvider = Callable[..., SceneSnapshot]
LightingStatusProvider = Callable[..., LightingStatus]


class _EstopKeepAlive:
    def __init__(
        self,
        request: Callable[..., Any],
        *,
        interval_s: float,
        request_timeout_s: float,
    ) -> None:
        self._request = request
        self._interval_s = interval_s
        self._request_timeout_s = request_timeout_s
        self._stop = threading.Event()
        self._failure: Exception | None = None
        self._failure_lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._beat()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="piper-estop-heartbeat"
        )
        self._thread.start()

    def check(self) -> None:
        with self._failure_lock:
            failure = self._failure
        if failure is not None:
            raise PiperWebError(f"Piper Web E-stop heartbeat failed: {failure}") from failure

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(
                timeout=self._request_timeout_s + self._interval_s + 0.1
            )
            if self._thread.is_alive():
                raise PiperWebError("Piper Web E-stop heartbeat did not stop cleanly")

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            try:
                self._beat()
            except Exception as exc:
                with self._failure_lock:
                    self._failure = exc
                self._stop.set()
                return

    def _beat(self) -> None:
        result = self._request(
            "POST", "/api/estop/heartbeat", {}, timeout_s=self._request_timeout_s
        )
        if not isinstance(result, dict) or result.get("status") != "ok":
            raise PiperWebError("Piper Web did not confirm its E-stop heartbeat")


class PiperStudio:
    """Synchronous client for the Piper Web v0.5.9 jog API."""

    def __init__(
        self,
        cfg: dict,
        *,
        opener=None,
        snapshot_provider: SnapshotProvider | None = None,
        lighting_status_provider: LightingStatusProvider | None = None,
    ) -> None:
        self.cfg = cfg["piper_web"]
        self._enabled = self.cfg.get("enabled", False)
        self._base_url = str(self.cfg["base_url"]).rstrip("/")
        self._iface = self.cfg.get("iface")
        self._request_timeout_s = self._positive_number(
            self.cfg["request_timeout_s"], "piper_web.request_timeout_s"
        )
        self._goal_timeout_s = self._positive_number(
            self.cfg["goal_timeout_s"], "piper_web.goal_timeout_s"
        )
        self._poll_interval_s = self._positive_number(
            self.cfg["poll_interval_s"], "piper_web.poll_interval_s"
        )
        self._arrival_tolerance_rad = self._positive_number(
            self.cfg["arrival_tolerance_rad"], "piper_web.arrival_tolerance_rad"
        )
        self._gripper_arrival_tolerance = self._positive_number(
            self.cfg["gripper_arrival_tolerance"], "piper_web.gripper_arrival_tolerance"
        )
        self._joint_limits_deg = self.cfg["joint_limits_deg"]
        self._snapshot_provider = snapshot_provider
        self._lighting_status_provider = lighting_status_provider
        self._opener = opener or build_opener(HTTPCookieProcessor(CookieJar()))
        self._authenticated = False
        self._auth_status_enabled: bool | None = None
        self._lock = threading.Lock()

        parsed = urlsplit(self._base_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("piper_web.base_url must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("piper_web.base_url must not contain credentials, query, or fragment")
        self._host = parsed.hostname or ""
        try:
            self._loopback = self._host == "localhost" or ipaddress.ip_address(self._host).is_loopback
        except ValueError:
            self._loopback = False
        self._scheme = parsed.scheme
        if not isinstance(self._iface, str) or not self._iface.strip():
            if self._enabled:
                raise ValueError("piper_web.iface must name a configured Piper Web follower")

    def joint_state(self) -> tuple[float, ...]:
        """Read the six arm joints from v0.5.9's raw millidegree endpoint."""
        self._require_enabled()
        self._ensure_authenticated(require_control=False)
        return self._read_joint_state()

    def gripper_width_m(self) -> float:
        """Measured finger width (m) from the raw gripper encoder and the measured width range."""
        self._require_enabled()
        self._ensure_authenticated(require_control=False)
        normalized = self._read_raw_state()[1]
        width_range = self.cfg.get("gripper_width_range_m")
        if not isinstance(width_range, (list, tuple)) or len(width_range) != 2:
            raise PiperWebError(
                "Set piper_web.gripper_width_range_m from measured closed/open widths "
                "before reading the gripper width"
            )
        closed_m, open_m = (float(value) for value in width_range)
        if not all(math.isfinite(value) for value in (closed_m, open_m)) or closed_m < 0 or open_m <= closed_m:
            raise ValueError("piper_web.gripper_width_range_m must be finite and increasing")
        return closed_m + normalized / 100.0 * (open_m - closed_m)

    def snapshot(self, *args, **kwargs):
        if self._snapshot_provider is None:
            raise PiperWebError(
                "Piper Web v0.5.9 exposes color JPEG preview only; provide a frame "
                "source for aligned depth and camera intrinsics before requesting SceneSnapshot"
            )
        result = self._snapshot_provider(*args, **kwargs)
        if not isinstance(result, SceneSnapshot):
            raise PiperWebError("snapshot provider must return a SceneSnapshot")
        return result

    def lighting_status(self, *args, **kwargs):
        if self._lighting_status_provider is None:
            raise PiperWebError(
                "Piper Web v0.5.9 exposes raw light metrics only; provide a "
                "calibration-aware LightingStatus provider"
            )
        result = self._lighting_status_provider(*args, **kwargs)
        if not isinstance(result, LightingStatus):
            raise PiperWebError("lighting-status provider must return LightingStatus")
        return result

    def camera_preview_jpeg(self, camera_id: str | None = None) -> bytes:
        """Fetch one JPEG preview without claiming it is a synchronized scene."""
        self._require_enabled()
        self._ensure_authenticated(require_control=False)
        selected_id = camera_id or self.cfg.get("camera_id")
        if not isinstance(selected_id, str) or not selected_id.strip():
            raise PiperWebError("Set piper_web.camera_id to a registered Piper Web camera")
        payload, content_type = self._request_bytes(
            "GET", f"/api/cameras/{quote(selected_id, safe='')}/preview"
        )
        if content_type.split(";", 1)[0].strip().lower() != "image/jpeg":
            raise PiperWebError(
                f"Piper Web camera preview returned unexpected content type: {content_type}"
            )
        if not payload:
            raise PiperWebError("Piper Web returned an empty camera preview")
        return payload

    def camera_light_metrics(self) -> dict[str, Any]:
        """Return Piper Web's raw per-camera light measurements."""
        self._require_enabled()
        self._ensure_authenticated(require_control=False)
        result = self._request("GET", "/api/cameras/light")
        if not isinstance(result, dict):
            raise PiperWebError("Piper Web returned an invalid camera-light payload")
        return result

    def send_joint_goal(self, position_rad: tuple[float, ...], timeout_s: float) -> None:
        """Send one absolute six-joint goal through the protected jog session."""
        self._require_enabled()
        self._require_robotd_safety_confirmation()
        target = self._normalize_joint_target(position_rad)
        self._move_to_normalized(target, timeout_s)

    def send_joint_trajectory(
        self,
        points_rad: tuple[tuple[float, ...], ...],
        timeout_s: float,
    ) -> None:
        """Send ordered joint waypoints in one protected jog session.

        Piper Web v0.5.9 does not expose per-goal speed, so this method only
        sequences acknowledged waypoints; it does not promise continuous or
        speed-scaled motion.
        """
        self._require_enabled()
        self._require_robotd_safety_confirmation()
        if not points_rad:
            raise ValueError("points_rad must contain at least one joint waypoint")
        targets = tuple(self._normalize_joint_target(point) for point in points_rad)
        self._move_sequence_to_normalized(targets, timeout_s)

    def send(self, cmd: ArmCommand, *, timeout_s: float | None = None) -> None:
        """Send a measured joint pose and calibrated gripper width through web jog."""
        self._require_enabled()
        self._require_robotd_safety_confirmation()
        joints = self._normalize_joint_target(cmd.position_rad)
        gripper = self._normalize_gripper_width(cmd.gripper_width_m)
        if cmd.gripper_effort != PIPER_WEB_FIXED_GRIPPER_EFFORT:
            raise PiperWebError(
                "Piper Web v0.5.9 robotd hard-codes GripperCtrl effort to 1000; "
                "change the Piper Web robotd implementation before tuning this value"
            )
        target = {**joints, "gripper": gripper}
        timeout = (
            self._goal_timeout_s
            if timeout_s is None
            else self._positive_number(timeout_s, "timeout_s")
        )
        self._move_to_normalized(
            target,
            timeout,
        )

    def move_named(
        self,
        name: str,
        joint_target_rad: tuple[float, ...],
        *,
        speed_scale: float,
        timeout_s: float,
    ) -> bool:
        """The v0.5.9 jog API cannot honor MotionAdapter's per-goal speed contract."""
        del name, joint_target_rad, speed_scale, timeout_s
        raise PiperWebError(
            "Piper Web v0.5.9 jog has no per-goal speed parameter; use send_joint_goal "
            "only after configuring and verifying robotd's global safety limits"
        )

    def execute_phase_a(self, *args: Any, **kwargs: Any):
        raise PiperWebError(
            "Piper Web v0.5.9 jog accepts joint goals only; Cartesian approach, "
            "verified grasp/holding, and transfer need a validated IK/backend"
        )

    def execute_pick(self, *args: Any, **kwargs: Any):
        raise PiperWebError(
            "Piper Web v0.5.9 jog has no validated pick/holding or measured TCP-pose workflow"
        )

    def execute_place_phase_a(self, *args: Any, **kwargs: Any):
        raise PiperWebError(
            "Piper Web v0.5.9 jog has no validated held-apple transfer-to-placement workflow"
        )

    def execute_phase_b(self, *args: Any, **kwargs: Any):
        raise PiperWebError(
            "Piper Web v0.5.9 jog does not provide a verified soft-landing/release "
            "sequence; Phase B is unavailable"
        )

    def inspect(self, *args: Any, **kwargs: Any) -> bool:
        raise PiperWebError(
            "Piper Web v0.5.9 jog does not provide a validated in-hand inspection sequence"
        )

    def inspect_views(self, *args: Any, **kwargs: Any):
        raise PiperWebError(
            "Piper Web v0.5.9 jog does not provide an in-hand capture sequence"
        )

    def _move_to_normalized(self, target: dict[str, float], timeout_s: float) -> None:
        self._move_sequence_to_normalized((target,), timeout_s)

    def _move_sequence_to_normalized(
        self,
        targets: tuple[dict[str, float], ...],
        timeout_s: float,
    ) -> None:
        timeout_s = self._positive_number(timeout_s, "timeout_s")
        with self._lock:
            self._ensure_authenticated(require_control=True)
            iface = quote(self._iface, safe="")
            keepalive = self._start_estop_keepalive()
            started = False
            operation_error: Exception | None = None
            try:
                start_result = self._request(
                    "POST", "/api/robots/jog/start", {"iface": self._iface}
                )
                started = True
                if (not isinstance(start_result, dict)
                        or start_result.get("status") != "started"
                        or start_result.get("running") is not True
                        or start_result.get("iface") != self._iface):
                    raise PiperWebError("Piper Web did not confirm ownership of the jog session")
                deadline = time.monotonic() + timeout_s
                for target in targets:
                    goal_result = self._request(
                        "POST", "/api/robots/jog/goal",
                        {"iface": self._iface, "values": target},
                    )
                    if not isinstance(goal_result, dict) or goal_result.get("status") != "ok":
                        raise PiperWebError("Piper Web did not confirm the joint goal")
                    target_rad = self._denormalize_joint_target(target)
                    while True:
                        keepalive.check()
                        current, gripper_position = self._read_raw_state()
                        joints_reached = all(
                            abs(actual - expected) <= self._arrival_tolerance_rad
                            for actual, expected in zip(current, target_rad)
                        )
                        target_gripper = target.get("gripper")
                        gripper_reached = (
                            target_gripper is None
                            or abs(gripper_position - target_gripper)
                            <= self._gripper_arrival_tolerance
                        )
                        if joints_reached and gripper_reached:
                            keepalive.check()
                            break
                        remaining_s = deadline - time.monotonic()
                        if remaining_s <= 0:
                            raise PiperWebError(
                                f"Timed out waiting for Piper Web follower {self._iface!r} "
                                "to reach the joint goal sequence"
                            )
                        time.sleep(min(self._poll_interval_s, remaining_s))
            except Exception as exc:
                operation_error = exc
                raise
            finally:
                cleanup_errors: list[str] = []
                if started:
                    try:
                        stop_result = self._request("POST", "/api/robots/jog/stop", {})
                        if not isinstance(stop_result, dict) or stop_result.get("status") != "stopped":
                            raise PiperWebError("Piper Web did not confirm that jog stopped")
                    except Exception as stop_error:
                        cleanup_errors.append(f"failed to stop Piper Web jog: {stop_error}")
                try:
                    keepalive.stop()
                    keepalive.check()
                except Exception as heartbeat_error:
                    cleanup_errors.append(
                        f"failed to stop Piper Web E-stop heartbeat cleanly: {heartbeat_error}"
                    )
                if cleanup_errors:
                    cleanup_message = "; ".join(cleanup_errors)
                    if operation_error is not None:
                        raise PiperWebError(
                            f"{operation_error}; additionally {cleanup_message}"
                        ) from operation_error
                    raise PiperWebError(cleanup_message)

    def _start_estop_keepalive(self) -> _EstopKeepAlive:
        status = self._request("GET", "/api/estop/status")
        if not isinstance(status, dict) or status.get("bus_available") is not True:
            raise PiperWebError(
                "Piper Web E-stop bus is unavailable; refusing to start robot motion"
            )
        try:
            timeout_ms = self._positive_number(
                status.get("timeout_ms"), "E-stop timeout_ms"
            )
            interval_ms = self._positive_number(
                status.get("heartbeat_interval_ms"), "E-stop heartbeat_interval_ms"
            )
        except ValueError as exc:
            raise PiperWebError(f"Piper Web returned invalid E-stop timing: {exc}") from exc
        timeout_s = timeout_ms / 1000.0
        interval_s = min(interval_ms / 2000.0, timeout_s / 3.0)
        request_timeout_s = min(self._request_timeout_s, timeout_s / 3.0)
        keepalive = _EstopKeepAlive(
            self._request,
            interval_s=interval_s,
            request_timeout_s=request_timeout_s,
        )
        keepalive.start()
        return keepalive

    def _read_joint_state(self) -> tuple[float, ...]:
        return self._read_raw_state()[0]

    def _read_raw_state(self) -> tuple[tuple[float, ...], float]:
        iface = quote(self._iface, safe="")
        raw = self._request("GET", f"/api/robots/joints/raw/{iface}")
        if not isinstance(raw, dict):
            raise PiperWebError("Piper Web returned an invalid raw joint-state payload")
        try:
            values = tuple(float(raw[name]) for name in JOINT_NAMES)
            gripper_raw = float(raw["gripper"])
        except (KeyError, TypeError, ValueError) as exc:
            raise PiperWebError("Piper Web joint-state payload is missing arm or gripper values") from exc
        if not all(math.isfinite(value) for value in (*values, gripper_raw)):
            raise PiperWebError("Piper Web returned a non-finite joint position")
        if not 0 <= gripper_raw <= 68000:
            raise PiperWebError("Piper Web returned a gripper encoder value outside calibration")
        joints_rad = tuple(math.radians(value / 1000.0) for value in values)
        gripper_normalized = gripper_raw * 100.0 / 68000.0
        return joints_rad, gripper_normalized

    def _normalize_joint_target(self, values: tuple[float, ...]) -> dict[str, float]:
        if len(values) != len(JOINT_NAMES):
            raise ValueError("joint_target_rad must contain exactly six joint angles")
        normalized: dict[str, float] = {}
        for name, value in zip(JOINT_NAMES, values):
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"{name} target must be finite")
            limits = self._joint_limits_deg.get(name)
            if not isinstance(limits, (list, tuple)) or len(limits) != 2:
                raise ValueError(f"piper_web.joint_limits_deg.{name} must be [min, max]")
            lower_deg, upper_deg = (float(limit) for limit in limits)
            if (not math.isfinite(lower_deg) or not math.isfinite(upper_deg)
                    or lower_deg >= upper_deg):
                raise ValueError(f"piper_web joint limits for {name} are invalid")
            degrees = math.degrees(value)
            if degrees < lower_deg or degrees > upper_deg:
                raise ValueError(f"{name} target is outside configured Piper Web joint limits")
            normalized[name] = (degrees - lower_deg) * 200.0 / (upper_deg - lower_deg) - 100.0
        return normalized

    def _denormalize_joint_target(self, target: dict[str, float]) -> tuple[float, ...]:
        result = []
        for name in JOINT_NAMES:
            lower_deg, upper_deg = (float(value) for value in self._joint_limits_deg[name])
            degrees = lower_deg + (target[name] + 100.0) * (upper_deg - lower_deg) / 200.0
            result.append(math.radians(degrees))
        return tuple(result)

    def _normalize_gripper_width(self, width_m: float) -> float:
        width_range = self.cfg.get("gripper_width_range_m")
        if not isinstance(width_range, (list, tuple)) or len(width_range) != 2:
            raise PiperWebError(
                "Set piper_web.gripper_width_range_m from measured closed/open widths "
                "before sending gripper commands"
            )
        closed_m, open_m = (float(value) for value in width_range)
        width_m = float(width_m)
        if (not all(math.isfinite(value) for value in (closed_m, open_m, width_m))
                or closed_m < 0 or open_m <= closed_m):
            raise ValueError("piper_web.gripper_width_range_m must be finite and increasing")
        if width_m < closed_m or width_m > open_m:
            raise ValueError("requested gripper width is outside measured closed/open limits")
        return (width_m - closed_m) * 100.0 / (open_m - closed_m)

    def _require_enabled(self) -> None:
        if not self._enabled:
            raise PiperWebError(
                "Piper Web control is disabled; explicitly enable piper_web.enabled "
                "after checking the robotd safety configuration"
            )

    def _require_robotd_safety_confirmation(self) -> None:
        if not self.cfg.get("robotd_safety_confirmed", False):
            raise PiperWebError(
                "Confirm Piper Web robotd's joint, velocity, workspace, and deadman "
                "safety settings before enabling motion"
            )

    def _ensure_authenticated(self, *, require_control: bool) -> None:
        if self._authenticated and (not require_control or self._auth_status_enabled):
            return
        if self._scheme != "https" and not self._loopback:
            raise PiperWebError(
                "Use HTTPS for a remote Piper Web host; plain HTTP is allowed only "
                "for loopback connections"
            )
        if self._auth_status_enabled is None:
            status = self._request("GET", "/api/auth/status")
            if not isinstance(status, dict) or not isinstance(status.get("enabled"), bool):
                raise PiperWebError("Piper Web returned an invalid authentication status")
            self._auth_status_enabled = status["enabled"]
            if status.get("authenticated") is True:
                self._authenticated = True
        if not self._auth_status_enabled:
            if require_control:
                raise PiperWebError(
                    "Piper Web login is disabled; set a password before enabling robot control"
                )
            self._authenticated = True
            return
        if self._authenticated:
            self._authenticated = True
            return
        password = os.environ.get("PIPER_WEB_PASSWORD")
        if not password:
            raise PiperWebError(
                "Piper Web login is enabled; set PIPER_WEB_PASSWORD in the process "
                "environment (do not put it in YAML or source code)"
            )
        result = self._request("POST", "/api/auth/login", {"password": password})
        if not isinstance(result, dict) or result.get("authenticated") is not True:
            raise PiperWebError("Piper Web login did not establish an authenticated session")
        self._authenticated = True

    def _request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        url = f"{self._base_url}{path}"
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        request = Request(url, data=data, headers=headers, method=method)
        try:
            with self._opener.open(
                request,
                timeout=self._request_timeout_s if timeout_s is None else timeout_s,
            ) as response:
                payload = response.read()
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise PiperWebError(
                f"Piper Web {method} {path} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise PiperWebError(f"Piper Web {method} {path} request failed: {exc}") from exc
        try:
            return json.loads(payload.decode("utf-8")) if payload else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise PiperWebError(f"Piper Web {method} {path} returned invalid JSON") from exc

    def _request_bytes(self, method: str, path: str) -> tuple[bytes, str]:
        request = Request(
            f"{self._base_url}{path}",
            headers={"Accept": "image/jpeg"},
            method=method,
        )
        try:
            with self._opener.open(request, timeout=self._request_timeout_s) as response:
                return response.read(), response.headers.get("Content-Type", "")
        except HTTPError as exc:
            detail = exc.read(4096).decode("utf-8", errors="replace")
            raise PiperWebError(
                f"Piper Web {method} {path} returned HTTP {exc.code}: {detail}"
            ) from exc
        except (URLError, TimeoutError, OSError) as exc:
            raise PiperWebError(f"Piper Web {method} {path} request failed: {exc}") from exc

    @staticmethod
    def _positive_number(value: Any, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive finite number") from exc
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return result


class PiperStudioMotionTransport:
    """Speed-aware motion transport built on top of PiperStudio.

    Piper Web v0.5.9 does not expose per-goal speed control, so this transport
    approximates deceleration by subdividing joint motion into smaller waypoint
    steps when a lower speed_scale is requested. The actual robotd runtime still
    decides the final servo profile, so simulation is required before treating
    this as a verified slowdown contract.
    """

    def __init__(
        self,
        studio: PiperStudio,
        *,
        max_joint_step_rad: float,
        tool_pose_provider: Callable[[tuple[float, ...]], Pose3D] | None = None,
        in_hand_capture_provider: Callable[..., InHandCapture | None] | None = None,
    ) -> None:
        self._studio = studio
        self._tool_pose_provider = tool_pose_provider
        self._in_hand_capture_provider = in_hand_capture_provider
        capture_fk = getattr(in_hand_capture_provider, "_tool_pose_fn", None)
        if capture_fk is not None and tool_pose_provider is not None and capture_fk is not tool_pose_provider:
            # 구 중심(검사)과 TCP 자세(복구·적재)가 다른 순기구학을 쓰면 서로 어긋난다 (PR #19)
            _LOG.warning(
                "in_hand_capture_provider and tool_pose_provider use different forward "
                "kinematics callables; pass the same function to both so sphere centers match"
            )
        self._max_joint_step_rad = PiperStudio._positive_number(
            max_joint_step_rad, "max_joint_step_rad"
        )

    def joint_state(self) -> tuple[float, ...]:
        return self._studio.joint_state()

    def move_joint_goal(
        self,
        position_rad: tuple[float, ...],
        *,
        speed_scale: float,
        timeout_s: float,
    ) -> bool:
        waypoints = self._build_waypoints(position_rad, speed_scale=speed_scale)
        if len(waypoints) == 1:
            self._studio.send_joint_goal(position_rad, timeout_s)
        else:
            self._studio.send_joint_trajectory(waypoints, timeout_s)
        return True

    def gripper_width_m(self) -> float:
        return self._studio.gripper_width_m()

    def set_gripper_width(
        self, width_m: float, effort: int, *, timeout_s: float
    ) -> bool:
        joints = self._studio.joint_state()
        self._studio.send(
            ArmCommand(joints, width_m, effort, "scripted"),
            timeout_s=timeout_s,
        )
        return True

    def tool_pose(self) -> Pose3D:
        """Forward kinematics of the measured joints via the injected provider (base_link, m)."""
        if self._tool_pose_provider is None:
            raise PiperWebError(
                "Provide a tool_pose_provider (e.g. Piper forward kinematics); "
                "Piper Web v0.5.9 does not report a TCP pose"
            )
        result = self._tool_pose_provider(self._studio.joint_state())
        if not isinstance(result, Pose3D):
            raise PiperWebError("tool_pose_provider must return a Pose3D in base_link meters")
        return result

    def capture_in_hand(
        self, apple_id: int, view_index: int, *, timeout_s: float
    ) -> InHandCapture | None:
        if self._in_hand_capture_provider is None:
            raise PiperWebError(
                "Provide an in_hand_capture_provider returning aligned color/depth "
                "and intrinsics; Piper Web v0.5.9 preview is JPEG only"
            )
        timeout = PiperStudio._positive_number(timeout_s, "timeout_s")
        result = self._in_hand_capture_provider(
            apple_id, view_index, self._studio.joint_state(), timeout_s=timeout
        )
        if result is not None and not isinstance(result, InHandCapture):
            raise PiperWebError(
                "in_hand_capture_provider must return InHandCapture or None"
            )
        return result

    def _build_waypoints(
        self,
        target_rad: tuple[float, ...],
        *,
        speed_scale: float,
    ) -> tuple[tuple[float, ...], ...]:
        if len(target_rad) != len(JOINT_NAMES):
            raise ValueError("joint_target_rad must contain exactly six joint angles")
        target = tuple(float(value) for value in target_rad)
        if not all(math.isfinite(value) for value in target):
            raise ValueError("joint_target_rad must contain only finite values")
        if not math.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError("speed_scale must be finite and in (0, 1]")

        current = self._studio.joint_state()
        max_delta = max(abs(end - start) for start, end in zip(current, target))
        if max_delta == 0:
            return (target,)
        effective_step = max(self._max_joint_step_rad * speed_scale, 1e-6)
        segment_count = max(1, math.ceil(max_delta / effective_step))
        waypoints = []
        for index in range(1, segment_count + 1):
            fraction = index / segment_count
            waypoint = tuple(
                start + (end - start) * fraction for start, end in zip(current, target)
            )
            waypoints.append(waypoint)
        return tuple(waypoints)
