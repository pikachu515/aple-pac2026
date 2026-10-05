"""Piper Web v0.5.9 HTTP adapter.

This uses Piper Web's version-specific authenticated robot jog endpoints. Jog
commands pass through Piper Web's robotd safety filter; this client never opens
the robot CAN device directly. It verifies the E-stop bus and keeps the
gateway heartbeat alive for the duration of each jog session.
"""

from __future__ import annotations

import json
import ipaddress
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

from aple.schema import ArmCommand

JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
PIPER_WEB_FIXED_GRIPPER_EFFORT = 1000


class PiperWebError(RuntimeError):
    """A Piper Web request failed or the requested operation is unsupported."""


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

    def __init__(self, cfg: dict, *, opener=None) -> None:
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

    def snapshot(self, *args, **kwargs):
        raise PiperWebError("Piper Web v0.5.9 camera snapshot adapter is not implemented")

    def lighting_status(self):
        raise PiperWebError("Piper Web v0.5.9 lighting-status adapter is not implemented")

    def send_joint_goal(self, position_rad: tuple[float, ...], timeout_s: float) -> None:
        """Send one absolute six-joint goal through the protected jog session."""
        self._require_enabled()
        self._require_robotd_safety_confirmation()
        target = self._normalize_joint_target(position_rad)
        self._move_to_normalized(target, timeout_s)

    def send(self, cmd: ArmCommand) -> None:
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
        self._move_to_normalized(target, self._goal_timeout_s)

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
                goal_result = self._request(
                    "POST", "/api/robots/jog/goal",
                    {"iface": self._iface, "values": target},
                )
                if not isinstance(goal_result, dict) or goal_result.get("status") != "ok":
                    raise PiperWebError("Piper Web did not confirm the joint goal")
                deadline = time.monotonic() + timeout_s
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
                        or abs(gripper_position - target_gripper) <= self._gripper_arrival_tolerance
                    )
                    if joints_reached and gripper_reached:
                        keepalive.check()
                        return
                    if time.monotonic() >= deadline:
                        raise PiperWebError(
                            f"Timed out waiting for Piper Web follower {self._iface!r} "
                            "to reach the joint goal"
                        )
                    time.sleep(min(self._poll_interval_s, max(0.0, deadline - time.monotonic())))
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

    @staticmethod
    def _positive_number(value: Any, name: str) -> float:
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must be a positive finite number") from exc
        if not math.isfinite(result) or result <= 0:
            raise ValueError(f"{name} must be a positive finite number")
        return result
