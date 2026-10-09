"""Piper Web jog adapter with v0.5.12 read-only gripper torque verification.

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


#: 관절 목표가 Studio 범위 밖으로 이만큼(°)까지는 경계로 자른다. 더 벗어나면 거부한다
LIMIT_EDGE_DEG = 5.0

#: 측정 속도 = 관절 변화 / max(읽기 간격, 이 값). 관절값에 로봇 쪽 시각이 없어 연달아 읽으면 속도가 부푼다
VELOCITY_MIN_WINDOW_S = 0.1


def _arrival_gap(current, target_rad, gripper_position, target) -> str:
    """도착 대기 시간 초과 때 어디서 멈췄는지 — robotd 바닥 필터·관절 범위가 목표를 줄였는지, 덜 따라왔는지 가린다."""
    errors = [abs(a-b) for a, b in zip(current, target_rad)]
    worst = max(range(len(errors)), key=errors.__getitem__)
    text = (f"(worst joint{worst+1} {math.degrees(errors[worst]):.1f}° off; per joint °: "
            + ", ".join(f"{math.degrees(e):.1f}" for e in errors))
    if target.get("gripper") is not None:
        text += f"; gripper {gripper_position:.1f} vs {target['gripper']:.1f}"
    return text + ")"


class PiperStudio:
    """Synchronous jog client requiring v0.5.12 torque readback for control."""

    def __init__(
        self,
        cfg: dict,
        *,
        opener=None,
        snapshot_provider: SnapshotProvider | None = None,
        lighting_status_provider: LightingStatusProvider | None = None,
    ) -> None:
        from aple.execution.gripper_geometry import apply_attachment_geometry
        cfg = apply_attachment_geometry(cfg)
        self.cfg = cfg["piper_web"]
        self._expected_gripper_effort = cfg.get("execution", {}).get("gripper", {}).get("fixed_effort")
        self._grasp_settings = cfg.get('execution', {}).get('gripper', {})
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

    def gripper_torque_nm(self) -> float:
        """Read robotd's configured torque (N·m), not measured fruit contact force."""
        self._require_enabled()
        self._ensure_authenticated(require_control=False)
        result = self._request(
            "GET", f"/api/robots/gripper-effort?iface={quote(self._iface, safe='')}"
        )
        if (not isinstance(result, dict) or result.get("iface") != self._iface
                or result.get("supported") is False):
            raise PiperWebError("Piper Web returned invalid or unsupported gripper torque")
        torque = result.get("effort_nm")
        if (isinstance(torque, bool) or not isinstance(torque, (int, float))
                or not math.isfinite(torque) or not 0 <= torque <= 5):
            raise PiperWebError("Piper Web gripper effort_nm must be finite and within 0..5 N·m")
        return float(torque)

    def verify_gripper_torque(self) -> float:
        """Fail closed on unknown/mismatched torque. Never write Studio settings.

        SDK effort is an integer in 0.001 N·m units. Compare using robotd's same
        rounding, rather than interpreting the setting as grip force in N.
        """
        expected = self._expected_gripper_effort
        if (isinstance(expected, bool) or not isinstance(expected, int)
                or not 0 <= expected <= 5000):
            raise PiperWebError("Set execution.gripper.fixed_effort to an integer 0..5000")
        actual = self.gripper_torque_nm()
        if int(round(actual * 1000)) != expected:
            raise PiperWebError(
                f"Gripper torque mismatch: Studio {actual:g} N·m, "
                f"expected {expected / 1000:g} N·m. Confirm the torque in Studio "
                "and execution.gripper.fixed_effort before executing; no setting was changed"
            )
        return actual

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
        if (isinstance(cmd.gripper_effort, bool) or not isinstance(cmd.gripper_effort, int)
                or cmd.gripper_effort != self._expected_gripper_effort):
            raise PiperWebError("command effort must match execution.gripper.fixed_effort")
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

    def send_timed_joint_trajectory(self, goals, limits, *, speed_scale, timeout_s, gripper_width_watch=None):
        """One protected jog session, scheduled setpoints and measured speed checks."""
        self._require_enabled()
        self._require_robotd_safety_confirmation()
        if not goals:
            raise ValueError('timed trajectory must contain at least one goal')
        if isinstance(speed_scale, bool) or not math.isfinite(speed_scale) or not 0 < speed_scale <= 1:
            raise ValueError('invalid timed trajectory speed scale')
        targets, intervals, last = [], [], 0.0
        for goal in goals:
            if not math.isfinite(goal.time_s) or goal.time_s <= last:
                raise ValueError('timed goal times must be finite and strictly increasing')
            targets.append(self._normalize_joint_target(goal.position_rad))
            intervals.append(goal.time_s-last)
            last = goal.time_s
        if last > timeout_s:
            raise ValueError('timed trajectory cannot fit within timeout')
        self._move_sequence_to_normalized(tuple(targets), timeout_s,
            intervals_s=tuple(intervals),
            velocity_limits_rad_s=tuple(v*speed_scale for v in limits.velocity_rad_s),
            tracking_tolerance_rad=limits.tracking_tolerance_rad,
            tracking_lag_s=getattr(limits, 'tracking_lag_s', 0.5),
            velocity_tolerance_rad_s=limits.velocity_tolerance_rad_s,
            gripper_width_watch=gripper_width_watch)

    def close_for_grasp(self, diameter_m, effort, *, timeout_s):
        """Accept settled contact before the inward goal; verify fruit after lifting."""
        from aple.execution.timed_motion import positive
        self._require_enabled()
        self._require_robotd_safety_confirmation()
        settings = self._grasp_settings
        compression = positive(settings.get('closing_compression_m'), 'closing_compression_m')
        positive(settings.get('contact_settle_s'), 'contact_settle_s')
        positive(settings.get('contact_width_tolerance_m'), 'contact_width_tolerance_m')
        positive(settings.get('contact_stability_m'), 'contact_stability_m')
        if effort != self._expected_gripper_effort:
            raise PiperWebError('command effort must match execution.gripper.fixed_effort')
        goal = self._normalize_joint_target(self.joint_state())
        goal['gripper'] = self._normalize_gripper_width(diameter_m-compression)
        self._move_sequence_to_normalized((goal,), timeout_s, grasp_diameter_m=diameter_m)
        return True

    def _move_sequence_to_normalized(
        self,
        targets: tuple[dict[str, float], ...],
        timeout_s: float,
        *, intervals_s=None, velocity_limits_rad_s=None,
        tracking_tolerance_rad=None, velocity_tolerance_rad_s=None, tracking_lag_s=0.5,
        grasp_diameter_m=None, gripper_width_watch=None,
    ) -> None:
        timeout_s = self._positive_number(timeout_s, "timeout_s")
        with self._lock:
            deadline = time.monotonic() + timeout_s
            self._ensure_authenticated(require_control=True)
            self.verify_gripper_torque()  # before acquiring any jog session
            if time.monotonic() >= deadline:
                raise PiperWebError("Timed out waiting for motion: authentication/torque check")
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
                previous, _ = self._read_raw_state()
                sampled_at = time.monotonic()
                last_sent = sampled_at
                # 추종 오차의 기준: 최근 tracking_lag_s 안에 보낸 목표들 (+ 그 직전 목표). 방금 보낸 목표 하나와만
                # 비교하면 로봇은 늘 한 스텝 + 응답 지연만큼 뒤에 있어 정상 이동도 멈췄다 (10/10 실기)
                sent = [(sampled_at, previous)]
                last_width, stable_since = None, sampled_at
                def sample_checked():
                    nonlocal previous, sampled_at
                    state = self._read_raw_state()
                    now = time.monotonic()
                    if velocity_limits_rad_s is not None and now > sampled_at:
                        # 관절값에는 로봇 쪽 시각이 없다 — 두 HTTP 읽기 사이 시간으로 나눈다. 목표 전송 직후처럼
                        # 몇 ms 만에 다시 읽으면 그 사이 몰려 갱신된 변화가 아주 작은 시간으로 나뉘어 속도가 부풀었다
                        # (10/10 실기 'measured joint velocity exceeded' 오판). 나누는 시간에 하한을 둔다
                        window = max(now-sampled_at, VELOCITY_MIN_WINDOW_S)
                        if any(abs(a-b)/window > v+velocity_tolerance_rad_s
                               for a, b, v in zip(state[0], previous, velocity_limits_rad_s)):
                            raise PiperWebError('measured joint velocity exceeded the configured limit')
                    if gripper_width_watch is not None:
                        closed, opened = self.cfg['gripper_width_range_m']
                        width = closed + state[1]/100 * (opened-closed)
                        anchor, tolerance = gripper_width_watch
                        if abs(width-anchor) > tolerance:
                            raise PiperWebError('gripper width changed during held-apple motion')
                    previous, sampled_at = state[0], now
                    return state
                for index, target in enumerate(targets):
                    if intervals_s is not None:
                        due = last_sent + intervals_s[index]
                        while time.monotonic() < due:
                            keepalive.check()
                            sample_checked()
                            now = time.monotonic()
                            if now >= deadline:
                                raise PiperWebError('timed trajectory exceeded its deadline')
                            remaining = due - now
                            if remaining <= 0:
                                break  # Feedback may finish after the next goal becomes due.
                            time.sleep(min(self._poll_interval_s, remaining))
                    if time.monotonic() >= deadline:
                        raise PiperWebError('Timed out waiting for trajectory: before sending next goal')
                    self.verify_gripper_torque()  # no cached assumption after a UI change
                    if time.monotonic() >= deadline:
                        raise PiperWebError('Timed out waiting for trajectory: torque check')
                    goal_result = self._request(
                        "POST", "/api/robots/jog/goal",
                        {"iface": self._iface, "values": target},
                    )
                    if not isinstance(goal_result, dict) or goal_result.get("status") != "ok":
                        raise PiperWebError("Piper Web did not confirm the joint goal")
                    last_sent = time.monotonic()
                    if last_sent >= deadline:
                        raise PiperWebError('Timed out waiting for trajectory: goal acknowledgment')
                    target_rad = self._denormalize_joint_target(target)
                    sent.append((last_sent, target_rad))
                    while len(sent) > 2 and sent[1][0] < last_sent - tracking_lag_s:
                        sent.pop(0)
                    if intervals_s is not None and index < len(targets)-1:
                        current, _ = sample_checked()
                        error = min(max(abs(a-b) for a, b in zip(current, ref)) for _, ref in sent)
                        if error > tracking_tolerance_rad:
                            raise PiperWebError(
                                'timed trajectory tracking error exceeded the configured limit '
                                f'({math.degrees(error):.1f}° from every goal sent in the last {tracking_lag_s:.2f} s)')
                        continue
                    while True:
                        keepalive.check()
                        self.verify_gripper_torque()
                        current, gripper_position = sample_checked()
                        if time.monotonic() >= deadline:
                            raise PiperWebError("Timed out waiting for trajectory: arrival feedback "
                                                + _arrival_gap(current, target_rad, gripper_position, target))
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
                        if grasp_diameter_m is not None:
                            closed, opened = self.cfg['gripper_width_range_m']
                            width = closed + gripper_position / 100 * (opened-closed)
                            now = time.monotonic()
                            if last_width is None or abs(width-last_width) > self._grasp_settings['contact_stability_m']:
                                stable_since = now
                                last_width = width
                            near_fruit = abs(width-grasp_diameter_m) <= self._grasp_settings['contact_width_tolerance_m']
                            settled = now-stable_since >= self._grasp_settings['contact_settle_s']
                            gripper_reached = settled and (gripper_reached or near_fruit)
                        if joints_reached and gripper_reached:
                            keepalive.check()
                            break
                        remaining_s = deadline - time.monotonic()
                        if remaining_s <= 0:
                            raise PiperWebError(
                                f"Timed out waiting for Piper Web follower {self._iface!r} "
                                "to reach the joint goal sequence "
                                + _arrival_gap(current, target_rad, gripper_position, target)
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
            if degrees < lower_deg - LIMIT_EDGE_DEG or degrees > upper_deg + LIMIT_EDGE_DEG:
                raise ValueError(f"{name} target is outside configured Piper Web joint limits")
            # 경계 바로 밖은 경계로 — robotd 도 같은 값으로 자른다. 10/10 실기: 접힌 자세(2축 -1.6°, 3축 +2.4°)에서
            # 출발하는 궤적의 첫 점이 Studio 범위(2축 0°, 3축 0°) 밖이라 접힌 자세로 가는 이동조차 거부됐다
            degrees = min(max(degrees, lower_deg), upper_deg)
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
    """Timed joint commands with measured tracking/velocity supervision."""

    def __init__(
        self,
        studio: PiperStudio,
        *,
        max_joint_step_rad: float,
        tool_pose_provider: Callable[[tuple[float, ...]], Pose3D] | None = None,
        in_hand_capture_provider: Callable[..., InHandCapture | None] | None = None,
        motion_limits=None,
        holding_verifier=None,
    ) -> None:
        self._studio = studio
        self._tool_pose_provider = tool_pose_provider
        self._in_hand_capture_provider = in_hand_capture_provider
        self._motion_limits = motion_limits
        self._holding_verifier = holding_verifier
        self._held_width_m = None
        capture_fk = getattr(in_hand_capture_provider, "_tool_pose_fn", None)
        if capture_fk is not None and tool_pose_provider is not None and capture_fk is not tool_pose_provider:
            # 구 중심(검사)과 TCP 자세(복구·적재)가 다른 순기구학을 쓰면 서로 어긋난다 (PR #19)
            _LOG.warning(
                "in_hand_capture_provider and tool_pose_provider use different forward "
                "kinematics callables; pass the same function to both so sphere centers match"
            )
        # Legacy constructor compatibility; timing is no longer based on spatial steps.
        PiperStudio._positive_number(max_joint_step_rad, "max_joint_step_rad")

    def joint_state(self) -> tuple[float, ...]:
        return self._studio.joint_state()

    def move_joint_goal(
        self,
        position_rad: tuple[float, ...],
        *,
        speed_scale: float,
        timeout_s: float,
    ) -> bool:
        return self.move_joint_trajectory((position_rad,), speed_scale=speed_scale, timeout_s=timeout_s)

    def move_joint_trajectory(self, waypoints, *, speed_scale, timeout_s):
        from aple.execution.timed_motion import time_parameterize
        if self._motion_limits is None:
            raise PiperWebError('measured motion limits are required for timed speed control')
        if self._held_width_m is not None:
            tolerance = self._studio._grasp_settings['contact_width_tolerance_m']
            if abs(self.gripper_width_m()-self._held_width_m) > tolerance:
                raise PiperWebError('gripper width changed after verified grasp')
        start = self.joint_state()
        goals = time_parameterize(start, waypoints, self._motion_limits, speed_scale=speed_scale)
        if goals[-1].time_s > timeout_s:
            raise PiperWebError('timed trajectory duration exceeds timeout before motion')
        watch = None if self._held_width_m is None else (
            self._held_width_m, self._studio._grasp_settings['contact_width_tolerance_m'])
        self._studio.send_timed_joint_trajectory(goals, self._motion_limits,
            speed_scale=speed_scale, timeout_s=timeout_s, gripper_width_watch=watch)
        return True

    def confirm_holding(self, apple, apple_in_tcp, *, timeout_s):
        if self._holding_verifier is None:
            return False
        confirmed = self._holding_verifier(apple, apple_in_tcp, timeout_s=timeout_s) is True
        if confirmed:
            self._held_width_m = self.gripper_width_m()
        return confirmed

    def close_for_grasp(self, apple, profile, effort, *, timeout_s):
        return self._studio.close_for_grasp(apple.diameter_m, effort, timeout_s=timeout_s)

    def gripper_width_m(self) -> float:
        return self._studio.gripper_width_m()

    def set_gripper_width(
        self, width_m: float, effort: int, *, timeout_s: float
    ) -> bool:
        self._held_width_m = None
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
