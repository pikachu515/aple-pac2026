"""Time-parameterized joint setpoints; real limits come from measured YAML.

These bound commanded motion. A follower must separately check measured tracking
and velocity; interpolation alone is not a certificate of physical motor speed.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class MotionLimits:
    velocity_rad_s: tuple[float, ...]
    acceleration_rad_s2: tuple[float, ...]
    period_s: float
    tracking_tolerance_rad: float
    velocity_tolerance_rad_s: float
    # 추종 오차는 최근 이 시간 안에 보낸 목표 중 가장 가까운 것과 잰다 — Studio 릴레이·모터 응답 지연만큼은 뒤처져도 된다
    tracking_lag_s: float = 0.5

    def __post_init__(self):
        for name in ('velocity_rad_s', 'acceleration_rad_s2'):
            values = getattr(self, name)
            if len(values) != 6:
                raise ValueError(f'{name} requires six joint limits')
            for value in values:
                positive(value, name)
        for name in ('period_s', 'tracking_tolerance_rad', 'velocity_tolerance_rad_s', 'tracking_lag_s'):
            positive(getattr(self, name), name)

    @classmethod
    def from_config(cls, settings):
        def vector(key):
            values = settings.get(key)
            if not isinstance(values, (list, tuple)) or len(values) != 6:
                raise ValueError(f'{key} requires six measured joint limits')
            return tuple(positive(v, key) for v in values)
        return cls(vector('velocity_limits_rad_s'), vector('acceleration_limits_rad_s2'),
                   positive(settings.get('command_period_s'), 'command_period_s'),
                   positive(settings.get('tracking_tolerance_rad'), 'tracking_tolerance_rad'),
                   positive(settings.get('velocity_tolerance_rad_s'), 'velocity_tolerance_rad_s'),
                   positive(settings.get('tracking_lag_s', cls.tracking_lag_s), 'tracking_lag_s'))


def positive(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be a positive finite measured value')
    return float(value)


@dataclass(frozen=True)
class TimedJointGoal:
    position_rad: tuple[float, ...]
    time_s: float


def time_parameterize(start, waypoints, limits: MotionLimits, *, speed_scale):
    """Minimum-jerk straight joint legs, preserving non-collinear path corners.

    Collinear subdivisions are collapsed because they represent the same checked
    segment. We never interpolate across a corner of the validated joint path.
    """
    scale = positive(speed_scale, 'speed_scale')
    if scale > 1:
        raise ValueError('speed_scale must be at most 1')
    values = np.asarray([start, *waypoints], dtype=float)
    if values.ndim != 2 or values.shape[1] != 6 or not np.isfinite(values).all() or len(values) < 2:
        raise ValueError('joint path must contain finite six-joint positions')
    compact = [values[0]]
    for i in range(1, len(values)-1):
        a, b = values[i] - compact[-1], values[i+1] - values[i]
        norm = float(np.dot(a, a))
        aligned = norm > 0 and np.dot(a, b) >= 0 and np.linalg.norm(b - a*np.dot(a, b)/norm) < 1e-10
        if not aligned and np.linalg.norm(a) > 1e-12:
            compact.append(values[i])
    compact.append(values[-1])
    result, elapsed = [], 0.0
    for left, right in zip(compact, compact[1:]):
        delta = np.abs(right-left)
        duration = max(limits.period_s,
            float(np.max(1.875 * delta / (np.asarray(limits.velocity_rad_s)*scale))),
            float(np.max(np.sqrt((10/math.sqrt(3))*delta / (np.asarray(limits.acceleration_rad_s2)*scale)))))
        count = max(1, math.ceil(duration/limits.period_s))
        # Round upward: no adjacent commands closer than the configured period.
        duration = count * limits.period_s
        for j in range(1, count+1):
            u = j/count
            s = 10*u**3 - 15*u**4 + 6*u**5
            result.append(TimedJointGoal(tuple(left+(right-left)*s), elapsed+j*limits.period_s))
        elapsed += duration
    return tuple(result)
