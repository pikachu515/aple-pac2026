"""직경으로 크기 구간과 보정된 파지력·그리퍼 effort를 선택한다."""

from __future__ import annotations

import math

from aple.schema import AppleState, GraspProfile

GRAVITY_M_S2 = 9.80665


def select(apple: AppleState, cfg: dict) -> GraspProfile:
    """Select a size profile and require calibrated force limits for execution."""
    diameter_m = float(apple.diameter_m)
    if not math.isfinite(diameter_m) or diameter_m <= 0:
        raise ValueError("apple.diameter_m must be a finite positive value")

    size_cfg = cfg["size_bins"]
    ranges = size_cfg["gripper_range_mm"]
    if len(ranges) != 2 or not 0 < float(ranges[0]) < float(ranges[1]):
        raise ValueError("size_bins.gripper_range_mm must be an increasing [min, max] pair")

    gripper_cfg = cfg["execution"]["gripper"]
    effort = gripper_cfg["fixed_effort"]
    if isinstance(effort, bool) or not isinstance(effort, int) or not 0 <= effort <= 5000:
        raise ValueError("execution.gripper.fixed_effort must be an integer from 0 to 5000")
    effort_limit = cfg.get("safety", {}).get("gripper_effort_max")
    if effort_limit is not None and (
        isinstance(effort_limit, bool)
        or not isinstance(effort_limit, int)
        or not 0 <= effort_limit <= 5000
    ):
        raise ValueError("safety.gripper_effort_max must be an integer from 0 to 5000")
    bins = size_cfg["bins"]
    if not bins:
        raise ValueError("size_bins.bins must contain at least one size bin")

    diameter_mm = diameter_m * 1000.0
    in_range = float(ranges[0]) <= diameter_mm <= float(ranges[1])
    selected_index = -1
    selected_bin = None
    if in_range:
        for index, size_bin in enumerate(bins):
            lower, upper = map(float, size_bin["range_mm"])
            final_bin = index == len(bins) - 1
            if lower <= diameter_mm < upper or (final_bin and lower <= diameter_mm <= upper):
                selected_index, selected_bin = index, size_bin
                break

    density_g_cm3 = float(size_cfg["mass"]["density_g_cm3"])
    if not math.isfinite(density_g_cm3) or density_g_cm3 <= 0:
        raise ValueError("size_bins.mass.density_g_cm3 must be finite and positive")
    mass_kg = density_g_cm3 * 1000.0 * (4.0 / 3.0) * math.pi * (diameter_m / 2.0) ** 3

    if selected_bin is None:
        return GraspProfile(
            scene_id=apple.scene_id,
            apple_id=apple.apple_id,
            size_bin=-1,
            est_mass_kg=mass_kg,
            open_width_m=0.0,
            grasp_height_offset_m=0.0,
            grasp_force_n=0.0,
            effort=effort,
            feasible=False,
        )

    configured_width_mm = selected_bin.get("open_width_mm")
    if configured_width_mm is None:
        opening_margin_m = float(gripper_cfg["opening_margin_m"])
        if not math.isfinite(opening_margin_m) or opening_margin_m < 0:
            raise ValueError("execution.gripper.opening_margin_m must be finite and non-negative")
        open_width_m = diameter_m + opening_margin_m
    else:
        open_width_m = float(configured_width_mm) / 1000.0
        if not math.isfinite(open_width_m) or open_width_m < diameter_m:
            raise ValueError("size-bin open_width_mm must be finite and at least the apple diameter")

    height_offset_mm = float(selected_bin["grasp_height_offset_mm"])
    if not math.isfinite(height_offset_mm):
        raise ValueError("size-bin grasp_height_offset_mm must be finite")

    calibrated_force = _calibrated_force(size_cfg, mass_kg)
    if calibrated_force is None:
        return GraspProfile(
            scene_id=apple.scene_id,
            apple_id=apple.apple_id,
            size_bin=selected_index,
            est_mass_kg=mass_kg,
            open_width_m=open_width_m,
            grasp_height_offset_m=height_offset_mm / 1000.0,
            grasp_force_n=0.0,
            effort=effort,
            feasible=False,
        )

    grasp_force_n, calibrated_effort = calibrated_force
    if effort_limit is None or calibrated_effort > effort_limit:
        return GraspProfile(
            scene_id=apple.scene_id,
            apple_id=apple.apple_id,
            size_bin=selected_index,
            est_mass_kg=mass_kg,
            open_width_m=open_width_m,
            grasp_height_offset_m=height_offset_mm / 1000.0,
            grasp_force_n=grasp_force_n,
            effort=calibrated_effort,
            feasible=False,
        )

    return GraspProfile(
        scene_id=apple.scene_id,
        apple_id=apple.apple_id,
        size_bin=selected_index,
        est_mass_kg=mass_kg,
        open_width_m=open_width_m,
        grasp_height_offset_m=height_offset_mm / 1000.0,
        grasp_force_n=grasp_force_n,
        effort=calibrated_effort,
        feasible=True,
    )


def _calibrated_force(size_cfg: dict, mass_kg: float) -> tuple[float, int] | None:
    force_cfg = size_cfg["force"]
    mu = force_cfg.get("mu")
    bruise_limit_n = force_cfg.get("f_bruise_n")
    if mu is None or bruise_limit_n is None:
        return None

    safety_factor = float(force_cfg["safety_factor"])
    acceleration_m_s2 = float(force_cfg["a_max_m_s2"])
    mu = float(mu)
    bruise_limit_n = float(bruise_limit_n)
    if (
        not all(math.isfinite(value) for value in
                (safety_factor, acceleration_m_s2, mu, bruise_limit_n))
        or safety_factor <= 0
        or acceleration_m_s2 < 0
        or mu <= 0
        or bruise_limit_n <= 0
    ):
        raise ValueError("size_bins.force values must be finite and physically valid")

    minimum_force_n = (
        safety_factor * mass_kg * (GRAVITY_M_S2 + acceleration_m_s2) / (2.0 * mu)
    )
    if minimum_force_n >= bruise_limit_n:
        return None

    table = size_cfg["gripper_calib"].get("effort_to_force_table")
    if table is None:
        return None
    if len(table) < 2:
        raise ValueError("effort_to_force_table must contain at least two calibration points")

    points: list[tuple[int, float]] = []
    for effort_value, force_value in table:
        if isinstance(effort_value, bool) or int(effort_value) != effort_value:
            raise ValueError("calibration effort values must be integers")
        effort_point = int(effort_value)
        force_point = float(force_value)
        if (
            not 0 <= effort_point <= 5000
            or not math.isfinite(force_point)
            or force_point < 0
        ):
            raise ValueError("calibration points must use effort 0..5000 and finite force >= 0")
        points.append((effort_point, force_point))

    if any(
        next_effort <= effort or next_force <= force
        for (effort, force), (next_effort, next_force) in zip(points, points[1:])
    ):
        raise ValueError("calibration effort and force values must be strictly increasing")

    effort_target = _interpolate_effort(points, minimum_force_n)
    if effort_target is None:
        return None
    calibrated_effort = math.ceil(effort_target)
    achieved_force_n = _interpolate_force(points, calibrated_effort)
    if achieved_force_n is None or achieved_force_n >= bruise_limit_n:
        return None
    return minimum_force_n, calibrated_effort


def _interpolate_effort(points: list[tuple[int, float]], force_n: float) -> float | None:
    if force_n <= points[0][1]:
        return float(points[0][0])
    for (effort_a, force_a), (effort_b, force_b) in zip(points, points[1:]):
        if force_a <= force_n <= force_b:
            return effort_a + (force_n - force_a) * (effort_b - effort_a) / (force_b - force_a)
    return None


def _interpolate_force(points: list[tuple[int, float]], effort: int) -> float | None:
    if effort < points[0][0] or effort > points[-1][0]:
        return None
    for (effort_a, force_a), (effort_b, force_b) in zip(points, points[1:]):
        if effort_a <= effort <= effort_b:
            return force_a + (effort - effort_a) * (force_b - force_a) / (effort_b - effort_a)
    return None
