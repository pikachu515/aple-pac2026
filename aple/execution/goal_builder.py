"""Build checked TCP targets from role-A grasp candidates and role-B plans."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from aple.schema import AppleState, GraspProfile, PlacementPlan


@dataclass(frozen=True)
class Pose3D:
    position_m: tuple[float, float, float]
    orientation_xyzw: tuple[float, float, float, float]


class GraspPoseCandidate(Protocol):
    grasp_pose: Pose3D
    grasp_yaw_rad: float
    approach_vector: tuple[float, float, float]
    approach_height_m: float


@dataclass(frozen=True)
class MotionTargets:
    pick_pose: Pose3D
    pregrasp_pose: Pose3D
    apple_in_tcp: Pose3D
    place_pose: Pose3D
    hover_pose: Pose3D
    grasp_yaw_rad: float


@dataclass(frozen=True)
class PickTargets:
    pick_pose: Pose3D
    pregrasp_pose: Pose3D
    apple_in_tcp: Pose3D
    grasp_yaw_rad: float


@dataclass(frozen=True)
class PlaceTargets:
    place_pose: Pose3D
    hover_pose: Pose3D
    apple_in_tcp: Pose3D


def build_targets(
    apple: AppleState,
    profile: GraspProfile,
    candidate: GraspPoseCandidate,
    placement: PlacementPlan,
) -> MotionTargets:
    """Derive base_link TCP targets; the spherical apple uses an identity object frame."""
    if not placement.feasible:
        raise ValueError("cannot build motion targets for an infeasible placement")
    _validate_pose(placement_pose(placement), "placement")
    pick_targets = build_pick_targets(apple, profile, candidate)
    place_targets = build_place_targets(
        pick_targets.apple_in_tcp,
        placement,
    )
    return MotionTargets(
        pick_pose=pick_targets.pick_pose,
        pregrasp_pose=pick_targets.pregrasp_pose,
        apple_in_tcp=pick_targets.apple_in_tcp,
        place_pose=place_targets.place_pose,
        hover_pose=place_targets.hover_pose,
        grasp_yaw_rad=pick_targets.grasp_yaw_rad,
    )


def build_pick_targets(
    apple: AppleState,
    profile: GraspProfile,
    candidate: GraspPoseCandidate,
) -> PickTargets:
    """Build the pick approach and expected apple transform for an in-hand workflow."""
    _validate_pose(candidate.grasp_pose, "candidate.grasp_pose")
    if not all(math.isfinite(value) for value in apple.center_m):
        raise ValueError("apple.center_m must contain finite coordinates")
    if not math.isfinite(profile.grasp_height_offset_m):
        raise ValueError("profile.grasp_height_offset_m must be finite")
    if not math.isfinite(candidate.grasp_yaw_rad):
        raise ValueError("candidate.grasp_yaw_rad must be finite")
    if not math.isfinite(candidate.approach_height_m) or candidate.approach_height_m <= 0:
        raise ValueError("candidate.approach_height_m must be finite and positive")
    if len(candidate.approach_vector) != 3 or not all(
        math.isfinite(value) for value in candidate.approach_vector
    ):
        raise ValueError("candidate.approach_vector must contain three finite values")

    approach_norm = math.sqrt(sum(value * value for value in candidate.approach_vector))
    if approach_norm <= 1e-9:
        raise ValueError("candidate.approach_vector must be non-zero")
    approach = tuple(value / approach_norm for value in candidate.approach_vector)
    pick_pose = Pose3D(
        (
            candidate.grasp_pose.position_m[0],
            candidate.grasp_pose.position_m[1],
            candidate.grasp_pose.position_m[2] + profile.grasp_height_offset_m,
        ),
        candidate.grasp_pose.orientation_xyzw,
    )
    pregrasp_pose = Pose3D(
        tuple(
            value + direction * candidate.approach_height_m
            for value, direction in zip(pick_pose.position_m, approach)
        ),
        pick_pose.orientation_xyzw,
    )
    apple_in_tcp = relative_pose(
        pick_pose,
        Pose3D(apple.center_m, (0.0, 0.0, 0.0, 1.0)),
    )
    return PickTargets(
        pick_pose,
        pregrasp_pose,
        apple_in_tcp,
        candidate.grasp_yaw_rad,
    )


def build_place_targets(
    apple_in_tcp: Pose3D,
    placement: PlacementPlan,
) -> PlaceTargets:
    """Build placement TCP targets from the transform measured after picking."""
    if not placement.feasible:
        raise ValueError("cannot build place targets for an infeasible placement")
    _validate_pose(apple_in_tcp, "apple_in_tcp")
    if not all(math.isfinite(value) for value in placement.place_center_m):
        raise ValueError("placement.place_center_m must contain finite coordinates")
    if not all(math.isfinite(value) for value in (placement.hover_z_m, placement.release_z_m)):
        raise ValueError("placement hover and release heights must be finite")
    if placement.hover_z_m < placement.release_z_m:
        raise ValueError("placement hover_z_m must not be below release_z_m")

    place_apple_pose = Pose3D(
        placement.place_center_m,
        (0.0, 0.0, 0.0, 1.0),
    )
    place_pose = compose(place_apple_pose, inverse(apple_in_tcp))
    hover_pose = Pose3D(
        (
            place_pose.position_m[0],
            place_pose.position_m[1],
            place_pose.position_m[2] + placement.hover_z_m - placement.release_z_m,
        ),
        place_pose.orientation_xyzw,
    )
    return PlaceTargets(place_pose, hover_pose, apple_in_tcp)


def relative_pose(parent_pose: Pose3D, child_pose: Pose3D) -> Pose3D:
    """Return the child transform expressed in the parent's coordinate frame."""
    _validate_pose(parent_pose, "parent_pose")
    _validate_pose(child_pose, "child_pose")
    return compose(inverse(parent_pose), child_pose)


def compose(parent_pose: Pose3D, child_pose: Pose3D) -> Pose3D:
    """Compose two position/quaternion transforms."""
    _validate_pose(parent_pose, "parent_pose")
    _validate_pose(child_pose, "child_pose")
    rotated = _rotate(parent_pose.orientation_xyzw, child_pose.position_m)
    return Pose3D(
        tuple(a + b for a, b in zip(parent_pose.position_m, rotated)),
        _multiply(parent_pose.orientation_xyzw, child_pose.orientation_xyzw),
    )


def inverse(pose: Pose3D) -> Pose3D:
    _validate_pose(pose, "pose")
    x, y, z, w = pose.orientation_xyzw
    inverse_orientation = (-x, -y, -z, w)
    inverse_position = _rotate(
        inverse_orientation,
        tuple(-value for value in pose.position_m),
    )
    return Pose3D(inverse_position, inverse_orientation)


def placement_pose(placement: PlacementPlan) -> Pose3D:
    return Pose3D(placement.place_center_m, (0.0, 0.0, 0.0, 1.0))


def _validate_pose(pose: Pose3D, name: str) -> None:
    if len(pose.position_m) != 3 or not all(
        math.isfinite(float(value)) for value in pose.position_m
    ):
        raise ValueError(f"{name}.position_m must contain three finite coordinates")
    if len(pose.orientation_xyzw) != 4 or not all(
        math.isfinite(float(value)) for value in pose.orientation_xyzw
    ):
        raise ValueError(f"{name}.orientation_xyzw must contain four finite values")
    norm = math.sqrt(sum(value * value for value in pose.orientation_xyzw))
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"{name}.orientation_xyzw must be a unit quaternion")


def _multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = left
    rx, ry, rz, rw = right
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _rotate(
    orientation: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    x, y, z, w = orientation
    vx, vy, vz = vector
    # q * (v, 0) * conjugate(q), expanded to avoid temporary quaternions.
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )
