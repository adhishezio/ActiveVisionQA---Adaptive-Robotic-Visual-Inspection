"""Small geometry helpers; camera optical axes are x right, y down, z forward."""

import math
from typing import Iterable

import numpy as np

from .models import Pose


def quaternion_matrix(quaternion: Iterable[float]) -> np.ndarray:
    """Convert a normalized x/y/z/w quaternion to a 3x3 rotation."""
    x, y, z, w = quaternion
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y-z*w), 2 * (x*z+y*w)],
        [2 * (x*y+z*w), 1 - 2 * (x*x+z*z), 2 * (y*z-x*w)],
        [2 * (x*z-y*w), 2 * (y*z+x*w), 1 - 2 * (x*x+y*y)],
    ], dtype=float)


def matrix_quaternion(rotation: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a proper 3x3 rotation matrix to a normalized quaternion."""
    r = np.asarray(rotation, dtype=float)
    if r.shape != (3, 3) or not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(r), 1):
        raise ValueError("Expected a proper rotation matrix")
    trace = np.trace(r)
    if trace > 0:
        scale = math.sqrt(trace + 1) * 2
        q = np.array([(r[2, 1]-r[1, 2])/scale, (r[0, 2]-r[2, 0])/scale,
                      (r[1, 0]-r[0, 1])/scale, scale/4])
    else:
        i = int(np.argmax(np.diag(r)))
        j, k = (i + 1) % 3, (i + 2) % 3
        scale = math.sqrt(1 + r[i, i] - r[j, j] - r[k, k]) * 2
        q = np.zeros(4)
        q[i] = scale / 4
        q[j] = (r[j, i] + r[i, j]) / scale
        q[k] = (r[k, i] + r[i, k]) / scale
        q[3] = (r[k, j] - r[j, k]) / scale
    q /= np.linalg.norm(q)
    return tuple(float(v) for v in q)


def pose_matrix(pose: Pose) -> np.ndarray:
    """Return the transform from the posed frame into pose.frame_id."""
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_matrix(pose.orientation)
    matrix[:3, 3] = pose.position
    return matrix


def pose_from_matrix(matrix: np.ndarray, frame_id: str) -> Pose:
    """Convert a homogeneous rigid transform into an explicitly framed pose."""
    if np.shape(matrix) != (4, 4) or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("Expected a homogeneous transform")
    return Pose(tuple(float(x) for x in matrix[:3, 3]), matrix_quaternion(matrix[:3, :3]), frame_id)


def optical_look_at(position: Iterable[float], target: Iterable[float], frame_id: str = "world") -> Pose:
    """Orient an optical camera at target, keeping image-up near world +z."""
    eye = np.asarray(tuple(position), dtype=float)
    forward = np.asarray(tuple(target), dtype=float) - eye
    if eye.shape != (3,) or forward.shape != (3,) or not np.all(np.isfinite(forward)):
        raise ValueError("Expected finite 3D vectors")
    if np.linalg.norm(forward) < 1e-8:
        raise ValueError("Camera and target cannot coincide")
    forward /= np.linalg.norm(forward)
    reference_up = np.array([0., 0., 1.])
    if abs(np.dot(forward, reference_up)) > 0.99:
        reference_up = np.array([0., 1., 0.])
    right = np.cross(forward, reference_up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.column_stack([right, down, forward])
    return Pose(tuple(float(x) for x in eye), matrix_quaternion(rotation), frame_id)


def pose_error(actual: Pose, desired: Pose) -> tuple[float, float]:
    """Return position error in meters and angular error in radians."""
    if actual.frame_id != desired.frame_id:
        raise ValueError("Cannot compare poses expressed in different frames")
    distance = float(np.linalg.norm(np.asarray(actual.position)-desired.position))
    dot = abs(float(np.dot(actual.orientation, desired.orientation)))
    return distance, 2 * math.acos(min(1.0, max(0.0, dot)))
