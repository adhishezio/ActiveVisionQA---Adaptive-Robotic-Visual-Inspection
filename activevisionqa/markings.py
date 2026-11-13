"""Synthetic text geometry and a known-fixture camera ROI, without OCR labels."""

import math

import cv2
import numpy as np

from .config import Config
from .geometry import pose_matrix
from .models import Pose

TEXT_WIDTH_M = .15
TEXT_HEIGHT_M = .06
RASTER_SIZE = (160, 64)  # width, height; same aspect ratio as the physical plate
TEXT_FONT = "OpenCV FONT_HERSHEY_SIMPLEX"


def surface_text(config: Config, scenario: str) -> str:
    """Return ground truth from scene generation, never an OCR prediction."""
    if config.surface_text is not None:
        return config.surface_text
    if scenario == "wrong_id":
        last = "0" if config.expected_text[-1] != "0" else "1"
        return config.expected_text[:-1] + last
    return config.expected_text


def text_bitmap(text: str) -> np.ndarray:
    """Draw a binary marking with OpenCV's built-in font; no font download."""
    from dataclasses import replace
    replace(Config(), expected_text=text)  # shared vocabulary validation
    width, height = RASTER_SIZE
    image = np.full((height, width), 255, np.uint8)
    scale = 1.0
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    scale = min((width-16)/tw, (height-16)/(th+baseline))
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    origin = ((width-tw)//2, (height+th-baseline)//2)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, 0, 2, cv2.LINE_8)
    return image


def ink_rectangles(bitmap: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Merge identical horizontal ink runs across rows into exact rectangles."""
    active = {}
    rectangles = []
    for row in range(bitmap.shape[0]+1):
        mask = bitmap[row] == 0 if row < bitmap.shape[0] else np.zeros(bitmap.shape[1], bool)
        edges = np.diff(np.r_[False, mask, False].astype(int))
        runs = set(zip(np.flatnonzero(edges == 1).tolist(), np.flatnonzero(edges == -1).tolist()))
        for run in list(active):
            if run not in runs:
                start = active.pop(run)
                rectangles.append((run[0], start, run[1], row))
        for run in runs:
            active.setdefault(run, row)
    return sorted(rectangles)


def text_corners_world(config: Config) -> np.ndarray:
    """Label-plane corners in world meters, in semantic TL/TR/BR/BL order."""
    xy = np.array([[-1, 1], [1, 1], [1, -1], [-1, -1]], float) * [TEXT_WIDTH_M/2, TEXT_HEIGHT_M/2]
    c, s = math.cos(config.marking_yaw_rad), math.sin(config.marking_yaw_rad)
    xy = xy @ np.array([[c, s], [-s, c]])
    return np.column_stack((xy+np.array(config.target[:2]), np.full(4, config.target[2]+.00016)))


def project_text_roi(config: Config, camera_pose: Pose, intrinsics: list[float]) -> np.ndarray:
    """Project the known part fixture using measured pose and CameraInfo K.

    This uses known synthetic workcell geometry, not a learned text detector.
    It does not establish visibility: an occluder can cover the projected ROI.
    """
    if camera_pose.frame_id != config.world_frame:
        raise ValueError("ROI camera pose must be expressed in the world frame")
    transform = np.linalg.inv(pose_matrix(camera_pose))
    points = np.c_[text_corners_world(config), np.ones(4)] @ transform.T
    if not np.isfinite(points).all() or np.any(points[:, 2] <= .001):
        raise ValueError("Text plane is behind or too close to the camera")
    k = np.asarray(intrinsics, float).reshape(3, 3)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid camera intrinsics")
    projected = points[:, :3] @ k.T
    return projected[:, :2] / projected[:, 2:3]


def rectify_text_roi(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Rectify a fully in-frame fixture ROI; preserve any physical occlusion."""
    height, width = image.shape[:2]
    if np.shape(corners) != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("ROI requires four finite image points")
    if np.any(corners < 0) or np.any(corners[:, 0] >= width) or np.any(corners[:, 1] >= height):
        raise ValueError("Text ROI extends outside image")
    destination = np.array([[0, 0], [479, 0], [479, 191], [0, 191]], np.float32)
    homography = cv2.getPerspectiveTransform(np.asarray(corners, np.float32), destination)
    return cv2.warpPerspective(image, homography, (480, 192))
