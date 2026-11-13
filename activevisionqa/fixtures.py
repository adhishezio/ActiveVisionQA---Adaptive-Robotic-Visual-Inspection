"""Offline image fixtures and injected failures, never a robot simulation claim."""

from dataclasses import dataclass

import numpy as np

from .config import Config
from .models import AcquisitionError, MotionResult, Observation, Viewpoint


@dataclass(frozen=True)
class Scenario:
    name: str
    truth: str
    description: str


SCENARIOS = (
    Scenario("clear", "acceptable", "Expected identifier visible at both views"),
    Scenario("occluded", "acceptable", "First view is occluded; second view is clear"),
    Scenario("small_then_clear", "acceptable", "First marking is too small; second is clear"),
    Scenario("blurred_then_clear", "acceptable", "First image is blurred; second is clear"),
    Scenario("wrong_id", "defective", "A clearly visible incorrect identifier"),
    Scenario("wrong_id_occluded", "defective", "Incorrect identifier hidden at first view"),
    Scenario("all_occluded", "acceptable", "No usable marking at either view"),
    Scenario("planning_failure", "acceptable", "Injected first-view planning failure"),
    Scenario("execution_failure", "acceptable", "Injected execution failure requires a stop"),
    Scenario("camera_timeout", "acceptable", "Injected missing images"),
    Scenario("stale_image", "acceptable", "Injected image older than motion completion"),
    Scenario("wrong_frame", "acceptable", "Injected image with wrong optical frame"),
)


def marker_image(identifier: int, size: int = 256) -> np.ndarray:
    """Generate a self-authored marking using OpenCV's predefined dictionary."""
    import cv2
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    if hasattr(cv2.aruco, "generateImageMarker"):
        return cv2.aruco.generateImageMarker(dictionary, identifier, size)
    return cv2.aruco.drawMarker(dictionary, identifier, size)


def fixture_image(scenario: str, view_index: int, seed: int, expected_id: int = 7) -> np.ndarray:
    """Render a deterministic 2D test image; no Gazebo/robot data is fabricated."""
    import cv2
    rng = np.random.default_rng(seed)
    image = np.full((480, 640, 3), 225, dtype=np.uint8)
    identifier = 23 if scenario in ("wrong_id", "wrong_id_occluded") else expected_id
    edge = 32 if scenario == "small_then_clear" and view_index == 0 else 220
    center = (320 + int(rng.integers(-8, 9)), 240 + int(rng.integers(-8, 9)))
    x, y = center[0]-edge//2, center[1]-edge//2
    image[y-20:y+edge+20, x-20:x+edge+20] = 255
    image[y:y+edge, x:x+edge] = marker_image(identifier, edge)[:, :, None]
    first_occluded = scenario in ("occluded", "wrong_id_occluded") and view_index == 0
    if first_occluded or scenario == "all_occluded":
        image[y-5:y+edge+5, x+edge//4:x+3*edge//4] = (110, 110, 110)
    if scenario == "blurred_then_clear" and view_index == 0:
        image = cv2.GaussianBlur(image, (41, 41), 8.0)
    return image


class FixtureBackend:
    """Deterministic policy test double; it does not move or validate an arm."""

    mode = "offline_fixture"

    def __init__(self, config: Config, scenario: str, seed: int):
        if scenario not in {item.name for item in SCENARIOS}:
            raise ValueError(f"Unknown fixture scenario: {scenario}")
        self.config, self.scenario, self.seed = config, scenario, seed
        self.index = -1
        self.clock = 1.0
        self.viewpoint: Viewpoint | None = None

    def move_to(self, viewpoint: Viewpoint) -> MotionResult:
        self.index += 1
        self.viewpoint = viewpoint
        self.clock += 1.0
        if self.scenario == "planning_failure" and self.index == 0:
            return MotionResult(False, "plan", "injected_planning_failure")
        if self.scenario == "execution_failure":
            return MotionResult(False, "execute", "injected_execution_failure")
        return MotionResult(True, "fixture", "fixture_pose_selected", self.clock,
                            viewpoint.camera_pose, {"scope": "test_double", "robot_motion_verified": False})

    def capture(self, after_stamp: float, timeout: float) -> Observation:
        if self.scenario == "camera_timeout":
            raise AcquisitionError("injected_camera_timeout")
        if self.viewpoint is None:
            raise AcquisitionError("no_fixture_view_selected")
        self.clock += 0.1
        stamp = after_stamp - 0.1 if self.scenario == "stale_image" else self.clock
        frame = "incorrect_optical_frame" if self.scenario == "wrong_frame" else self.config.optical_frame
        return Observation(fixture_image(self.scenario, self.index, self.seed, self.config.expected_id),
                           stamp, frame, self.viewpoint.camera_pose)

    def close(self) -> None:
        pass
