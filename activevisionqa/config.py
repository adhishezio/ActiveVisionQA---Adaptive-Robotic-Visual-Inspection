"""Validated configuration, shared with the simulator's workcell generator."""

from dataclasses import dataclass
import json
import math
import re
from pathlib import Path
from typing import Any

from .geometry import optical_look_at
from .models import Viewpoint

POLICY_FIELDS = ("max_views", "selection_policy", "max_episode_seconds", "max_candidate_previews",
                 "min_predicted_visibility", "view_visibility_weight", "view_resolution_weight",
                 "view_duration_weight", "view_path_weight", "view_novelty_weight")


@dataclass(frozen=True)
class Config:
    selection_policy: str = "sequence"
    max_episode_seconds: float = 60.
    max_candidate_previews: int = 4
    min_predicted_visibility: float = .95
    view_visibility_weight: float = 4.
    view_resolution_weight: float = 1.
    view_duration_weight: float = .08
    view_path_weight: float = .03
    view_novelty_weight: float = .2
    marking_kind: str = "aruco"
    expected_text: str = "AVQ-0427"
    surface_text: str | None = None
    marking_yaw_rad: float = 0.0
    light_scale: float = 1.0
    expected_id: int = 7
    marker_dictionary: str = "DICT_4X4_50"
    min_marker_edge_px: float = 55.0
    min_sharpness: float = 45.0
    max_views: int = 2
    image_timeout: float = 6.0
    world_frame: str = "world"
    optical_frame: str = "inspection_camera_optical_frame"
    camera_topic: str = "/inspection/camera/image_raw"
    camera_info_topic: str = "/inspection/camera/camera_info"
    group_name: str = "ur_manipulator"
    tool_frame: str = "tool0"
    velocity_scale: float = 0.15
    acceleration_scale: float = 0.15
    planning_seconds: float = 8.0
    execution_timeout: float = 45.0
    position_tolerance: float = 0.025
    orientation_tolerance: float = 0.12
    # Numerical IK seed only, never a commanded pose. UR joint order is
    # shoulder pan/lift, elbow, wrist 1/2/3; values are radians.
    ik_seed_positions: tuple[float, ...] = (-.74, -1.33, .92, -.78, -1.18, -.81)
    target: tuple[float, float, float] = (0.50, 0.0, 0.242)
    camera_positions: tuple[tuple[float, float, float], ...] = ((0.50, -0.18, 0.54), (0.50, 0.18, 0.54))
    workspace_min: tuple[float, float, float] = (-0.80, -0.80, 0.10)
    workspace_max: tuple[float, float, float] = (0.85, 0.80, 1.00)

    def __post_init__(self) -> None:
        if self.selection_policy not in ("sequence", "next_best_view"):
            raise ValueError("selection_policy must be sequence or next_best_view")
        if type(self.max_candidate_previews) is not int or not 1 <= self.max_candidate_previews <= 5:
            raise ValueError("max_candidate_previews must be between 1 and 5")
        if not math.isfinite(self.max_episode_seconds) or self.max_episode_seconds <= 0:
            raise ValueError("max_episode_seconds must be finite and positive")
        if not math.isfinite(self.min_predicted_visibility) or not 0 < self.min_predicted_visibility <= 1:
            raise ValueError("min_predicted_visibility must be in (0, 1]")
        weights = (self.view_visibility_weight, self.view_resolution_weight, self.view_duration_weight,
                   self.view_path_weight, self.view_novelty_weight)
        if not all(math.isfinite(v) and v >= 0 for v in weights):
            raise ValueError("View-selection weights must be finite and nonnegative")
        if self.marking_kind not in ("aruco", "text"):
            raise ValueError("marking_kind must be aruco or text")
        if not isinstance(self.expected_text, str):
            raise ValueError("expected_text must be a string")
        for text in (self.expected_text, self.surface_text):
            if text is not None and (not isinstance(text, str) or not re.fullmatch(r"[A-Z0-9-]{4,12}", text)):
                raise ValueError("Text markings require 4..12 uppercase ASCII letters, digits or hyphens")
        if not math.isfinite(self.marking_yaw_rad) or abs(self.marking_yaw_rad) > math.pi:
            raise ValueError("Marking yaw must be finite radians in [-pi, pi]")
        if not math.isfinite(self.light_scale) or not .2 <= self.light_scale <= 1.5:
            raise ValueError("light_scale must be between 0.2 and 1.5")
        if type(self.expected_id) is not int or not 0 <= self.expected_id < 50 or self.marker_dictionary != "DICT_4X4_50":
            raise ValueError("The baseline uses DICT_4X4_50 and identifiers 0..49")
        if type(self.max_views) is not int or not 1 <= self.max_views <= len(self.camera_positions) <= 5:
            raise ValueError("max_views must fit the bounded candidate list (1..5)")
        numeric = (self.min_marker_edge_px, self.min_sharpness, self.image_timeout,
                   self.planning_seconds, self.execution_timeout, self.position_tolerance,
                   self.orientation_tolerance)
        if any(not math.isfinite(v) or v <= 0 for v in numeric):
            raise ValueError("Thresholds and timeouts must be finite and positive")
        if not 0 < self.velocity_scale <= 0.3 or not 0 < self.acceleration_scale <= 0.3:
            raise ValueError("Demo motion scales must be positive and at most 0.3")
        if len(self.workspace_min) != 3 or len(self.workspace_max) != 3:
            raise ValueError("Workspace bounds must be 3-vectors")
        for low, high in zip(self.workspace_min, self.workspace_max):
            if not math.isfinite(low+high) or low >= high:
                raise ValueError("Invalid workspace bounds")
        for point in self.camera_positions:
            if len(point) != 3 or any(not math.isfinite(v) for v in point):
                raise ValueError("Candidate camera positions must be finite 3-vectors")
        if len(self.target) != 3 or not all(math.isfinite(v) for v in self.target) or self.target[2] <= .08:
            raise ValueError("The part target must be a finite 3-vector above the floor")
        if not all((self.world_frame, self.optical_frame, self.tool_frame, self.group_name)):
            raise ValueError("Frame and group names cannot be empty")
        if len(self.ik_seed_positions) != 6 or not all(math.isfinite(v) for v in self.ik_seed_positions):
            raise ValueError("The UR IK seed must contain six finite joint positions")

    def viewpoints(self) -> tuple[Viewpoint, ...]:
        """Return deterministic camera optical goals, expressed in meters."""
        return tuple(Viewpoint(f"view_{i+1}", optical_look_at(point, self.target, self.world_frame))
                     for i, point in enumerate(self.camera_positions))

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path | None) -> "Config":
        """Read a JSON config, rejecting unknown fields and invalid values."""
        if path is None:
            return cls()
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        """Validate an in-memory config using the same rules as JSON loading."""
        data = dict(data)
        for key in ("target", "workspace_min", "workspace_max", "ik_seed_positions"):
            if key in data:
                data[key] = tuple(data[key])
        if "camera_positions" in data:
            data["camera_positions"] = tuple(tuple(p) for p in data["camera_positions"])
        return cls(**data)
