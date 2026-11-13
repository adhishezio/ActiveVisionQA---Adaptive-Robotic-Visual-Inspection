"""Explicit interfaces shared by policy, perception and motion backends."""

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Protocol

import numpy as np


class Outcome(str, Enum):
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"
    REINSPECT = "REINSPECT"
    ESCALATE = "ESCALATE"
    ERROR = "ERROR"


@dataclass(frozen=True)
class Pose:
    """Position in meters and quaternion x/y/z/w, relative to frame_id."""

    position: tuple[float, float, float]
    orientation: tuple[float, float, float, float]
    frame_id: str = "world"

    def __post_init__(self) -> None:
        if len(self.position) != 3 or len(self.orientation) != 4:
            raise ValueError("Pose requires a 3-vector and a quaternion")
        if not self.frame_id or not all(math.isfinite(v) for v in (*self.position, *self.orientation)):
            raise ValueError("Pose needs a frame and finite coordinates")
        norm = math.sqrt(sum(v * v for v in self.orientation))
        if not math.isclose(norm, 1.0, abs_tol=1e-5):
            raise ValueError("Pose quaternion must be normalized")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Viewpoint:
    """A proposed camera optical-frame pose; a planner must validate motion."""

    name: str
    camera_pose: Pose


@dataclass
class Observation:
    """A camera image with simulation timestamp and measured optical pose."""

    image: np.ndarray
    timestamp: float
    frame_id: str
    camera_pose: Pose


@dataclass(frozen=True)
class Evidence:
    outcome: Outcome
    reason: str
    identifiers: tuple[int, ...] = ()
    metrics: dict[str, float] = field(default_factory=dict)
    text: str | None = None


@dataclass(frozen=True)
class MotionResult:
    ok: bool
    phase: str
    reason: str
    arrived_at: float = 0.0
    actual_camera_pose: Pose | None = None
    checks: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneBox:
    """Known MoveIt world box in meters; its name is only for traceability."""

    name: str
    pose: Pose
    size: tuple[float, float, float]

    def __post_init__(self):
        if len(self.size) != 3 or not all(math.isfinite(v) and v > 0 for v in self.size):
            raise ValueError("Scene boxes require finite positive dimensions")


@dataclass(frozen=True)
class ViewContext:
    """Measured camera pose/calibration and known world geometry; no task labels."""

    camera_pose: Pose
    intrinsics: tuple[float, ...]
    image_size: tuple[int, int]
    boxes: tuple[SceneBox, ...]
    source: str
    simulation_time: float = 0.0  # clock at retrieval; camera TF is the latest available


@dataclass(frozen=True)
class InspectionResult:
    outcome: Outcome
    reason: str
    views: int
    motion_attempts: int
    elapsed_seconds: float
    backend: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["outcome"] = self.outcome.value
        return data


class AcquisitionError(RuntimeError):
    """A bounded image-acquisition failure, not a negative inspection."""


class Backend(Protocol):
    mode: str

    def move_to(self, viewpoint: Viewpoint) -> MotionResult: ...

    def capture(self, after_stamp: float, timeout: float) -> Observation: ...

    def close(self) -> None: ...


class ActiveVisionBackend(Backend, Protocol):
    """Additional read/plan-only capabilities required by next-view selection."""

    def view_context(self) -> ViewContext: ...

    def preview_viewpoint(self, viewpoint: Viewpoint) -> MotionResult: ...
