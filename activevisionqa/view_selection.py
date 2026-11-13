"""Evidence-triggered, geometry/cost-aware next-view selection.

Scores are engineering utilities, not calibrated success probabilities or
information gain. The selector receives no scene name or ground-truth ID.
"""

from dataclasses import dataclass, fields
import math
import time

import numpy as np

from .config import Config
from .geometry import pose_matrix
from .models import Evidence, Pose, SceneBox, ViewContext, Viewpoint


@dataclass(frozen=True)
class Selection:
    viewpoint: Viewpoint | None
    reason: str
    candidates: list[dict]
    fatal: bool = False


@dataclass(frozen=True)
class SelectionParameters:
    """Geometry/policy inputs with no expected IDs, surface strings or scenario."""

    world_frame: str
    target: tuple[float, float, float]
    marking_kind: str
    marking_yaw_rad: float
    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    min_predicted_visibility: float
    max_candidate_previews: int
    view_visibility_weight: float
    view_resolution_weight: float
    view_duration_weight: float
    view_path_weight: float
    view_novelty_weight: float

    @classmethod
    def from_config(cls, config: Config) -> "SelectionParameters":
        """Expose only quantities allowed to influence view selection."""
        return cls(**{field.name: getattr(config, field.name) for field in fields(cls)})


def segment_hits_box(start, end, box: SceneBox) -> bool:
    """Conservative slab intersection of a world-frame segment and oriented box."""
    matrix = np.linalg.inv(pose_matrix(box.pose))
    a = (matrix @ np.r_[start, 1.])[:3]
    b = (matrix @ np.r_[end, 1.])[:3]
    direction = b-a
    half = np.asarray(box.size)/2
    near, far = 1e-5, 1.-1e-5
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if abs(a[axis]) > half[axis]:
                return False
        else:
            x, y = (-half[axis]-a[axis])/direction[axis], (half[axis]-a[axis])/direction[axis]
            near, far = max(near, min(x, y)), min(far, max(x, y))
            if near > far:
                return False
    return True


def label_points(config: SelectionParameters) -> tuple[np.ndarray, np.ndarray]:
    """Known fixture footprint and 25 target rays, independent of label content."""
    width, height = .15, .06 if config.marking_kind == "text" else .15
    corners = np.array([[-width/2, height/2], [width/2, height/2],
                        [width/2, -height/2], [-width/2, -height/2]])
    grid = np.array([(x, y) for y in np.linspace(-height/2, height/2, 5)
                     for x in np.linspace(-width/2, width/2, 5)])
    c, s = math.cos(config.marking_yaw_rad), math.sin(config.marking_yaw_rad)
    rotation = np.array([[c, s], [-s, c]])
    def world(points):
        xy = points @ rotation + np.array(config.target[:2])
        return np.c_[xy, np.full(len(points), config.target[2]+.00016)]
    return world(corners), world(grid)


def visibility_features(viewpoint: Viewpoint, context: ViewContext, config: SelectionParameters) -> dict:
    """Estimate known-box visibility and projected label size for a camera goal."""
    pose = viewpoint.camera_pose
    if pose.frame_id != config.world_frame or context.camera_pose.frame_id != config.world_frame:
        raise ValueError("View-selection frame mismatch")
    if any(box.pose.frame_id != config.world_frame for box in context.boxes):
        raise ValueError("Scene-box frame mismatch")
    if any(not low <= value <= high for value, low, high in zip(pose.position, config.workspace_min, config.workspace_max)):
        return {"eligible": False, "reason": "viewpoint_outside_workspace"}
    corners, points = label_points(config)
    camera = np.asarray(pose.position)
    delta = camera-np.asarray(config.target)
    if delta[2] <= 0:
        return {"eligible": False, "reason": "label_not_front_facing"}
    transform = np.linalg.inv(pose_matrix(pose))
    optical = np.c_[corners, np.ones(4)] @ transform.T
    if np.any(optical[:, 2] <= .001):
        return {"eligible": False, "reason": "label_behind_camera"}
    k = np.asarray(context.intrinsics, float).reshape(3, 3)
    if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid view-selection camera intrinsics")
    pixels = optical[:, :3] @ k.T
    pixels = pixels[:, :2]/pixels[:, 2:3]
    width, height = context.image_size
    if np.any(pixels < 0) or np.any(pixels[:, 0] >= width) or np.any(pixels[:, 1] >= height):
        return {"eligible": False, "reason": "label_outside_camera_image"}
    blocked = []
    names = set()
    for point in points:
        hits = [box.name for box in context.boxes if segment_hits_box(camera, point, box)]
        blocked.append(bool(hits))
        names.update(hits)
    fraction = 1.-sum(blocked)/len(points)
    edge = min(float(np.linalg.norm(pixels[(i+1) % 4]-pixels[i])) for i in range(4))
    incidence = float(delta[2]/np.linalg.norm(delta))
    old = np.asarray(context.camera_pose.position)-np.asarray(config.target)
    angle = math.acos(float(np.clip(np.dot(old, delta)/max(np.linalg.norm(old)*np.linalg.norm(delta), 1e-9), -1, 1)))
    return {"eligible": fraction >= config.min_predicted_visibility,
            "reason": "visibility_candidate" if fraction >= config.min_predicted_visibility else "predicted_occlusion",
            "predicted_visible_fraction": fraction, "target_rays": len(points),
            "blocking_objects": sorted(names), "projected_min_label_edge_px": edge,
            "incidence_cosine": incidence, "view_change_rad": angle,
            "camera_displacement_m": float(np.linalg.norm(camera-np.asarray(context.camera_pose.position)))}


def select_view(candidates: list[Viewpoint], context: ViewContext, config: SelectionParameters,
                evidence: Evidence | None, preview, deadline: float) -> Selection:
    """Rank useful candidates with non-executing plans; execution must replan."""
    evaluations = []
    geometric = []
    for viewpoint in candidates:
        features = visibility_features(viewpoint, context, config)
        row = {"viewpoint": viewpoint.name, "camera_pose": viewpoint.camera_pose.to_dict(), **features}
        evaluations.append(row)
        if features["eligible"]:
            geometric.append((viewpoint, row))
    # Geometry shortlists independently of candidate naming/order. Plan at most
    # the configured count, preferring visibility and projected resolution.
    geometric.sort(key=lambda pair: (-pair[1]["predicted_visible_fraction"], -pair[1]["projected_min_label_edge_px"], pair[0].name))
    ranked = []
    for index, (viewpoint, row) in enumerate(geometric):
        if index >= config.max_candidate_previews:
            row.update(eligible=False, reason="preview_budget_exhausted")
            continue
        if time.monotonic() >= deadline:
            return Selection(None, "episode_time_budget_exhausted", evaluations)
        plan = preview(viewpoint)
        row["preview"] = {"ok": plan.ok, "phase": plan.phase, "reason": plan.reason, "checks": plan.checks}
        if not plan.ok:
            row.update(eligible=False, reason=plan.reason)
            if plan.phase != "plan":
                return Selection(None, plan.reason, evaluations, fatal=True)
            continue
        if plan.phase != "plan" or plan.checks.get("robot_motion_verified"):
            return Selection(None, "preview_must_not_execute_motion", evaluations, fatal=True)
        if (not plan.checks.get("joint_limits_checked") or not plan.checks.get("collision_checked")
                or plan.checks.get("state_validity_samples", 0) < 1):
            return Selection(None, "missing_preview_safety_checks", evaluations, fatal=True)
        duration = plan.checks.get("planned_duration_s")
        path = plan.checks.get("planned_joint_path_length_rad")
        if duration is None or path is None or not math.isfinite(duration) or not math.isfinite(path) or min(duration, path) < 0:
            return Selection(None, "invalid_preview_motion_cost", evaluations, fatal=True)
        resolution_weight = config.view_resolution_weight
        novelty_weight = config.view_novelty_weight
        if evidence and evidence.reason == "marking_too_small":
            resolution_weight *= 2
        elif evidence and evidence.reason in ("insufficient_image_quality", "character_outside_reference_gate", "text_format_mismatch"):
            novelty_weight *= 2
        score = (config.view_visibility_weight*row["predicted_visible_fraction"]
                 + resolution_weight*min(row["projected_min_label_edge_px"]/150., 1.)
                 + .25*row["incidence_cosine"] + novelty_weight*row["view_change_rad"]/math.pi
                 - config.view_duration_weight*duration - config.view_path_weight*path)
        row.update(score=float(score), reason="ranked_feasible_candidate", effective_resolution_weight=resolution_weight,
                   effective_novelty_weight=novelty_weight)
        ranked.append((score, viewpoint))
    if time.monotonic() >= deadline:
        return Selection(None, "episode_time_budget_exhausted", evaluations)
    if not ranked:
        return Selection(None, "no_feasible_informative_view", evaluations)
    ranked.sort(key=lambda pair: (-pair[0], pair[1].name))
    return Selection(ranked[0][1], "visibility_and_motion_cost", evaluations)
