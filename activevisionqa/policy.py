"""Bounded reinspection supervisor, independent of ROS and simulator APIs."""

import math
import time
from typing import cast

from .config import Config
from .geometry import pose_error
from .models import AcquisitionError, ActiveVisionBackend, Backend, InspectionResult, Outcome, Viewpoint
from .recording import RunRecorder
from .view_selection import SelectionParameters, select_view


def validate_viewpoint(viewpoint: Viewpoint, config: Config) -> None:
    """Validate a proposed camera goal; whole-robot checks belong to planning."""
    pose = viewpoint.camera_pose
    if pose.frame_id != config.world_frame:
        raise ValueError("viewpoint_frame_mismatch")
    if any(not low <= coordinate <= high for coordinate, low, high in
           zip(pose.position, config.workspace_min, config.workspace_max)):
        raise ValueError("viewpoint_outside_workspace")


class InspectionRunner:
    """Try each candidate at most once; stop on unsafe/uncertain execution."""

    def __init__(self, config: Config, backend: Backend, inspector, recorder: RunRecorder):
        self.config = config
        self.backend = backend
        self.inspector = inspector
        self.recorder = recorder

    def run(self) -> InspectionResult:
        """Execute a bounded inspection, preserving all abstentions and failures."""
        start = time.monotonic()
        views = attempts = 0
        outcome, reason = Outcome.ESCALATE, "view_budget_exhausted"
        pending = list(self.config.viewpoints())
        last_evidence = None
        deadline = start+self.config.max_episode_seconds
        self.recorder.event("START", backend=self.backend.mode, selection_policy=self.config.selection_policy)
        try:
            while attempts < self.config.max_views and pending:
                if time.monotonic() >= deadline:
                    reason = "episode_time_budget_exhausted"
                    break
                if self.config.selection_policy == "next_best_view" and attempts:
                    active_backend = cast(ActiveVisionBackend, self.backend)
                    context = active_backend.view_context()
                    self.recorder.event("VIEW_CONTEXT", context=context)
                    selected = select_view(pending, context, SelectionParameters.from_config(self.config),
                                           last_evidence, active_backend.preview_viewpoint, deadline)
                    self.recorder.event("VIEW_SELECTION", selected=selected.viewpoint.name if selected.viewpoint else None,
                                        reason=selected.reason, candidates=selected.candidates,
                                        evidence_reason=last_evidence.reason if last_evidence else None,
                                        views_used=views, motion_attempts=attempts, context_source=context.source)
                    if selected.viewpoint is None:
                        outcome = Outcome.ERROR if selected.fatal else Outcome.ESCALATE
                        reason = selected.reason
                        break
                    viewpoint = selected.viewpoint
                    pending.remove(viewpoint)
                else:
                    # A shared initial view makes fixed/sequence/NBV comparisons
                    # start with the same observation opportunity.
                    viewpoint = pending.pop(0)
                try:
                    validate_viewpoint(viewpoint, self.config)
                except ValueError as exc:
                    self.recorder.event("SAFETY_REJECT", viewpoint=viewpoint.name, reason=str(exc))
                    reason = str(exc)
                    continue
                attempts += 1
                self.recorder.event("MOTION_REQUEST", viewpoint=viewpoint)
                motion = self.backend.move_to(viewpoint)
                self.recorder.event("MOTION_RESULT", viewpoint=viewpoint.name, motion=motion)
                if not motion.ok:
                    reason = motion.reason
                    if motion.phase != "plan":
                        outcome = Outcome.ERROR
                        break
                    continue
                if motion.actual_camera_pose is None or not math.isfinite(motion.arrived_at):
                    outcome, reason = Outcome.ERROR, "missing_motion_evidence"
                    break
                try:
                    observation = self.backend.capture(motion.arrived_at, self.config.image_timeout)
                    if observation.frame_id != self.config.optical_frame:
                        raise AcquisitionError("image_frame_mismatch")
                    if not math.isfinite(observation.timestamp) or observation.timestamp <= motion.arrived_at:
                        raise AcquisitionError("image_not_newer_than_motion")
                    if observation.camera_pose.frame_id != self.config.world_frame:
                        raise AcquisitionError("observation_pose_frame_mismatch")
                except AcquisitionError as exc:
                    reason = str(exc)
                    self.recorder.event("ACQUISITION_FAILURE", viewpoint=viewpoint.name, reason=reason)
                    if self.config.selection_policy == "next_best_view":
                        # An unavailable/untrusted sensor is not evidence that
                        # moving to another location will restore it.
                        break
                    continue
                views += 1
                distance, angle = pose_error(observation.camera_pose, viewpoint.camera_pose)
                if distance > self.config.position_tolerance or angle > self.config.orientation_tolerance:
                    outcome, reason = Outcome.ERROR, "image_pose_drift_after_motion"
                    self.recorder.event("POSE_DRIFT", position_error_m=distance, orientation_error_rad=angle)
                    break
                image_path = self.recorder.image(observation.image, f"view_{views:02d}.png")
                inspection_image = observation.image
                roi_path = None
                if hasattr(self.inspector, "prepare_observation"):
                    inspection_image = self.inspector.prepare_observation(observation)
                    if inspection_image is not None:
                        roi_path = self.recorder.image(inspection_image, f"view_{views:02d}_roi.png")
                evidence = self.inspector.inspect(inspection_image)
                last_evidence = evidence
                self.recorder.event("ASSESS", viewpoint=viewpoint.name, image=image_path,
                                    image_stamp=observation.timestamp, optical_frame=observation.frame_id,
                                    actual_camera_pose=observation.camera_pose, evidence=evidence, roi_image=roi_path)
                self.recorder.event("ACTION_FEEDBACK", viewpoint=viewpoint.name,
                                    selection_policy=self.config.selection_policy,
                                    definitive_inspection_observed=evidence.outcome in (Outcome.ACCEPT, Outcome.REJECT),
                                    observed_outcome=evidence.outcome, reason=evidence.reason,
                                    unselected_candidate_outcomes="unknown")
                reason = evidence.reason
                if evidence.outcome in (Outcome.ACCEPT, Outcome.REJECT):
                    outcome = evidence.outcome
                    break
                if evidence.outcome != Outcome.REINSPECT:
                    outcome, reason = Outcome.ERROR, "invalid_perception_outcome"
                    break
                self.recorder.event("REINSPECT", reason=reason, views_used=views)
            else:
                outcome = Outcome.ESCALATE
        except Exception as exc:
            outcome, reason = Outcome.ERROR, f"{type(exc).__name__}: {exc}"
            self.recorder.event("EXCEPTION", reason=reason)
        result = InspectionResult(outcome, reason, views, attempts, time.monotonic()-start, self.backend.mode)
        self.recorder.event("FINISH", result=result)
        self.recorder.write_json("result.json", result.to_dict())
        return result
