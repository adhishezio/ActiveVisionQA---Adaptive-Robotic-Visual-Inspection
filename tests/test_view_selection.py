"""Geometry, bounded action selection and sensor/execution failure contracts."""

from dataclasses import replace
import json
import math
import time

import numpy as np
import pytest

from activevisionqa.assets import scene_boxes
from activevisionqa.config import Config
from activevisionqa.models import AcquisitionError, Evidence, MotionResult, Observation, Outcome, Pose, SceneBox, ViewContext
from activevisionqa.policy import InspectionRunner
from activevisionqa.recording import RunRecorder
from activevisionqa.view_selection import SelectionParameters, segment_hits_box, select_view, visibility_features
from activevisionqa.active_benchmark import matching_starts


def active_config(**changes):
    base = Config.load("config/active_inspection.json")
    return replace(base, **changes)


def context(config, scenario="all_occluded"):
    boxes = tuple(SceneBox(b["name"], Pose(tuple(b["position"]), (0., 0., 0., 1.)), tuple(b["size"]))
                  for b in scene_boxes(scenario, config))
    return ViewContext(config.viewpoints()[0].camera_pose, (692., 0., 400., 0., 692., 300., 0., 0., 1.),
                       (800, 600), boxes, "unit_box_scene")


def preview_cost(duration=1., path=1., ok=True, phase="plan"):
    return MotionResult(ok, phase, "unit_preview", checks={
        "joint_limits_checked": True, "collision_checked": True, "state_validity_samples": 3,
        "planned_duration_s": duration, "planned_joint_path_length_rad": path,
        "robot_motion_verified": False})


def test_segment_box_intersection_handles_parallel_rotated_and_endpoint_cases():
    box = SceneBox("b", Pose((0., 0., 0.), (0., 0., 0., 1.)), (2., 2., 2.))
    assert segment_hits_box((-2, 0, 0), (2, 0, 0), box)
    assert not segment_hits_box((-2, 2, 0), (2, 2, 0), box)
    assert not segment_hits_box((-2, 0, 0), (-1.1, 0, 0), box)
    rotated = SceneBox("r", Pose((0., 0., 0.), (0., 0., math.sin(math.pi/8), math.cos(math.pi/8))), (2., .2, .2))
    assert segment_hits_box((-1, -1, 0), (1, 1, 0), rotated)
    assert not segment_hits_box((-1, 1, .3), (1, -1, .3), rotated)


def test_visibility_changes_with_geometry_not_scene_or_identifier_labels():
    config = active_config()
    parameters = SelectionParameters.from_config(config)
    views = config.viewpoints()
    known = context(config)
    assert visibility_features(views[1], known, parameters)["predicted_visible_fraction"] < .5
    assert visibility_features(views[2], known, parameters)["predicted_visible_fraction"] == 1.
    modified = SelectionParameters.from_config(replace(config, expected_text="XYZ-9999", surface_text="ABC-1111"))
    assert modified == parameters and not hasattr(modified, "expected_text")
    renamed = replace(known, boxes=tuple(replace(b, name="unrelated_name") for b in known.boxes))
    assert visibility_features(views[2], renamed, parameters)["predicted_visible_fraction"] == 1.
    more_boxes = context(config, "three_sides_occluded")
    assert not visibility_features(views[2], more_boxes, parameters)["eligible"]
    roof = context(config, "top_occluded")
    assert not visibility_features(views[4], roof, parameters)["eligible"]
    assert visibility_features(views[2], roof, parameters)["eligible"]


def test_selection_filters_blocked_view_before_motion_preview():
    config = active_config()
    called = []
    def preview(view):
        called.append(view.name)
        return preview_cost()
    selected = select_view(list(config.viewpoints()[1:3]), context(config), SelectionParameters.from_config(config),
                           Evidence(Outcome.REINSPECT, "marking_not_detected"), preview, time.monotonic()+30)
    assert selected.viewpoint.name == "view_3" and called == ["view_3"]
    assert selected.candidates[0]["reason"] == "predicted_occlusion"


def test_motion_cost_affects_choice_among_visible_alternatives():
    config = active_config(view_visibility_weight=0., view_resolution_weight=0., view_novelty_weight=0.,
                           view_path_weight=0., view_duration_weight=1.)
    candidates = list(config.viewpoints()[2:4])
    selected = select_view(candidates, context(config, "clear"), SelectionParameters.from_config(config), None,
                           lambda view: preview_cost(duration=1. if view.name == "view_4" else 10.), time.monotonic()+30)
    assert selected.viewpoint.name == "view_4"


def test_fully_covered_scene_escalates_without_previewing_more_motion():
    config = active_config()
    selected = select_view(list(config.viewpoints()[1:]), context(config, "covered"), SelectionParameters.from_config(config),
                           Evidence(Outcome.REINSPECT, "insufficient_text_contrast"),
                           lambda view: pytest.fail("Fully blocked candidate was planned"), time.monotonic()+30)
    assert selected.viewpoint is None and selected.reason == "no_feasible_informative_view"


@pytest.mark.parametrize("fault", ["execution", "unchecked", "nan_cost"])
def test_invalid_preview_cannot_authorize_a_view(fault):
    config = active_config()
    plan = preview_cost()
    if fault == "execution":
        plan = replace(plan, phase="execute")
    elif fault == "unchecked":
        plan.checks["collision_checked"] = False
    else:
        plan.checks["planned_duration_s"] = float("nan")
    selected = select_view([config.viewpoints()[2]], context(config), SelectionParameters.from_config(config), None,
                           lambda view: plan, time.monotonic()+30)
    assert selected.fatal and selected.viewpoint is None


def test_preview_and_time_budgets_are_bounded():
    config = active_config(max_candidate_previews=1)
    called = []
    def preview(view):
        called.append(view.name)
        return preview_cost()
    select_view(list(config.viewpoints()[1:]), context(config, "clear"), SelectionParameters.from_config(config),
                None, preview, time.monotonic()+30)
    assert len(called) == 1
    expired = select_view([config.viewpoints()[2]], context(config), SelectionParameters.from_config(config), None,
                          lambda view: pytest.fail("Expired episode planned motion"), time.monotonic()-1)
    assert expired.viewpoint is None and expired.reason == "episode_time_budget_exhausted"


def test_fresh_world_comparison_rejects_mismatched_start_states():
    records = [{"episode": {"initial_joint_state": {"a": 0., "b": .2}}},
               {"episode": {"initial_joint_state": {"a": .001, "b": .201}}}]
    assert matching_starts(records)
    records[1]["episode"]["initial_joint_state"]["a"] = .1
    assert not matching_starts(records)


class FeedbackBackend:
    """Explicit test double; no claim of actual robot movement from these tests."""
    mode = "unit_feedback_backend"
    def __init__(self, config, camera_failure=False, execution_failure=False):
        self.config = config
        self.camera_failure, self.execution_failure = camera_failure, execution_failure
        self.moves, self.previews = [], []
        self.current = None
    def move_to(self, view):
        self.moves.append(view.name)
        self.current = view
        if self.execution_failure and len(self.moves) > 1:
            return MotionResult(False, "execute", "unit_execution_failure")
        return MotionResult(True, "verify", "unit_pose", float(len(self.moves)), view.camera_pose)
    def capture(self, after_stamp, timeout):
        if self.camera_failure:
            raise AcquisitionError("unit_camera_timeout")
        return Observation(np.zeros((20, 20, 3), np.uint8), after_stamp+.1,
                           self.config.optical_frame, self.current.camera_pose)
    def view_context(self):
        return context(self.config)
    def preview_viewpoint(self, view):
        self.previews.append(view.name)
        return preview_cost(duration=.1 if view.name == "view_3" else 10.)
    def close(self):
        pass


class FeedbackInspector:
    def __init__(self, clear_first=False):
        self.count = 0
        self.clear_first = clear_first
    def inspect(self, image):
        self.count += 1
        return Evidence(Outcome.ACCEPT, "unit_readable") if self.clear_first or self.count > 1 else Evidence(Outcome.REINSPECT, "unit_blocked")


@pytest.mark.parametrize("failure,expected,move_count", [(None, Outcome.ACCEPT, 2),
    ("camera", Outcome.ESCALATE, 1), ("execution", Outcome.ERROR, 2), ("clear_first", Outcome.ACCEPT, 1)])
def test_closed_loop_uses_feedback_and_stops_on_sensor_or_execution_faults(tmp_path, failure, expected, move_count):
    config = active_config()
    backend = FeedbackBackend(config, camera_failure=failure == "camera", execution_failure=failure == "execution")
    with RunRecorder(tmp_path / "run", {}) as recorder:
        result = InspectionRunner(config, backend, FeedbackInspector(clear_first=failure == "clear_first"), recorder).run()
    assert result.outcome == expected and len(backend.moves) == move_count
    if failure in ("camera", "clear_first"):
        assert not backend.previews
    else:
        assert backend.moves == ["view_1", "view_3"]
    events = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text().splitlines()]
    assert all(e["unselected_candidate_outcomes"] == "unknown" for e in events if e["stage"] == "ACTION_FEEDBACK")
