"""Known-value policy/perception/geometry tests; no robot claims from fixtures."""

from dataclasses import replace
import json

import numpy as np
import pytest

from activevisionqa.cli import run_fixture
from activevisionqa.config import Config
from activevisionqa.evaluation import summarize
from activevisionqa.fixtures import fixture_image
from activevisionqa.geometry import optical_look_at, pose_error, pose_matrix, pose_from_matrix
from activevisionqa.models import Outcome, Pose, Viewpoint
from activevisionqa.perception import MarkerInspector
from activevisionqa.policy import validate_viewpoint
from activevisionqa.recording import RunRecorder


@pytest.mark.parametrize("scenario,views,expected", [
    ("clear", 1, "ACCEPT"), ("clear", 2, "ACCEPT"),
    ("occluded", 1, "ESCALATE"), ("occluded", 2, "ACCEPT"),
    ("small_then_clear", 1, "ESCALATE"), ("small_then_clear", 2, "ACCEPT"),
    ("blurred_then_clear", 1, "ESCALATE"), ("blurred_then_clear", 2, "ACCEPT"),
    ("wrong_id", 1, "REJECT"), ("wrong_id_occluded", 1, "ESCALATE"),
    ("wrong_id_occluded", 2, "REJECT"), ("all_occluded", 2, "ESCALATE"),
    ("planning_failure", 2, "ACCEPT"), ("execution_failure", 2, "ERROR"),
    ("camera_timeout", 2, "ESCALATE"), ("stale_image", 2, "ESCALATE"),
    ("wrong_frame", 2, "ESCALATE"),
])
def test_predefined_outcomes_and_budgets(tmp_path, scenario, views, expected):
    result = run_fixture(replace(Config(), max_views=views), scenario, 42, tmp_path / "run")
    assert result["outcome"] == expected
    assert result["views"] <= views
    assert result["motion_attempts"] <= views
    assert result["backend"] == "offline_fixture"
    events = [json.loads(line) for line in (tmp_path / "run/events.jsonl").read_text().splitlines()]
    assert events[0]["stage"] == "START"
    assert events[-1]["stage"] == "FINISH"
    assert [e["sequence"] for e in events] == list(range(len(events)))
    if scenario == "execution_failure":
        assert result["motion_attempts"] == 1
        assert result["views"] == 0


def test_fixture_replay_seed_is_reproducible():
    np.testing.assert_array_equal(fixture_image("clear", 0, 101), fixture_image("clear", 0, 101))
    assert not np.array_equal(fixture_image("clear", 0, 101), fixture_image("clear", 0, 102))


@pytest.mark.parametrize("image", [None, np.zeros((0, 0), np.uint8), np.zeros((48, 48, 4), np.uint8), np.zeros((48, 48), float)])
def test_invalid_images_abstain(image):
    assert MarkerInspector(Config()).inspect(image).outcome == Outcome.REINSPECT


def test_optical_frame_points_at_target_and_round_trips():
    for position in ((.5, -.18, .54), (.5, .18, .54), (0, 0, 1), (0, 0, -1)):
        pose = optical_look_at(position, (0, 0, 0))
        transform = pose_matrix(pose)
        expected = -np.asarray(position) / np.linalg.norm(position)
        np.testing.assert_allclose(transform[:3, 2], expected, atol=1e-8)
        reconstructed = pose_from_matrix(transform, "world")
        distance, angle = pose_error(pose, reconstructed)
        assert distance < 1e-9 and angle < 1e-6


def test_frame_and_workspace_rejections():
    config = Config()
    with pytest.raises(ValueError, match="frame_mismatch"):
        validate_viewpoint(Viewpoint("wrong", Pose((.5, 0, .5), (0, 0, 0, 1), "camera")), config)
    with pytest.raises(ValueError, match="outside_workspace"):
        validate_viewpoint(Viewpoint("far", Pose((5, 0, .5), (0, 0, 0, 1))), config)
    with pytest.raises(ValueError, match="normalized"):
        Pose((0, 0, 0), (0, 0, 0, 0))
    with pytest.raises(ValueError, match="different frames"):
        pose_error(Pose((0, 0, 0), (0, 0, 0, 1), "a"), Pose((0, 0, 0), (0, 0, 0, 1), "b"))


@pytest.mark.parametrize("kwargs", [{"max_views": 0}, {"max_views": 3}, {"expected_id": 50},
                                    {"image_timeout": -1}, {"velocity_scale": 1.0}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_evaluation_keeps_abstentions_in_denominators():
    records = [{"truth": "defective", "outcome": "ACCEPT", "views": 1},
               {"truth": "defective", "outcome": "ESCALATE", "views": 2},
               {"truth": "acceptable", "outcome": "REJECT", "views": 1},
               {"truth": "acceptable", "outcome": "ERROR", "views": 0}]
    metrics = summarize(records)
    assert metrics["false_accept_rate"] == .5
    assert metrics["false_reject_rate"] == .5
    assert metrics["escalation_fraction"] == .25
    assert metrics["error_fraction"] == .25


def test_recordings_do_not_overwrite_previous_work(tmp_path):
    with RunRecorder(tmp_path / "run", {"scope": "test"}) as recorder:
        recorder.event("TEST")
    with pytest.raises(FileExistsError):
        RunRecorder(tmp_path / "run", {})
