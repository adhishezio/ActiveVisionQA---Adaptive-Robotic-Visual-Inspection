"""Regression checks for reporting and evidence-based simulator validation."""

from dataclasses import replace
import json

import pytest

from activevisionqa.cli import run_fixture
from activevisionqa.config import Config
from activevisionqa.report import render_report
from activevisionqa.validation import audit_run


def test_report_preserves_recorded_outcome_and_escapes_text(tmp_path):
    directory = tmp_path / "run"
    result = run_fixture(Config(), "occluded", 42, directory)
    metadata = json.loads((directory / "run.json").read_text())
    metadata["scenario"] = "<script>alert('x')</script>"
    (directory / "run.json").write_text(json.dumps(metadata))
    report = render_report(directory).read_text()
    assert f"ActiveVisionQA · {result['outcome']}" in report
    assert "view_01.png" in report and "view_02.png" in report
    assert "offline_fixture" in report
    assert "<script>" not in report and "&lt;script&gt;" in report
    assert "https://" not in report


def test_fixture_outcome_cannot_pass_as_robot_validation(tmp_path):
    directory = tmp_path / "run"
    run_fixture(Config(), "clear", 42, directory)
    audit = audit_run(directory, "ACCEPT", 0, 1)
    assert not audit["passed"]
    assert "not_simulator_evidence" in audit["errors"]


def _recorded_simulator_case(directory):
    """A minimal schema specimen; this helper never claims to move a robot."""
    directory.mkdir()
    (directory / "run.json").write_text(json.dumps({"config": Config().to_dict(), "simulation_only": True}))
    (directory / "result.json").write_text(json.dumps({"backend": "gazebo_moveit", "outcome": "ACCEPT", "views": 1, "motion_attempts": 1}))
    (directory / "view.png").write_bytes(b"schema-specimen")
    events = [
        {"sequence": 0, "stage": "MOTION_RESULT", "viewpoint": "view_1", "motion": {
            "ok": True, "phase": "verify", "arrived_at": 1., "checks": {
                "robot_motion_verified": True, "joint_limits_checked": True, "collision_checked": True,
                "state_validity_samples": 2, "planned_joint_trajectory": {"points": [0]},
                "position_error_m": .001, "orientation_error_rad": .01}}},
        {"sequence": 1, "stage": "ASSESS", "viewpoint": "view_1", "image": "view.png", "image_stamp": 2.,
         "optical_frame": Config().optical_frame, "actual_camera_pose": {"frame_id": "world"}},
        {"sequence": 2, "stage": "FINISH"},
    ]
    return events


@pytest.mark.parametrize("fault,reason", [(None, None), ("exit", "process_exit_mismatch"),
    ("stale", "image_not_newer_than_verified_motion"), ("collision", "missing_motion_checks"),
    ("image", "missing_image_artifact"), ("pose", "motion_pose_outside_tolerance")])
def test_simulation_audit_requires_underlying_evidence(tmp_path, fault, reason):
    events = _recorded_simulator_case(tmp_path / "run")
    if fault == "stale":
        events[1]["image_stamp"] = 1.
    if fault == "collision":
        events[0]["motion"]["checks"]["collision_checked"] = False
    if fault == "image":
        events[1]["image"] = "missing.png"
    if fault == "pose":
        events[0]["motion"]["checks"]["position_error_m"] = .5
    (tmp_path / "run/events.jsonl").write_text("\n".join(map(json.dumps, events))+"\n")
    audit = audit_run(tmp_path / "run", "ACCEPT", 2 if fault == "exit" else 0, 1)
    assert audit["passed"] == (fault is None)
    if reason:
        assert reason in audit["errors"]


@pytest.mark.parametrize("kwargs", [{"workspace_min": (-1., -1.)}, {"workspace_max": ()},
                                     {"max_views": 1.5}, {"expected_id": 7.5},
                                     {"ik_seed_positions": (0.,)}, {"ik_seed_positions": (float("nan"),)*6}])
def test_configuration_cannot_silently_truncate_safety_dimensions(kwargs):
    with pytest.raises(ValueError):
        replace(Config(), **kwargs)
