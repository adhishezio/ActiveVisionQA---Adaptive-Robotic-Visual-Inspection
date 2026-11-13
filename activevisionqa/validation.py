"""Audit persisted simulator evidence independently of its terminal outcome."""

import json
import math
from pathlib import Path


def audit_run(directory: str | Path, expected: str, exit_code: int, expected_views: int) -> dict:
    """Require measured motions and fresh images for the predefined smoke cases.

    This checks the integrity of our recorded evidence, not a safety certificate
    or independent proof of Gazebo's physical accuracy.
    """
    directory = Path(directory)
    errors = []
    try:
        result = json.loads((directory / "result.json").read_text())
        metadata = json.loads((directory / "run.json").read_text())
        events = [json.loads(line) for line in (directory / "events.jsonl").read_text().splitlines()]
        config = metadata["config"]
        if result["outcome"] != expected:
            errors.append("outcome_mismatch")
        if exit_code != (3 if expected == "ESCALATE" else 0):
            errors.append("process_exit_mismatch")
        if result["backend"] != "gazebo_moveit" or not metadata.get("simulation_only"):
            errors.append("not_simulator_evidence")
        if result["views"] != expected_views:
            errors.append("view_count_mismatch")
        if not events or events[-1]["stage"] != "FINISH":
            errors.append("run_not_finished")
        if [event["sequence"] for event in events] != list(range(len(events))):
            errors.append("event_sequence_broken")
        motions = [event for event in events if event["stage"] == "MOTION_RESULT"]
        verified = [event for event in motions if event["motion"]["ok"]]
        assessments = [event for event in events if event["stage"] == "ASSESS"]
        if len(motions) != result["motion_attempts"] or len(verified) != expected_views or len(assessments) != expected_views:
            errors.append("motion_or_image_evidence_count_mismatch")
        samples = 0
        positions = []
        for event in verified:
            motion = event["motion"]
            checks = motion["checks"]
            if (motion["phase"] != "verify" or not checks.get("robot_motion_verified")
                    or not checks.get("joint_limits_checked") or not checks.get("collision_checked")
                    or checks.get("state_validity_samples", 0) <= 0
                    or not checks.get("planned_joint_trajectory", {}).get("points")):
                errors.append("missing_motion_checks")
            samples += checks.get("state_validity_samples", 0)
            distance, angle = checks["position_error_m"], checks["orientation_error_rad"]
            positions.append(distance)
            if (not math.isfinite(distance) or not math.isfinite(angle)
                    or distance > config["position_tolerance"] or angle > config["orientation_tolerance"]):
                errors.append("motion_pose_outside_tolerance")
        for event in assessments:
            prior = [motion for motion in verified if motion["viewpoint"] == event["viewpoint"]
                     and motion["sequence"] < event["sequence"]]
            if not prior or not event["image_stamp"] > prior[-1]["motion"]["arrived_at"]:
                errors.append("image_not_newer_than_verified_motion")
            if (event["optical_frame"] != config["optical_frame"]
                    or event["actual_camera_pose"]["frame_id"] != config["world_frame"]):
                errors.append("image_frame_mismatch")
            image = directory / event["image"]
            if image.parent != directory or not image.is_file() or image.stat().st_size == 0:
                errors.append("missing_image_artifact")
        if expected_views == 0 and not any(not event["motion"]["ok"] and event["motion"]["phase"] == "plan" for event in motions):
            errors.append("missing_planning_failure")
        return {"passed": not errors, "errors": errors, "verified_motions": len(verified),
                "sampled_collision_states": samples, "max_position_error_m": max(positions, default=None)}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "errors": errors + [f"invalid_artifacts:{type(exc).__name__}:{exc}"]}
