"""Predeclared synthetic-text capture plans, provenance and split audits."""

from collections import Counter
import csv
from dataclasses import replace
import hashlib
import html
import json
from pathlib import Path
import re

import numpy as np

from .config import Config
from .markings import TEXT_FONT, surface_text
from .models import Pose
from .geometry import pose_error

SPLITS = ("train", "validation", "test")
SCENARIOS = ("clear", "occluded", "wrong_id", "all_occluded")


def write_json(path: Path, value) -> None:
    """Write a new artifact exclusively, with strict JSON numeric values."""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def file_sha256(path: Path) -> str:
    """Return the content hash used to audit captured-image provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_plan(seed: int = 123, parts_per_split: int = 1,
              scenarios: tuple[str, ...] = ("clear", "occluded")) -> dict:
    """Allocate part identities before capture; keep all their views together."""
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("seed must be an integer in [0, 2**31)")
    if type(parts_per_split) is not int or not 1 <= parts_per_split <= 20:
        raise ValueError("parts_per_split must be between 1 and 20")
    if not scenarios or len(set(scenarios)) != len(scenarios) or any(s not in SCENARIOS for s in scenarios):
        raise ValueError("Choose unique supported scenarios")
    rng = np.random.default_rng(seed)
    identifiers = iter(rng.choice(10000, parts_per_split*len(SPLITS)*2, replace=False).tolist())
    cases = []
    for split in SPLITS:
        for part in range(parts_per_split):
            expected, wrong = (f"AVQ-{next(identifiers):04d}" for _ in range(2))
            part_id = f"seed{seed}_{split}_part{part:03d}"
            yaw = float(rng.uniform(-.18, .18))
            light = float(rng.uniform(.65, 1.25))
            shift = rng.uniform([-.012, -.008, -.008], [.012, .008, .008])
            base = Config()
            target = tuple(float(v) for v in np.array(base.target)+shift)
            positions = tuple(tuple(float(v) for v in np.array(p)+shift) for p in base.camera_positions)
            for scenario in scenarios:
                actual = wrong if scenario == "wrong_id" else expected
                config = replace(base, marking_kind="text", expected_text=expected, surface_text=actual,
                                 marking_yaw_rad=yaw, light_scale=light, target=target, camera_positions=positions)
                cases.append({"case_id": f"{split}_p{part:03d}_{scenario}", "part_id": part_id,
                              "split": split, "scenario": scenario, "seed": seed*1000+len(cases),
                              "expected_text": expected, "surface_text": actual, "config": config.to_dict()})
    plan = {"schema_version": 1, "purpose": "synthetic text data collection; no trained OCR or industrial accuracy claim",
            "seed": seed, "parts_per_split": parts_per_split, "scenarios": list(scenarios),
            "split_unit": "part identity and all associated surface strings, scenarios and views",
            "font": TEXT_FONT, "readability_labels": "unreviewed; ground-truth string does not imply readability",
            "expected_images": sum(c["config"]["max_views"] for c in cases), "cases": cases}
    validate_plan(plan)
    return plan


def validate_plan(plan: dict) -> None:
    """Reject duplicate cases, label/config disagreement and cross-split leakage."""
    if plan.get("schema_version") != 1 or not plan.get("cases"):
        raise ValueError("Unsupported or empty dataset plan")
    ids = set()
    groups = {}
    texts = {}
    for case in plan["cases"]:
        name, split, group = case["case_id"], case["split"], case["part_id"]
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", name) or name in ids:
            raise ValueError("Invalid or duplicate case ID")
        ids.add(name)
        if split not in SPLITS or case["scenario"] not in SCENARIOS:
            raise ValueError("Invalid split or scenario")
        if not isinstance(group, str) or not group:
            raise ValueError("Missing part identity")
        if groups.setdefault(group, split) != split:
            raise ValueError("Part identity leaks across splits")
        config = Config.from_dict(case["config"])
        if config.marking_kind != "text" or config.expected_text != case["expected_text"] or surface_text(config, case["scenario"]) != case["surface_text"]:
            raise ValueError("Scene configuration and text ground truth disagree")
        for text in (case["expected_text"], case["surface_text"]):
            if texts.setdefault(text, split) != split:
                raise ValueError("Text identity leaks across splits")


def load_plan(path: Path) -> dict:
    """Read and validate a predeclared capture plan before any simulation starts."""
    plan = json.loads(path.read_text())
    validate_plan(plan)
    return plan


def audit_dataset(directory: Path, require_complete: bool = True) -> dict:
    """Check image integrity, timestamps, provenance and grouping of all captures."""
    import cv2
    errors = []
    counts = Counter()
    ids = set()
    hashes = {}
    seen_views = Counter()
    seen_viewpoints = set()
    try:
        plan = load_plan(directory / "plan.json")
        cases = {case["case_id"]: case for case in plan["cases"]}
        rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
        if not rows:
            errors.append("no_camera_samples")
        for row in rows:
            case = cases[row["case_id"]]
            config = Config.from_dict(case["config"])
            if row["sample_id"] in ids:
                errors.append("duplicate_sample_id")
            ids.add(row["sample_id"])
            view_key = (row["case_id"], row["viewpoint"])
            if view_key in seen_viewpoints:
                errors.append("duplicate_case_viewpoint")
            seen_viewpoints.add(view_key)
            for key in ("part_id", "split", "expected_text", "surface_text", "scenario", "seed"):
                if row[key] != case[key]:
                    errors.append("sample_plan_disagreement:"+key)
            if row["source"] != "gazebo_robot_camera" or not row["simulation_only"]:
                errors.append("invalid_capture_source")
            if row["readability_label"] is not None:
                errors.append("unexpected_automatic_readability_label")
            if row["label_matches_expected"] != (row["surface_text"] == row["expected_text"]):
                errors.append("invalid_identifier_ground_truth")
            if not np.isfinite(row["image_stamp"]) or not row["image_stamp"] > row["motion_arrived_at"]:
                errors.append("image_not_after_motion")
            if row["optical_frame"] != config.optical_frame:
                errors.append("wrong_optical_frame")
            pose = Pose(**row["camera_pose"])
            if pose.frame_id != config.world_frame:
                errors.append("wrong_pose_frame")
            desired = {v.name: v.camera_pose for v in config.viewpoints()[:config.max_views]}
            if row["viewpoint"] not in desired:
                errors.append("unknown_viewpoint")
            else:
                distance, angle = pose_error(pose, desired[row["viewpoint"]])
                if distance > config.position_tolerance or angle > config.orientation_tolerance:
                    errors.append("capture_pose_outside_tolerance")
            calibration = row["camera_info"]
            k = np.asarray(calibration["k"]).reshape(3, 3)
            if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
                errors.append("invalid_calibration")
            if calibration["frame_id"] != config.optical_frame:
                errors.append("wrong_calibration_frame")
            for field, digest in (("image", "image_sha256"), ("rectified_image", "rectified_sha256")):
                path = (directory / row[field]).resolve()
                if not path.is_relative_to(directory.resolve()) or not path.is_file():
                    errors.append("missing_or_unsafe_image_path")
                    continue
                if file_sha256(path) != row[digest]:
                    errors.append("image_hash_mismatch")
                image = cv2.imread(str(path))
                if image is None:
                    errors.append("invalid_image_artifact")
                if field == "image" and image is not None:
                    if image.shape[:2] != (calibration["height"], calibration["width"]):
                        errors.append("image_calibration_size_mismatch")
                    if hashes.setdefault(row[digest], row["split"]) != row["split"]:
                        errors.append("identical_raw_image_across_splits")
            motion = row["motion_checks"]
            position_error = motion.get("position_error_m", float("inf"))
            angle_error = motion.get("orientation_error_rad", float("inf"))
            if (not motion.get("robot_motion_verified") or not motion.get("collision_checked")
                    or not motion.get("joint_limits_checked") or motion.get("state_validity_samples", 0) < 1
                    or not np.isfinite(position_error) or not np.isfinite(angle_error)
                    or position_error > config.position_tolerance or angle_error > config.orientation_tolerance):
                errors.append("missing_or_invalid_motion_verification")
            event_path = (directory / row["events"]).resolve()
            if not event_path.is_relative_to(directory.resolve()):
                errors.append("unsafe_event_path")
            else:
                events = [json.loads(line) for line in event_path.read_text().splitlines()]
                if not events or events[-1]["stage"] != "FINISH":
                    errors.append("capture_run_not_finished")
                if [event["sequence"] for event in events] != list(range(len(events))):
                    errors.append("broken_event_sequence")
                captured = [e for e in events if e["stage"] == "CAPTURE" and e["sample"]["sample_id"] == row["sample_id"]]
                if len(captured) != 1:
                    errors.append("missing_capture_event")
                else:
                    original = captured[0]["sample"]
                    for key in ("image_sha256", "rectified_sha256", "image_stamp", "camera_pose", "camera_info", "surface_text", "motion_checks"):
                        if original[key] != row[key]:
                            errors.append("capture_event_disagreement:"+key)
                    prior = [e["motion"] for e in events if e["stage"] == "MOTION_RESULT" and e["viewpoint"] == row["viewpoint"] and e["sequence"] < captured[0]["sequence"]]
                    if (not prior or not prior[-1]["ok"] or prior[-1]["arrived_at"] != row["motion_arrived_at"]
                            or not prior[-1]["checks"].get("planned_joint_trajectory", {}).get("points")):
                        errors.append("missing_verified_trajectory_event")
            counts[row["split"]] += 1
            seen_views[row["case_id"]] += 1
            if seen_views[row["case_id"]] > config.max_views:
                errors.append("too_many_views_for_case")
        complete = all(seen_views[name] == c["config"]["max_views"] for name, c in cases.items())
        if require_complete and not complete:
            errors.append("incomplete_planned_views")
        return {"passed": not errors, "errors": sorted(set(errors)), "samples": len(rows),
                "samples_by_split": {split: counts[split] for split in SPLITS},
                "captured_cases": len(seen_views), "planned_cases": len(cases),
                "all_planned_views_captured": complete,
                "readability_reviewed": 0, "scope": "artifact integrity; not OCR performance or real-world validation"}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {"passed": False, "errors": errors+[f"invalid_dataset:{type(exc).__name__}:{exc}"]}


def write_review_template(directory: Path) -> Path:
    """Export blank manual readability labels separately from immutable captures."""
    rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    path = directory / "readability_review.csv"
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "split", "surface_text", "readability", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in ("sample_id", "split", "surface_text")})
    return path


def render_gallery(directory: Path) -> Path:
    """Create a local visual review page from actual saved camera samples."""
    rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
    esc = lambda value: html.escape(str(value), quote=True)
    cards = []
    for row in rows:
        cards.append(f"<article><h2>{esc(row['sample_id'])}</h2><p>Split: {esc(row['split'])} · "
                     f"scene: {esc(row['scenario'])}</p><img src='{esc(row['image'])}' alt='Robot camera capture'>"
                     f"<img src='{esc(row['rectified_image'])}' alt='Geometry-rectified label ROI'>"
                     f"<p>Surface: <strong>{esc(row['surface_text'])}</strong>; expected: {esc(row['expected_text'])}. "
                     f"Readability: <strong>unreviewed</strong>.</p></article>")
    path = directory / "gallery.html"
    path.write_text("<!doctype html><html lang='en'><meta charset='utf-8'><title>ActiveVisionQA text dataset</title>"
                    "<style>body{font:16px system-ui;max-width:1000px;margin:2rem auto;padding:1rem;background:#eef2f6}"
                    "article{padding:1rem;background:white;margin:1rem 0}img{max-width:100%;display:block;margin:.5rem 0}</style>"
                    "<h1>Synthetic text camera dataset</h1><p>Actual Gazebo robot-camera captures. "
                    "The crop uses known fixture geometry. Ground-truth text does not establish readability. "
                    "No OCR predictions or learned-model results are shown.</p>"+"".join(cards)+"</html>", encoding="utf-8")
    return path
