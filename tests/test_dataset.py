"""Text generation, known-fixture geometry and dataset leakage regressions."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np
import pytest

from activevisionqa.assets import generate_workcell
from activevisionqa.config import Config
from activevisionqa.dataset import audit_dataset, file_sha256, make_plan, render_gallery, validate_plan, write_json, write_review_template
from activevisionqa.markings import ink_rectangles, project_text_roi, rectify_text_roi, surface_text, text_bitmap
from activevisionqa.models import Pose
from activevisionqa.perception import MarkerInspector


def test_plan_reproducibility_and_part_level_split_separation():
    plan = make_plan(123, 2, ("clear", "occluded", "wrong_id"))
    assert plan == make_plan(123, 2, ("clear", "occluded", "wrong_id"))
    assert plan != make_plan(124, 2, ("clear", "occluded", "wrong_id"))
    assert len(plan["cases"]) == 18 and plan["expected_images"] == 36
    for split in ("train", "validation", "test"):
        here = {c["part_id"] for c in plan["cases"] if c["split"] == split}
        other = {c["part_id"] for c in plan["cases"] if c["split"] != split}
        assert here.isdisjoint(other)


@pytest.mark.parametrize("fault", ["group", "text", "config", "case_path"])
def test_plan_rejects_leakage_and_false_ground_truth(fault):
    plan = make_plan(scenarios=("clear",))
    a, b = plan["cases"][:2]
    if fault == "group":
        b["part_id"] = a["part_id"]
    elif fault == "text":
        for key in ("expected_text", "surface_text"):
            b[key] = b["config"][key] = a[key]
    elif fault == "config":
        b["surface_text"] = "WRONG-42"
    else:
        a["case_id"] = "../outside"
    with pytest.raises(ValueError):
        validate_plan(plan)


@pytest.mark.parametrize("changes", [{"marking_kind": "invalid"}, {"expected_text": None},
    {"expected_text": "hello"}, {"surface_text": "A/B"}, {"light_scale": 0},
    {"marking_yaw_rad": float("nan")}])
def test_invalid_text_scene_configuration_is_rejected(changes):
    with pytest.raises(ValueError):
        replace(Config(), **changes)


def test_merged_native_ink_geometry_exactly_matches_bitmap():
    bitmap = text_bitmap("AVQ-0427")
    reconstructed = np.full_like(bitmap, 255)
    rectangles = ink_rectangles(bitmap)
    for x0, y0, x1, y1 in rectangles:
        reconstructed[y0:y1, x0:x1] = 0
    np.testing.assert_array_equal(bitmap, reconstructed)
    assert 20 < len(rectangles) < 1000
    assert np.count_nonzero(bitmap == 0) > 200


def test_text_world_has_native_glyphs_and_explicit_ground_truth(tmp_path):
    config = replace(Config(), marking_kind="text", marking_yaw_rad=.1)
    world = generate_workcell(tmp_path / "world", "wrong_id", config)
    metadata = json.loads((world.parent / "marking_metadata.json").read_text())
    assert metadata["surface_text"] != config.expected_text
    assert metadata["physical_engraving_simulated"] is False
    link = ET.parse(world).find("world/model[@name='inspection_part']/link")
    assert len([v for v in link.findall("visual") if v.get("name").startswith("text_ink_")]) > 20
    assert surface_text(config, "clear") == config.expected_text
    with pytest.raises(ValueError, match="requires aruco"):
        MarkerInspector(config)


def test_projection_uses_metric_geometry_and_measured_pose():
    config = replace(Config(), marking_kind="text")
    pose = Pose((.5, 0, .742), (1., 0., 0., 0.))
    corners = project_text_roi(config, pose, [500., 0., 320., 0., 500., 240., 0., 0., 1.])
    z = .742-(.242+.00016)
    expected = [[320-500*.075/z, 240-500*.03/z], [320+500*.075/z, 240-500*.03/z],
                [320+500*.075/z, 240+500*.03/z], [320-500*.075/z, 240+500*.03/z]]
    np.testing.assert_allclose(corners, expected)
    image = np.full((480, 640, 3), 190, np.uint8)
    crop = rectify_text_roi(image, corners)
    assert crop.shape == (192, 480, 3) and crop.mean() == 190
    with pytest.raises(ValueError, match="outside"):
        rectify_text_roi(image, corners+1000)
    with pytest.raises(ValueError, match="behind"):
        project_text_roi(config, Pose((.5, 0, .1), (1., 0., 0., 0.)), [500., 0., 320., 0., 500., 240., 0., 0., 1.])


def _schema_specimen(directory: Path):
    """Make audit schema specimens; these are not actual robot captures."""
    directory.mkdir()
    plan = make_plan(scenarios=("clear",))
    write_json(directory / "plan.json", plan)
    rows = []
    for index, case in enumerate(plan["cases"]):
        events = []
        config = Config.from_dict(case["config"])
        for view, viewpoint in enumerate(config.viewpoints(), 1):
            image = f"{index}_{view}.png"
            crop = f"{index}_{view}_crop.png"
            cv2.imwrite(str(directory / image), np.full((480, 640, 3), 50+index*10+view, np.uint8))
            cv2.imwrite(str(directory / crop), np.full((192, 480, 3), 50+index*10+view, np.uint8))
            checks = {"robot_motion_verified": True, "joint_limits_checked": True, "collision_checked": True,
                      "state_validity_samples": 2, "position_error_m": .001, "orientation_error_rad": .01}
            row = {k: case[k] for k in ("case_id", "part_id", "split", "scenario", "seed", "expected_text", "surface_text")}
            row.update(sample_id=f"{case['case_id']}_{view}", viewpoint=viewpoint.name, source="gazebo_robot_camera",
                simulation_only=True, readability_label=None, label_matches_expected=True,
                image_stamp=2., motion_arrived_at=1., optical_frame=config.optical_frame,
                camera_pose=viewpoint.camera_pose.to_dict(), camera_info={"width": 640, "height": 480,
                "frame_id": config.optical_frame, "k": [500., 0., 320., 0., 500., 240., 0., 0., 1.]},
                image=image, rectified_image=crop, image_sha256=file_sha256(directory / image),
                rectified_sha256=file_sha256(directory / crop), motion_checks=checks, events=f"{index}.jsonl")
            motion = {"ok": True, "arrived_at": 1., "checks": {**checks, "planned_joint_trajectory": {"points": [0]}}}
            events.append({"stage": "MOTION_RESULT", "sequence": len(events), "viewpoint": viewpoint.name, "motion": motion})
            events.append({"stage": "CAPTURE", "sequence": len(events), "sample": deepcopy(row)})
            rows.append(row)
        events.append({"stage": "FINISH", "sequence": len(events)})
        (directory / f"{index}.jsonl").write_text("".join(json.dumps(e)+"\n" for e in events))
    (directory / "samples.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    return rows


def test_complete_dataset_schema_audit_and_gallery(tmp_path):
    directory = tmp_path / "dataset"
    _schema_specimen(directory)
    audit = audit_dataset(directory)
    assert audit["passed"] and audit["all_planned_views_captured"]
    assert audit["samples_by_split"] == {"train": 2, "validation": 2, "test": 2}
    assert audit["readability_reviewed"] == 0
    gallery = render_gallery(directory).read_text()
    assert "unreviewed" in gallery and "0_1_crop.png" in gallery
    import csv
    labels = list(csv.DictReader(write_review_template(directory).open()))
    assert len(labels) == 6 and all(label["readability"] == "" for label in labels)


def test_partial_dataset_is_not_reported_as_complete(tmp_path):
    directory = tmp_path / "dataset"
    rows = _schema_specimen(directory)
    (directory / "samples.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows[:-1]))
    assert "incomplete_planned_views" in audit_dataset(directory)["errors"]
    preview = audit_dataset(directory, require_complete=False)
    assert preview["passed"] and not preview["all_planned_views_captured"]


@pytest.mark.parametrize("fault,reason", [("split", "sample_plan_disagreement:split"),
    ("stale", "image_not_after_motion"), ("hash", "image_hash_mismatch"),
    ("pose", "capture_pose_outside_tolerance"), ("motion", "missing_or_invalid_motion_verification"),
    ("readability", "unexpected_automatic_readability_label"), ("path", "missing_or_unsafe_image_path")])
def test_dataset_audit_rejects_corrupted_provenance(tmp_path, fault, reason):
    directory = tmp_path / "dataset"
    rows = _schema_specimen(directory)
    if fault == "split":
        rows[0]["split"] = "test"
    elif fault == "stale":
        rows[0]["image_stamp"] = 1.
    elif fault == "hash":
        rows[0]["image_sha256"] = "incorrect"
    elif fault == "pose":
        rows[0]["camera_pose"]["position"] = [.1, .1, .1]
    elif fault == "motion":
        rows[0]["motion_checks"]["collision_checked"] = False
    elif fault == "readability":
        rows[0]["readability_label"] = True
    elif fault == "path":
        rows[0]["image"] = "../outside.png"
    (directory / "samples.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
    result = audit_dataset(directory)
    assert not result["passed"] and reason in result["errors"]
