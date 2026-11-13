"""OCR recognition, training split isolation and evidence/metric regressions."""

import csv
from dataclasses import replace
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import pytest

from activevisionqa.config import Config
from activevisionqa.markings import text_bitmap
from activevisionqa.models import Observation, Outcome, Pose
from activevisionqa.ocr import ALPHABET, LineRecognizer, OcrPrediction, glyph_feature, reference_glyph
from activevisionqa.ocr_data import generate_words, read_annotations, read_rows
from activevisionqa.ocr_evaluation import edit_distance, evaluate, summarize_ocr
from activevisionqa.ocr_training import train_model, training_word_features
from activevisionqa.text_inspection import TextInspector, text_decision


@pytest.fixture(scope="module")
def model(tmp_path_factory):
    directory = tmp_path_factory.mktemp("ocr") / "model"
    metadata = train_model(directory, per_character=80)
    assert metadata["training_features"] == len(ALPHABET)*80
    assert metadata["trained"] and metadata["hardware"] == "CPU"
    return directory


@pytest.mark.parametrize("text", ["ABC-1234", "AVQ-0427", "XYZ-9876"])
def test_trained_model_reads_known_unoccluded_word_images(model, text):
    image = cv2.resize(text_bitmap(text), (480, 192))
    prediction = LineRecognizer.load(model).recognize(image)
    assert prediction.text == text


@pytest.mark.parametrize("image", [None, np.zeros((192, 480, 3), np.uint8),
    np.full((192, 480), 140, np.uint8), np.zeros((10, 10), np.uint8),
    np.zeros((192, 480, 4), np.uint8), np.zeros((192, 480), float)])
def test_invalid_or_blank_crops_abstain(model, image):
    assert LineRecognizer.load(model).recognize(image).text is None


def test_features_are_finite_and_versioned():
    features = glyph_feature(reference_glyph("A"))
    assert features.shape == (324,) and np.isfinite(features).all()
    assert np.linalg.norm(features) == pytest.approx(1.)
    with pytest.raises(ValueError, match="Empty"):
        glyph_feature(np.zeros((32, 32), np.uint8))


def test_expected_or_surface_metadata_cannot_fill_in_the_prediction(model):
    recognizer = LineRecognizer.load(model)
    pixels = cv2.resize(text_bitmap("ABC-1234"), (480, 192))
    a = TextInspector(replace(Config(), marking_kind="text", expected_text="ABC-1234", surface_text="AAA-0000"), recognizer, [500., 0., 320., 0., 500., 240., 0., 0., 1.])
    b = TextInspector(replace(Config(), marking_kind="text", expected_text="ABC-9999", surface_text="ZZZ-1111"), recognizer, [500., 0., 320., 0., 500., 240., 0., 0., 1.])
    accepted, rejected = a.inspect(pixels), b.inspect(pixels)
    assert accepted.text == rejected.text == "ABC-1234"
    assert accepted.outcome == Outcome.ACCEPT and rejected.outcome == Outcome.REJECT
    assert text_decision(OcrPrediction(None, "blocked"), "ABC-1234").outcome == Outcome.REINSPECT


def test_invalid_registered_roi_abstains(model):
    inspector = TextInspector(replace(Config(), marking_kind="text"), LineRecognizer.load(model), [500., 0., 320., 0., 500., 240., 0., 0., 1.])
    observation = Observation(np.zeros((480, 640, 3), np.uint8), 2., Config().optical_frame,
                              Pose((.5, 0, .1), (1., 0., 0., 0.)))
    assert inspector.prepare_observation(observation) is None
    assert inspector.inspect(None).outcome == Outcome.REINSPECT


def test_model_artifacts_are_hash_checked_and_never_overwritten(model, tmp_path):
    damaged = tmp_path / "damaged"
    shutil.copytree(model, damaged)
    with (damaged / "svm.yml").open("a") as stream:
        stream.write("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        LineRecognizer.load(damaged)
    with pytest.raises(FileExistsError):
        train_model(model)


def test_synthetic_word_groups_and_strings_do_not_cross_splits(tmp_path):
    directory = tmp_path / "data"
    generate_words(directory, counts=(2, 1, 1))
    rows, source = read_rows(directory)
    assert source == "offline_synthetic_text" and len(rows) == 16
    groups = {}
    words = {}
    for row in rows:
        assert groups.setdefault(row["part_id"], row["split"]) == row["split"]
        for text in (row["expected_text"], row["surface_text"]):
            assert words.setdefault(text, row["split"]) == row["split"]


def test_word_training_opens_only_eligible_training_images(tmp_path, monkeypatch):
    import activevisionqa.ocr_training as training
    directory = tmp_path / "data"
    generate_words(directory, counts=(1, 1, 1))
    original = training.read_image
    opened = []
    def record(root, row):
        assert row["split"] == "train"
        assert row["condition"] in ("clear", "wrong_id")
        opened.append(row["sample_id"])
        return original(root, row)
    monkeypatch.setattr(training, "read_image", record)
    _, _, provenance = training_word_features(directory)
    assert opened and all(name.startswith("train_") for name in opened)
    assert provenance["split"] == "train"


def test_unreviewed_camera_images_are_not_used_for_training(monkeypatch):
    import activevisionqa.ocr_training as training
    monkeypatch.setattr(training, "read_rows", lambda *args: ([{"split": "train", "readability": "unreviewed"}], "gazebo_robot_camera"))
    monkeypatch.setattr(training, "read_image", lambda *args: pytest.fail("Unreviewed image was opened for fitting"))
    with pytest.raises(ValueError, match="reviewed readable"):
        training_word_features(Path("unused"))


def test_readability_annotation_integrity(tmp_path):
    row = {"sample_id": "a", "split": "train", "surface_text": "ABC-1234"}
    path = tmp_path / "labels.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "split", "surface_text", "readability"])
        writer.writeheader()
        writer.writerow({**row, "readability": "readable"})
    assert read_annotations(path, {"a": row}) == {"a": "readable"}
    path.write_text(path.read_text().replace("readable", "invented"))
    with pytest.raises(ValueError, match="annotation value"):
        read_annotations(path, {"a": row})


def test_ocr_metrics_retain_abstentions_and_false_decision_denominators():
    records = [{"surface_text": "AB", "expected_text": "AB", "prediction": {"text": "AC"}, "decision": "REJECT", "recognition_ms": 1.},
               {"surface_text": "AC", "expected_text": "AB", "prediction": {"text": "AB"}, "decision": "ACCEPT", "recognition_ms": 1.},
               {"surface_text": "AB", "expected_text": "AB", "prediction": {"text": None}, "decision": "REINSPECT", "recognition_ms": 1.}]
    result = summarize_ocr(records)
    assert result["character_error_rate_all"] == pytest.approx(4/6)
    assert result["false_accept_rate"] == 1 and result["false_reject_rate"] == .5
    assert result["abstention_fraction"] == pytest.approx(1/3)
    assert edit_distance("kitten", "sitting") == 3


def test_test_evaluation_requires_matching_validation_artifacts(model, tmp_path):
    directory = tmp_path / "data"
    generate_words(directory, seed=900, counts=(1, 1, 1))
    with pytest.raises(ValueError, match="validation-report"):
        evaluate(directory, model, tmp_path / "premature", "test")
    evaluate(directory, model, tmp_path / "validation", "validation")
    report = evaluate(directory, model, tmp_path / "test", "test", validation_report=tmp_path / "validation/report.json")
    assert report["baselines"]["svm"]["overall"]["samples"] == 4
    prior = json.loads((tmp_path / "validation/report.json").read_text())
    prior["model_sha256"] = "changed"
    (tmp_path / "validation/report.json").write_text(json.dumps(prior))
    with pytest.raises(ValueError, match="frozen model"):
        evaluate(directory, model, tmp_path / "changed", "test", validation_report=tmp_path / "validation/report.json")


def test_annotation_changes_invalidate_test_freeze(model, tmp_path):
    directory = tmp_path / "data"
    generate_words(directory, seed=700, counts=(1, 1, 1))
    rows, _ = read_rows(directory)
    annotations = directory / "readability_review.csv"
    with annotations.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["sample_id", "split", "surface_text", "readability", "notes"])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in ("sample_id", "split", "surface_text")})
    evaluate(directory, model, tmp_path / "validation", "validation")
    annotations.write_text(annotations.read_text()+"\n")
    with pytest.raises(ValueError, match="annotations changed"):
        evaluate(directory, model, tmp_path / "test", "test", validation_report=tmp_path / "validation/report.json")
