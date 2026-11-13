"""CPU HOG/SVM fitting with synthetic bootstrap glyphs and train-only word data."""

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time

import cv2
import numpy as np

from .dataset import file_sha256, write_json
from .ocr import ALPHABET, FEATURE_VERSION, OcrSettings, glyph_feature, reference_features, reference_glyph, segment_characters
from .ocr_data import read_image, read_rows


def generated_features(seed: int, per_character: int) -> tuple[np.ndarray, np.ndarray]:
    """Create labeled character examples without reading any camera capture."""
    rng = np.random.default_rng(seed)
    features, labels = [], []
    for label, character in enumerate(ALPHABET):
        for index in range(per_character):
            scale = 1. if index == 0 else float(rng.uniform(.65, 2.2))
            thickness = 2 if index == 0 else int(rng.integers(1, 5))
            glyph = reference_glyph(character, scale, thickness)
            if index:
                matrix = cv2.getRotationMatrix2D((48, 48), float(rng.uniform(-7, 7)), 1.)
                matrix[0, 1] += float(rng.uniform(-.06, .06))
                glyph = cv2.warpAffine(glyph, matrix, (96, 96))
                if rng.random() < .5:
                    glyph = cv2.GaussianBlur(glyph, (3, 3), float(rng.uniform(.3, .9)))
                _, glyph = cv2.threshold(glyph, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            features.append(glyph_feature(glyph))
            labels.append(label)
    return np.stack(features), np.asarray(labels, np.int32)


def training_word_features(directory: Path, annotations: Path | None = None) -> tuple[list, list, dict]:
    """Use only training words; camera frames additionally need human review."""
    rows, source = read_rows(directory, annotations)
    selected = [row for row in rows if row["split"] == "train" and (
        (source == "offline_synthetic_text" and row["condition"] in ("clear", "wrong_id")) or
        (source == "gazebo_robot_camera" and row["readability"] == "readable"))]
    if not selected:
        raise ValueError("No eligible training samples; camera training requires reviewed readable training rows")
    features, labels, accepted, excluded = [], [], [], []
    for row in selected:
        glyphs, _, reason = segment_characters(read_image(directory, row))
        text = row["surface_text"]
        if reason or len(glyphs) != len(text):
            excluded.append({"sample_id": row["sample_id"], "reason": reason or "glyph_transcription_alignment_failed"})
            continue
        features.extend(glyph_feature(glyph) for glyph in glyphs)
        labels.extend(ALPHABET.index(character) for character in text)
        accepted.append(row["sample_id"])
    if not accepted:
        raise ValueError("No training images passed segmentation/alignment")
    return features, labels, {"source": source, "split": "train", "sample_ids": accepted,
                              "surface_strings": sorted({row["surface_text"] for row in selected if row["sample_id"] in accepted}),
                              "excluded": excluded, "annotations_sha256": file_sha256(annotations or directory / "readability_review.csv")
                              if (annotations or directory / "readability_review.csv").exists() else None,
                              "manifest_sha256": file_sha256(directory / ("manifest.jsonl" if source == "offline_synthetic_text" else "samples.jsonl"))}


def train_model(output: Path, dataset: Path | None = None, annotations: Path | None = None,
                seed: int = 991, per_character: int = 80) -> dict:
    """Fit a real classifier, save its artifacts and record exactly what was used."""
    if type(seed) is not int or not 0 <= seed < 2**31 or type(per_character) is not int or not 5 <= per_character <= 500:
        raise ValueError("Invalid SVM training seed/count")
    if output.exists():
        raise FileExistsError(output)
    if annotations is not None and dataset is None:
        raise ValueError("Annotations require a dataset")
    started = time.monotonic()
    cv2.setRNGSeed(seed)
    cv2.setNumThreads(1)
    features, labels = generated_features(seed, per_character)
    provenance = None
    if dataset is not None:
        word_features, word_labels, provenance = training_word_features(dataset, annotations)
        features = np.vstack((features, word_features)).astype(np.float32)
        labels = np.r_[labels, np.asarray(word_labels, np.int32)]
    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(10.)
    svm.setGamma(3.)
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER+cv2.TERM_CRITERIA_EPS, 3000, 1e-5))
    if not svm.train(features, cv2.ml.ROW_SAMPLE, labels):
        raise RuntimeError("SVM training failed")
    training_predictions = svm.predict(features)[1].reshape(-1).astype(int)
    output.mkdir(parents=True, exist_ok=False)
    svm.save(str(output / "svm.yml"))
    np.save(output / "references.npy", reference_features(), allow_pickle=False)
    metadata = {"schema_version": 1, "algorithm": "hog_svm", "trained": True,
                "feature_version": FEATURE_VERSION, "alphabet": ALPHABET, "opencv": cv2.__version__,
                "created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
                "settings": asdict(OcrSettings()), "kernel": "RBF", "C": 10., "gamma": 3.,
                "generated_glyphs": len(ALPHABET)*per_character, "training_features": len(labels),
                "class_counts": {ALPHABET[i]: n for i, n in Counter(labels.tolist()).items()},
                "word_training": provenance, "support_vectors": int(svm.getSupportVectors().shape[0]),
                "training_resubstitution_accuracy": float(np.mean(training_predictions == labels)),
                "training_seconds": time.monotonic()-started, "hardware": "CPU",
                "scope": "single synthetic font family, segmented characters; not industrial OCR validation",
                "artifact_sha256": {name: file_sha256(output / name) for name in ("svm.yml", "references.npy")},
                "source_sha256": {name: file_sha256(Path(__file__).parent / name)
                                  for name in ("ocr.py", "ocr_data.py", "ocr_training.py")}}
    write_json(output / "model.json", metadata)
    return metadata
