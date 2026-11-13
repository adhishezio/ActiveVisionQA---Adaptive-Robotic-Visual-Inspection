"""Offline synthetic OCR examples and split-aware access to camera captures."""

import csv
import json
from pathlib import Path
import re
import string

import cv2
import numpy as np

from .dataset import SPLITS, audit_dataset, file_sha256, write_json
from .markings import text_bitmap

CONDITIONS = ("clear", "blurred", "blocked", "wrong_id")


def generate_words(directory: Path, seed: int = 217, counts: tuple[int, int, int] = (80, 20, 20)) -> dict:
    """Generate grouped whole-word diagnostics, separate from real camera data."""
    if type(seed) is not int or not 0 <= seed < 2**31 or len(counts) != 3 or any(type(n) is not int or not 1 <= n <= 2000 for n in counts):
        raise ValueError("Invalid OCR dataset seed/counts")
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "images").mkdir()
    rng = np.random.default_rng(seed)
    reserved = set()
    rows = []
    for split, count in zip(SPLITS, counts):
        for index in range(count):
            while True:
                text = "".join(rng.choice(list(string.ascii_uppercase), 3))+f"-{int(rng.integers(10000)):04d}"
                wrong = text[:-1]+str((int(text[-1])+1) % 10)
                if text not in reserved and wrong not in reserved:
                    reserved.update((text, wrong))
                    break
            part_id = f"word_{seed}_{split}_{index:04d}"
            for condition in CONDITIONS:
                surface = wrong if condition == "wrong_id" else text
                sample_seed = seed*1000000+len(rows)
                image = render_word(surface, condition, np.random.default_rng(sample_seed))
                sample_id = f"{split}_{index:04d}_{condition}"
                path = directory / "images" / (sample_id+".png")
                if not cv2.imwrite(str(path), image):
                    raise OSError("Failed to save OCR image")
                rows.append({"sample_id": sample_id, "part_id": part_id, "split": split,
                             "condition": condition, "seed": sample_seed, "expected_text": text,
                             "surface_text": surface, "image": path.relative_to(directory).as_posix(),
                             "image_sha256": file_sha256(path), "source": "offline_synthetic_text",
                             "readability": "unreviewed"})
    with (directory / "manifest.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
    metadata = {"schema_version": 1, "source": "offline_synthetic_text", "seed": seed,
                "word_identities_by_split": dict(zip(SPLITS, counts)), "conditions": list(CONDITIONS),
                "samples": len(rows), "font_family": "OpenCV FONT_HERSHEY_SIMPLEX",
                "scope": "same-font synthetic diagnostics; not industrial accuracy or robot motion"}
    write_json(directory / "ocr_dataset.json", metadata)
    return metadata


def render_word(text: str, condition: str, rng) -> np.ndarray:
    """Render a labeled crop with bounded optical/lighting variations."""
    bitmap = cv2.resize(text_bitmap(text), (480, 192), interpolation=cv2.INTER_NEAREST).astype(float)/255
    if condition not in CONDITIONS:
        raise ValueError("Unknown OCR condition")
    height, width = bitmap.shape
    if condition == "blocked":
        gray = rng.uniform(35, 65)+rng.normal(0, .8, (height, width))
    else:
        background = rng.uniform(185, 240)
        shade = np.linspace(float(rng.uniform(.7, .95)), 1., height)[:, None]
        gray = (15+(background-15)*bitmap)*shade
        gray += rng.normal(0, float(rng.uniform(.1, 2.)), gray.shape)
        sigma = float(rng.uniform(2.5, 3.8)) if condition == "blurred" else float(rng.uniform(.3, .8))
        gray = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), sigma)
    return cv2.cvtColor(np.clip(gray, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def read_rows(directory: Path, annotations: Path | None = None) -> tuple[list[dict], str]:
    """Read metadata and enforce grouping before any split is used for fitting."""
    if (directory / "ocr_dataset.json").is_file():
        metadata = json.loads((directory / "ocr_dataset.json").read_text())
        if metadata.get("schema_version") != 1 or metadata.get("source") != "offline_synthetic_text":
            raise ValueError("Unsupported synthetic OCR dataset")
        rows = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
        source = "offline_synthetic_text"
    else:
        audit = audit_dataset(directory)
        if not audit["passed"]:
            raise ValueError("Camera dataset audit failed: "+str(audit["errors"]))
        rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
        rows = [{**row, "image": row["rectified_image"], "image_sha256": row["rectified_sha256"],
                 "condition": row["scenario"], "readability": "unreviewed"} for row in rows]
        source = "gazebo_robot_camera"
    if not rows:
        raise ValueError("Empty OCR dataset")
    ids, groups, texts = set(), {}, {}
    for row in rows:
        if row["sample_id"] in ids or row["split"] not in SPLITS:
            raise ValueError("Invalid OCR sample ID/split")
        ids.add(row["sample_id"])
        if groups.setdefault(row["part_id"], row["split"]) != row["split"]:
            raise ValueError("OCR part identity leaks across splits")
        for text in (row["surface_text"], row["expected_text"]):
            if not re.fullmatch(r"[A-Z0-9-]{4,12}", text):
                raise ValueError("Invalid OCR transcription")
            if texts.setdefault(text, row["split"]) != row["split"]:
                raise ValueError("OCR word identity leaks across splits")
        if row["source"] != source:
            raise ValueError("Mixed OCR dataset sources")
    annotation_path = annotations or directory / "readability_review.csv"
    if annotation_path.exists():
        reviewed = read_annotations(annotation_path, {r["sample_id"]: r for r in rows})
        rows = [{**row, "readability": reviewed.get(row["sample_id"], "unreviewed")} for row in rows]
    elif annotations is not None:
        raise ValueError("Requested annotation file does not exist")
    return rows, source


def read_annotations(path: Path, rows: dict[str, dict]) -> dict[str, str]:
    """Validate human labels against immutable identities, splits and strings."""
    reviewed = {}
    with path.open(newline="") as stream:
        for label in csv.DictReader(stream):
            sample_id = label["sample_id"]
            if sample_id not in rows or sample_id in reviewed:
                raise ValueError("Unknown or duplicate readability annotation")
            row = rows[sample_id]
            if label["split"] != row["split"] or label["surface_text"] != row["surface_text"]:
                raise ValueError("Readability annotation disagrees with immutable metadata")
            value = label["readability"].strip().lower() or "unreviewed"
            if value not in ("readable", "unreadable", "uncertain", "unreviewed"):
                raise ValueError("Invalid readability annotation value")
            reviewed[sample_id] = value
    return reviewed


def read_image(directory: Path, row: dict) -> np.ndarray:
    """Verify and decode one selected image without consuming its label."""
    path = (directory / row["image"]).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError("Invalid OCR image path")
    if file_sha256(path) != row["image_sha256"]:
        raise ValueError("OCR image hash mismatch")
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError("Cannot decode OCR image")
    return image
