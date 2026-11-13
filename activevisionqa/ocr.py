"""Small single-line OCR baseline: segmentation, HOG and optional trained SVM.

This recognizer consumes only image pixels. Expected IDs are compared later
by the inspection layer and never used to fill in missing characters.
"""

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import string

import cv2
import numpy as np

ALPHABET = string.ascii_uppercase + string.digits + "-"
FEATURE_VERSION = "avqa_hog32_v2"
HOG = cv2.HOGDescriptor((32, 32), (16, 16), (8, 8), (8, 8), 9)


@dataclass(frozen=True)
class OcrSettings:
    """Engineering rejection gates, not calibrated confidence probabilities."""

    min_contrast: float = 25.0
    max_feature_distance: float = .65
    max_class_distance_gap: float = .15
    pattern: str = r"[A-Z]{3}-[0-9]{4}"

    def __post_init__(self):
        values = (self.min_contrast, self.max_feature_distance, self.max_class_distance_gap)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("OCR gates must be finite and positive")
        if len(self.pattern) > 128:
            raise ValueError("OCR format pattern is too long")
        re.compile(self.pattern)


@dataclass(frozen=True)
class OcrPrediction:
    text: str | None
    reason: str
    metrics: dict[str, float] = field(default_factory=dict)
    candidate_text: str | None = None


def normalize_glyph(mask: np.ndarray) -> np.ndarray:
    """Center white foreground in a 32x32 black image, retaining aspect ratio."""
    points = cv2.findNonZero(mask)
    if points is None:
        raise ValueError("Empty character")
    x, y, width, height = cv2.boundingRect(points)
    glyph = mask[y:y+height, x:x+width]
    scale = 24 / max(width, height)
    width, height = max(1, round(width*scale)), max(1, round(height*scale))
    resized = cv2.resize(glyph, (width, height), interpolation=cv2.INTER_AREA)
    result = np.zeros((32, 32), np.uint8)
    left, top = (32-width)//2, (32-height)//2
    result[top:top+height, left:left+width] = resized
    return result


def glyph_feature(mask: np.ndarray) -> np.ndarray:
    """Return a unit-normalized 324-dimensional HOG descriptor."""
    feature = HOG.compute(normalize_glyph(mask)).reshape(-1).astype(np.float32)
    return feature / max(float(np.linalg.norm(feature)), 1e-8)


def reference_glyph(character: str, scale: float = 1., thickness: int = 2) -> np.ndarray:
    """Generate a labeled reference with the same declared synthetic font family."""
    if character not in ALPHABET or len(character) != 1:
        raise ValueError("Unsupported character")
    image = np.zeros((96, 96), np.uint8)
    (width, height), baseline = cv2.getTextSize(character, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.putText(image, character, ((96-width)//2, (96+height-baseline)//2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, 255, thickness, cv2.LINE_8)
    return image


def reference_features() -> np.ndarray:
    """Fixed reference dictionary; character labels are independent of part IDs."""
    return np.stack([glyph_feature(reference_glyph(character)) for character in ALPHABET])


def _split_wide_component(mask: np.ndarray, median_height: float, depth: int = 0) -> list[np.ndarray]:
    """Separate touching glyphs only at a narrow vertical ink valley.

    This uses component geometry, never the expected transcription or its
    length. Ambiguous wide shapes remain unresolved for later rejection.
    """
    width = mask.shape[1]
    if depth >= 2 or width <= 1.25*median_height:
        return [mask]
    projection = (mask > 0).sum(axis=0)
    low, high = round(.3*width), round(.7*width)
    valley = low+int(np.argmin(projection[low:high]))
    if projection[valley] > .18*median_height:
        return [mask]
    pieces = []
    for piece in (mask[:, :valley], mask[:, valley:]):
        points = cv2.findNonZero(piece)
        if points is None:
            return [mask]
        x, y, w, h = cv2.boundingRect(points)
        if h < .55*median_height:
            return [mask]
        pieces.extend(_split_wide_component(piece[y:y+h, x:x+w], median_height, depth+1))
    return pieces


def segment_characters(image: np.ndarray, min_contrast: float = 25.) -> tuple[list[np.ndarray], dict[str, float], str | None]:
    """Segment a known-fixture single text line; reject weak/ambiguous images."""
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
        return [], {}, "invalid_ocr_image"
    if image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        return [], {}, "unsupported_ocr_image_shape"
    if min(gray.shape) < 16:
        return [], {}, "ocr_image_too_small"
    # Normalize resolution before component-size gates; no label is an input.
    gray = cv2.resize(gray, (480, 192), interpolation=cv2.INTER_AREA)
    contrast = float(np.percentile(gray, 95)-np.percentile(gray, 5))
    metrics = {"text_contrast": contrast}
    if contrast < min_contrast:
        return [], metrics, "insufficient_text_contrast"
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, np.ones((31, 31), np.uint8))
    normalized = cv2.divide(gray, np.maximum(background, 1), scale=255)
    _, mask = cv2.threshold(normalized, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    candidates = []
    for label in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[label])
        if area < 20:
            continue
        # Thin boundary artifacts are generated by fixture rectification.
        if (width > 430 and height < 10) or (height > 170 and width < 10):
            continue
        if x <= 1 or y <= 1 or x+width >= 479 or y+height >= 191:
            return [], metrics, "text_component_clipped_or_background"
        candidates.append((x, y, width, height, label))
    if not 4 <= len(candidates) <= 12:
        return [], metrics, "text_segmentation_count"
    candidates.sort()
    major = [box for box in candidates if box[3] >= 25]
    if len(major) < 3:
        return [], metrics, "no_consistent_text_line"
    heights = [box[3] for box in major]
    line_y = float(np.median([box[1]+box[3]/2 for box in major]))
    median_height = float(np.median(heights))
    if min(heights) < .55*median_height or max(heights) > 1.65*median_height:
        return [], metrics, "inconsistent_character_sizes"
    glyphs = []
    for x, y, width, height, label in candidates:
        if abs(y+height/2-line_y) > .45*median_height:
            return [], metrics, "inconsistent_text_baseline"
        if height < .4*median_height and not (width > .45*median_height and width > 2*height):
            return [], metrics, "unexplained_small_component"
        component = np.where(labels[y:y+height, x:x+width] == label, 255, 0).astype(np.uint8)
        glyphs.extend(_split_wide_component(component, median_height))
    if not 4 <= len(glyphs) <= 12:
        return [], metrics, "text_segmentation_count_after_split"
    metrics.update(character_count=float(len(glyphs)), median_character_height_px=median_height)
    return glyphs, metrics, None


class LineRecognizer:
    """Reference-template baseline or trained multi-class HOG/SVM recognizer."""

    def __init__(self, settings: OcrSettings | None = None, svm=None, references=None, metadata=None):
        self.settings = settings or OcrSettings()
        self.svm = svm
        self.references = reference_features() if references is None else np.asarray(references, np.float32)
        if self.references.shape != (len(ALPHABET), 324) or not np.isfinite(self.references).all():
            raise ValueError("Invalid OCR reference features")
        self.metadata = metadata or {"algorithm": "reference_templates", "feature_version": FEATURE_VERSION,
                                     "alphabet": ALPHABET, "settings": asdict(self.settings), "trained": False}

    @classmethod
    def load(cls, directory: str | Path) -> "LineRecognizer":
        """Load a versioned, hash-checked model bundle with its rejection gates."""
        directory = Path(directory)
        metadata = json.loads((directory / "model.json").read_text())
        if (metadata.get("schema_version") != 1 or metadata.get("trained") is not True
                or metadata.get("feature_version") != FEATURE_VERSION or metadata.get("alphabet") != ALPHABET or metadata.get("algorithm") != "hog_svm"):
            raise ValueError("Incompatible OCR model metadata")
        for filename, digest in metadata["artifact_sha256"].items():
            if filename not in ("svm.yml", "references.npy"):
                raise ValueError("Unexpected OCR artifact")
            if hashlib.sha256((directory / filename).read_bytes()).hexdigest() != digest:
                raise ValueError("OCR artifact hash mismatch")
        if set(metadata["artifact_sha256"]) != {"svm.yml", "references.npy"}:
            raise ValueError("Incomplete OCR model bundle")
        svm = cv2.ml.SVM_load(str(directory / "svm.yml"))
        if not svm.isTrained() or svm.getVarCount() != 324:
            raise ValueError("Invalid trained SVM")
        references = np.load(directory / "references.npy", allow_pickle=False)
        return cls(OcrSettings(**metadata["settings"]), svm, references, metadata)

    def recognize(self, image: np.ndarray) -> OcrPrediction:
        """Decode pixels without receiving an expected ID or scene ground truth."""
        glyphs, metrics, reason = segment_characters(image, self.settings.min_contrast)
        if reason:
            return OcrPrediction(None, reason, metrics)
        features = np.stack([glyph_feature(glyph) for glyph in glyphs])
        distances = 1 - features @ self.references.T
        labels = np.argmin(distances, axis=1) if self.svm is None else self.svm.predict(features)[1].reshape(-1).astype(int)
        if np.any(labels < 0) or np.any(labels >= len(ALPHABET)):
            return OcrPrediction(None, "invalid_character_class", metrics)
        chosen = distances[np.arange(len(labels)), labels]
        gaps = chosen-distances.min(axis=1)
        metrics.update(max_feature_distance=float(chosen.max()), max_class_distance_gap=float(gaps.max()))
        text = "".join(ALPHABET[index] for index in labels)
        if chosen.max() > self.settings.max_feature_distance or gaps.max() > self.settings.max_class_distance_gap:
            return OcrPrediction(None, "character_outside_reference_gate", metrics, text)
        if not re.fullmatch(self.settings.pattern, text):
            return OcrPrediction(None, "text_format_mismatch", metrics, text)
        return OcrPrediction(text, "decoded_text_and_quality", metrics, text)
