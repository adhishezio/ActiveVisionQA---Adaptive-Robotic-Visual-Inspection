"""Known-fixture text inspection, with expected-ID comparison after recognition."""

import re

from .config import Config
from .markings import project_text_roi, rectify_text_roi
from .models import Evidence, Observation, Outcome
from .ocr import LineRecognizer, OcrPrediction


def text_decision(prediction: OcrPrediction, expected_text: str) -> Evidence:
    """Keep OCR abstention distinct from a sufficiently evidenced wrong ID."""
    if prediction.text is None:
        return Evidence(Outcome.REINSPECT, prediction.reason, metrics=prediction.metrics)
    outcome = Outcome.ACCEPT if prediction.text == expected_text else Outcome.REJECT
    reason = "expected_text_and_quality" if outcome == Outcome.ACCEPT else "unexpected_text"
    return Evidence(outcome, reason, metrics=prediction.metrics, text=prediction.text)


class TextInspector:
    """Rectify a registered part's label, recognize pixels, then compare its ID."""

    def __init__(self, config: Config, recognizer: LineRecognizer, intrinsics: list[float]):
        if config.marking_kind != "text":
            raise ValueError("TextInspector requires a text workcell")
        if not re.fullmatch(recognizer.settings.pattern, config.expected_text):
            raise ValueError("Expected ID does not match this OCR model's supported format")
        self.config, self.recognizer, self.intrinsics = config, recognizer, intrinsics
        self.roi_error = None

    def prepare_observation(self, observation: Observation):
        """Use measured pose and known geometry; never use surface_text as input."""
        self.roi_error = None
        try:
            corners = project_text_roi(self.config, observation.camera_pose, self.intrinsics)
            return rectify_text_roi(observation.image, corners)
        except ValueError as exc:
            self.roi_error = str(exc)
            return None

    def inspect(self, image) -> Evidence:
        """Inspect a prepared ROI; no expected-string character substitution."""
        if image is None and self.roi_error:
            return Evidence(Outcome.REINSPECT, "text_roi_unavailable:"+self.roi_error)
        return text_decision(self.recognizer.recognize(image), self.config.expected_text)
