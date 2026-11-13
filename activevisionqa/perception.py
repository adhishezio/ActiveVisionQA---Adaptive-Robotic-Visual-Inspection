"""Deterministic identifier/quality baseline; no neural confidence claims."""

import numpy as np

from .config import Config
from .models import Evidence, Outcome


class MarkerInspector:
    """Check one synthetic ArUco marking and conservative quality thresholds."""

    def __init__(self, config: Config):
        if config.marking_kind != "aruco":
            raise ValueError("MarkerInspector requires aruco; use dataset collection for text scenes until OCR is implemented")
        import cv2
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV with the ArUco module is required")
        self.cv2 = cv2
        self.config = config
        self.dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if hasattr(cv2.aruco, "ArucoDetector"):
            self.parameters = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        else:
            # OpenCV 4.6 exposes the class but needs its legacy factory here.
            self.parameters = cv2.aruco.DetectorParameters_create()
            self.detector = None
        # Keep a valid marker nested inside a larger white label candidate.
        self.parameters.minMarkerDistanceRate = 0.01
        if self.detector is not None:
            self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)

    def inspect(self, image: np.ndarray) -> Evidence:
        """Return a decision with measured pixel/contrast evidence, or abstain."""
        cv2 = self.cv2
        if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.size == 0:
            return Evidence(Outcome.REINSPECT, "invalid_image")
        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            return Evidence(Outcome.REINSPECT, "unsupported_image_shape")
        if min(gray.shape) < 32:
            return Evidence(Outcome.REINSPECT, "image_too_small")
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary, parameters=self.parameters)
        if ids is None:
            return Evidence(Outcome.REINSPECT, "marking_not_detected")
        identifiers = tuple(int(v) for v in ids.flatten())
        if len(identifiers) != 1:
            return Evidence(Outcome.REINSPECT, "multiple_markings", identifiers)
        polygon = corners[0].reshape(4, 2)
        edge = float(min(np.linalg.norm(polygon[(i+1) % 4]-polygon[i]) for i in range(4)))
        x0, y0 = np.maximum(np.floor(polygon.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(polygon.max(axis=0)).astype(int)+1, [gray.shape[1], gray.shape[0]])
        crop = gray[y0:y1, x0:x1]
        sharpness = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0
        contrast = float(np.percentile(crop, 95)-np.percentile(crop, 5)) if crop.size else 0.0
        metrics = {"min_marker_edge_px": edge, "laplacian_variance": sharpness, "contrast": contrast}
        if edge < self.config.min_marker_edge_px:
            return Evidence(Outcome.REINSPECT, "marking_too_small", identifiers, metrics)
        if sharpness < self.config.min_sharpness or contrast < 80:
            return Evidence(Outcome.REINSPECT, "insufficient_image_quality", identifiers, metrics)
        if identifiers[0] != self.config.expected_id:
            return Evidence(Outcome.REJECT, "unexpected_identifier", identifiers, metrics)
        return Evidence(Outcome.ACCEPT, "expected_identifier_and_quality", identifiers, metrics)
