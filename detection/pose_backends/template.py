from __future__ import annotations

from config.config import CANONICAL_LANDMARKS
from detection.pose_backends.base import PoseBackend
from models.detection import Detection
from models.landmark import LandmarkSet


class TemplatePoseBackend(PoseBackend):
    """Fallback backend that returns empty canonical landmarks."""

    @property
    def schema_name(self) -> str:
        return "canonical"

    def estimate(self, frame, detections: list[Detection]) -> list[Detection]:
        return [
            Detection(
                bbox=detection.bbox,
                confidence=detection.confidence,
                class_id=detection.class_id,
                landmarks=LandmarkSet.template(CANONICAL_LANDMARKS),
            )
            for detection in detections
        ]
