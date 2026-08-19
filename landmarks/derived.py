from __future__ import annotations

from config.config import KEYPOINT_SCORE_THRESHOLD
from models.landmark import LandmarkSet
from models.landmark_types import LandmarkSource, LandmarkStatus


def derive_spine_mid(landmarks: LandmarkSet) -> LandmarkSet:
    """Estimate spine_mid midway between withers and tail when both are detected."""

    withers = landmarks.get("withers")
    tail = landmarks.get("tail_head")

    if (
        withers is None
        or tail is None
        or withers.x is None
        or withers.y is None
        or tail.x is None
        or tail.y is None
        or withers.status not in {LandmarkStatus.DETECTED, LandmarkStatus.LOW_CONFIDENCE}
        or tail.status not in {LandmarkStatus.DETECTED, LandmarkStatus.LOW_CONFIDENCE}
    ):
        return landmarks

    confidence_values = [value for value in (withers.confidence, tail.confidence) if value is not None]
    derived_confidence = min(confidence_values) * 0.8 if confidence_values else None
    visible = derived_confidence is not None and derived_confidence >= KEYPOINT_SCORE_THRESHOLD

    return landmarks.with_point(
        "spine_mid",
        x=(withers.x + tail.x) / 2.0,
        y=(withers.y + tail.y) / 2.0,
        confidence=derived_confidence,
        visible=visible,
        status=LandmarkStatus.DERIVED,
        source=LandmarkSource.DERIVED,
    )


def apply_derived_landmarks(landmarks: LandmarkSet) -> LandmarkSet:
    return derive_spine_mid(landmarks)
