from __future__ import annotations

from enum import Enum


class LandmarkStatus(str, Enum):
    """Lifecycle state of a canonical landmark in a frame."""

    DETECTED = "detected"
    DERIVED = "derived"
    LOW_CONFIDENCE = "low_confidence"
    NOT_VISIBLE = "not_visible"
    UNMAPPED = "unmapped"
    UNAVAILABLE = "unavailable"


class LandmarkSource(str, Enum):
    """Where a landmark value came from."""

    NONE = "none"
    ANIMALPOSE = "animalpose"
    HORSE10 = "horse10"
    DERIVED = "derived"
    CANONICAL = "canonical"
