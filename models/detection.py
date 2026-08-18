from __future__ import annotations

from dataclasses import dataclass

from models.landmark import LandmarkSet


@dataclass
class Detection:
    bbox: list[float]
    confidence: float
    class_id: int
    landmarks: LandmarkSet | None = None
