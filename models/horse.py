from __future__ import annotations

from dataclasses import dataclass

from models.landmark import LandmarkSet


@dataclass
class HorseInstance:
    track_id: int
    bbox: list[float]
    detection_confidence: float
    class_id: int
    landmarks: LandmarkSet | None = None
    identity_confidence: float = 1.0

    @property
    def bbox_center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @property
    def bbox_area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
