from __future__ import annotations

from abc import ABC, abstractmethod

from models.detection import Detection


class PoseBackend(ABC):
    """Run pose inference on horse bounding boxes."""

    @property
    @abstractmethod
    def schema_name(self) -> str:
        """Schema identifier for downstream landmark mapping."""

    @abstractmethod
    def estimate(self, frame, detections: list[Detection]) -> list[Detection]:
        """Attach backend-specific raw landmarks to each detection."""
