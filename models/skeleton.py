from __future__ import annotations

from dataclasses import dataclass

from config.config import CANONICAL_SKELETON
from models.landmark import Landmark, LandmarkSet
from models.landmark_types import LandmarkStatus


@dataclass
class Skeleton:
    """Canonical skeleton for one horse in one frame."""

    landmarks: LandmarkSet
    backend: str = "canonical"

    def get(self, name: str) -> Landmark | None:
        return self.landmarks.get(name)

    def status_of(self, name: str) -> LandmarkStatus:
        landmark = self.get(name)
        if landmark is None:
            return LandmarkStatus.UNAVAILABLE
        return landmark.status

    def status_summary(self) -> dict[str, str]:
        return {landmark.name: landmark.status.value for landmark in self.landmarks.landmarks}

    def landmarks_with_status(self, status: LandmarkStatus) -> list[Landmark]:
        return [landmark for landmark in self.landmarks.landmarks if landmark.status is status]

    def detected_like(self) -> list[Landmark]:
        return [
            landmark
            for landmark in self.landmarks.landmarks
            if landmark.status in {LandmarkStatus.DETECTED, LandmarkStatus.DERIVED}
            and landmark.is_drawable
        ]

    @property
    def drawable_links(self) -> list[tuple[str, str]]:
        landmark_map = {landmark.name: landmark for landmark in self.detected_like()}

        links = []
        for start_name, end_name in CANONICAL_SKELETON:
            start = landmark_map.get(start_name)
            end = landmark_map.get(end_name)
            if start is None or end is None:
                continue
            if not start.is_reliable or not end.is_reliable:
                continue
            links.append((start_name, end_name))
        return links

    @classmethod
    def from_landmark_set(cls, landmarks: LandmarkSet, backend: str) -> Skeleton:
        return cls(landmarks=landmarks, backend=backend)
