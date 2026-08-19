from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from config.config import KEYPOINT_SCORE_THRESHOLD


@dataclass
class Landmark:
    name: str
    index: int
    x: float | None = None
    y: float | None = None
    confidence: float | None = None
    visible: bool = False
    occluded: bool = False
    interpolated: bool = False
    predicted: bool = False
    outlier: bool = False
    identity_confidence: float = 1.0
    effective_confidence: float | None = None

    @property
    def is_drawable(self) -> bool:
        return self.visible and not self.outlier and self.x is not None and self.y is not None

    @property
    def is_reliable(self) -> bool:
        if not self.is_drawable:
            return False
        threshold = self.effective_confidence if self.effective_confidence is not None else self.confidence
        return threshold is not None and threshold >= KEYPOINT_SCORE_THRESHOLD


@dataclass
class LandmarkSet:
    landmarks: list[Landmark]
    schema: tuple[str, ...]

    @classmethod
    def template(cls, schema: list[str]) -> LandmarkSet:
        return cls(
            schema=tuple(schema),
            landmarks=[
                Landmark(name=name, index=index)
                for index, name in enumerate(schema)
            ],
        )

    @classmethod
    def from_arrays(
        cls,
        keypoints: np.ndarray,
        scores: np.ndarray,
        schema: list[str],
        score_threshold: float = KEYPOINT_SCORE_THRESHOLD,
    ) -> LandmarkSet:
        landmarks = []

        for index, name in enumerate(schema):
            x, y = keypoints[index]
            score = float(scores[index])

            if score >= score_threshold:
                landmarks.append(
                    Landmark(
                        name=name,
                        index=index,
                        x=float(x),
                        y=float(y),
                        confidence=score,
                        visible=True,
                        effective_confidence=score,
                    )
                )
            else:
                landmarks.append(
                    Landmark(
                        name=name,
                        index=index,
                        confidence=score,
                        visible=False,
                        effective_confidence=score,
                    )
                )

        return cls(landmarks=landmarks, schema=tuple(schema))

    def get(self, name: str) -> Landmark | None:
        index = self.schema.index(name)
        return self.landmarks[index]

    def with_point(
        self,
        name: str,
        x: float,
        y: float,
        confidence: float | None = None,
        *,
        visible: bool = True,
        identity_confidence: float = 1.0,
    ) -> LandmarkSet:
        index = self.schema.index(name)
        landmark = replace(
            self.landmarks[index],
            x=x,
            y=y,
            confidence=confidence,
            visible=visible,
            effective_confidence=confidence,
            identity_confidence=identity_confidence,
        )
        landmarks = self.landmarks.copy()
        landmarks[index] = landmark
        return LandmarkSet(landmarks=landmarks, schema=self.schema)

    def replace_landmark(self, name: str, landmark: Landmark) -> LandmarkSet:
        index = self.schema.index(name)
        landmarks = self.landmarks.copy()
        landmarks[index] = landmark
        return LandmarkSet(landmarks=landmarks, schema=self.schema)

    @property
    def visible_landmarks(self) -> list[Landmark]:
        return [landmark for landmark in self.landmarks if landmark.is_drawable]

    @property
    def reliable_landmarks(self) -> list[Landmark]:
        return [landmark for landmark in self.landmarks if landmark.is_reliable]

    def count_reliable(self) -> int:
        return len(self.reliable_landmarks)
