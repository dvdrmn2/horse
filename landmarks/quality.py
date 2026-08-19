from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field, replace

from config.config import KEYPOINT_SCORE_THRESHOLD
from config.landmark_spec import LANDMARK_DEFINITIONS, ViewName, observable_landmarks_for_view
from models.landmark import Landmark
from models.session import FrameRecord


def bbox_diagonal(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    return math.hypot(width, height)


def view_confidence_multiplier(view: ViewName, landmark_name: str) -> float:
    definition = LANDMARK_DEFINITIONS.get(landmark_name)
    if definition is None:
        return 0.85 if view == "unknown" else 0.5

    if view == "unknown":
        # Do not over-penalize before view is confidently classified.
        return 0.9

    return definition.view_suitability.score(view)


@dataclass
class LandmarkQualityController:
    """Temporal outlier rejection and view-adjusted confidence scoring."""

    jump_threshold_ratio: float = 0.55
    history_length: int = 5
    _history: dict[tuple[int, str], deque[tuple[float, float, float]]] = field(default_factory=dict)

    def refine(self, frame: FrameRecord) -> FrameRecord:
        observable_names = set(observable_landmarks_for_view(frame.view, min_score=0.45))
        reliable_count = 0

        for horse in frame.horses:
            if horse.landmarks is None:
                continue

            max_jump = self.jump_threshold_ratio * bbox_diagonal(horse.bbox)
            updated_landmarks = []

            for landmark in horse.landmarks.landmarks:
                updated = self._score_landmark(
                    horse.track_id,
                    landmark,
                    frame.view,
                    max_jump,
                    observable_names,
                    horse.identity_confidence,
                )
                updated_landmarks.append(updated)
                if updated.is_reliable:
                    reliable_count += 1

            horse.landmarks = horse.landmarks.__class__(
                landmarks=updated_landmarks,
                schema=horse.landmarks.schema,
            )

        frame.reliable_landmark_count = reliable_count
        frame.observable_landmark_count = len(observable_names)
        return frame

    def _score_landmark(
        self,
        track_id: int,
        landmark: Landmark,
        view: ViewName,
        max_jump: float,
        observable_names: set[str],
        identity_confidence: float,
    ) -> Landmark:
        base_confidence = landmark.confidence or 0.0
        view_score = view_confidence_multiplier(view, landmark.name)
        effective = base_confidence * view_score * identity_confidence

        if landmark.name not in observable_names and view != "unknown":
            effective *= 0.75

        updated = replace(landmark, effective_confidence=effective)

        if not updated.visible or updated.x is None or updated.y is None:
            return updated

        history_key = (track_id, landmark.name)
        history = self._history.setdefault(history_key, deque(maxlen=self.history_length))

        if history:
            last_x, last_y, _ = history[-1]
            jump = math.hypot(updated.x - last_x, updated.y - last_y)
            if jump > max_jump:
                updated = replace(
                    updated,
                    outlier=True,
                    visible=False,
                    effective_confidence=min(effective, KEYPOINT_SCORE_THRESHOLD * 0.5),
                )
                return updated

        history.append((updated.x, updated.y, base_confidence))
        return updated
