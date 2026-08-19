from __future__ import annotations

from detection.pose_backends import create_pose_backend
from detection.pose_backends.template import TemplatePoseBackend
from landmarks.mapping import map_to_canonical
from models.detection import Detection


class PoseEstimator:
    """Estimate horse pose and map backend landmarks to our canonical schema."""

    def __init__(self, backend: str | None = None):
        self.backend = create_pose_backend(backend)
        self._fallback = TemplatePoseBackend()
        self._warned = False

    @property
    def backend_name(self) -> str:
        return self.backend.schema_name

    def _warn_once(self, error: Exception) -> None:
        if self._warned:
            return

        print(f"Pose backend failed ({error}). Using template backend.")
        self._warned = True

    def estimate(self, frame, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return []

        backend = self.backend
        schema_name = backend.schema_name

        try:
            raw_detections = backend.estimate(frame, detections)
        except (ImportError, NotImplementedError, FileNotFoundError, ValueError, OSError, RuntimeError) as error:
            self._warn_once(error)
            backend = self._fallback
            schema_name = backend.schema_name
            raw_detections = backend.estimate(frame, detections)

        enriched = []

        for detection in raw_detections:
            raw_landmarks = detection.landmarks
            if raw_landmarks is None:
                canonical_landmarks = None
            else:
                canonical_landmarks = map_to_canonical(raw_landmarks, schema_name)

            enriched.append(
                Detection(
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    class_id=detection.class_id,
                    landmarks=canonical_landmarks,
                )
            )

        return enriched
