from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from config.recording_protocol import RECORDING_QUALITY_WEIGHTS
from models.horse import HorseInstance


@dataclass(frozen=True)
class RecordingQualityFactors:
    """Per-factor measurement quality scores on a 0–100 scale."""

    motion_blur: float
    horse_visibility: float
    occlusion: float
    camera_motion: float
    resolution: float

    def overall(self) -> float:
        weighted = (
            self.motion_blur * RECORDING_QUALITY_WEIGHTS["motion_blur"]
            + self.horse_visibility * RECORDING_QUALITY_WEIGHTS["horse_visibility"]
            + self.occlusion * RECORDING_QUALITY_WEIGHTS["occlusion"]
            + self.camera_motion * RECORDING_QUALITY_WEIGHTS["camera_motion"]
            + self.resolution * RECORDING_QUALITY_WEIGHTS["resolution"]
        )
        return round(_clamp(weighted, 0.0, 100.0), 1)

    def as_dict(self) -> dict[str, float]:
        payload = asdict(self)
        payload["overall"] = self.overall()
        return payload


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _score_range(value: float, *, low: float, high: float, invert: bool = False) -> float:
    """Map a metric into 0–100 where low/high are poor/good (or reversed if invert)."""

    if high <= low:
        return 0.0
    normalized = _clamp((value - low) / (high - low), 0.0, 1.0)
    if invert:
        normalized = 1.0 - normalized
    return normalized * 100.0


class RecordingQualityAssessor:
    """Estimate how suitable a frame is for biomechanical measurement."""

    def __init__(self) -> None:
        self._prev_gray: np.ndarray | None = None

    def assess_frame(
        self,
        frame_bgr: np.ndarray,
        horses: list[HorseInstance],
        *,
        frame_width: int,
        frame_height: int,
    ) -> RecordingQualityFactors:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        frame_area = max(frame_width * frame_height, 1)

        motion_blur = self._motion_blur_score(gray)
        horse_visibility = self._horse_visibility_score(horses, frame_area, frame_width, frame_height)
        occlusion = self._occlusion_score(horses, frame_width, frame_height)
        camera_motion = self._camera_motion_score(gray)
        resolution = self._resolution_score(frame_width, frame_height)

        self._prev_gray = gray
        return RecordingQualityFactors(
            motion_blur=motion_blur,
            horse_visibility=horse_visibility,
            occlusion=occlusion,
            camera_motion=camera_motion,
            resolution=resolution,
        )

    def _motion_blur_score(self, gray: np.ndarray) -> float:
        # Laplacian variance: sharp frames typically > 120, blurry often < 40.
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return round(_score_range(variance, low=35.0, high=180.0), 1)

    def _horse_visibility_score(
        self,
        horses: list[HorseInstance],
        frame_area: int,
        frame_width: int,
        frame_height: int,
    ) -> float:
        if not horses:
            return 0.0

        primary = max(
            horses,
            key=lambda horse: (horse.bbox[2] - horse.bbox[0]) * (horse.bbox[3] - horse.bbox[1]),
        )
        x1, y1, x2, y2 = primary.bbox
        bbox_area = max(0.0, (x2 - x1) * (y2 - y1))
        coverage = bbox_area / frame_area

        # Ideal: horse fills a meaningful but not cropped portion of the frame.
        size_score = _score_range(coverage, low=0.04, high=0.35)
        if coverage > 0.75:
            size_score *= 0.7

        edge_penalty = 0.0
        margin = 6.0
        if x1 <= margin or y1 <= margin or x2 >= frame_width - margin or y2 >= frame_height - margin:
            edge_penalty = 15.0

        detection_bonus = _clamp(primary.detection_confidence * 12.0, 0.0, 12.0)
        return round(_clamp(size_score + detection_bonus - edge_penalty, 0.0, 100.0), 1)

    def _occlusion_score(
        self,
        horses: list[HorseInstance],
        frame_width: int,
        frame_height: int,
    ) -> float:
        if not horses:
            return 0.0

        primary = max(
            horses,
            key=lambda horse: len(horse.landmarks.visible_landmarks) if horse.landmarks else 0,
        )
        if primary.landmarks is None:
            return 40.0

        visible = primary.landmarks.visible_landmarks
        if not visible:
            return 20.0

        total = len(primary.landmarks.landmarks) or 1
        visibility_ratio = len(visible) / total
        confidence_values = [
            landmark.effective_confidence or landmark.confidence or 0.0
            for landmark in visible
            if (landmark.effective_confidence or landmark.confidence) is not None
        ]
        mean_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0

        x1, y1, x2, y2 = primary.bbox
        clipped = (
            x1 <= 2
            or y1 <= 2
            or x2 >= frame_width - 2
            or y2 >= frame_height - 2
        )
        clip_penalty = 12.0 if clipped else 0.0

        score = 55.0 * visibility_ratio + 45.0 * mean_confidence
        return round(_clamp(score - clip_penalty, 0.0, 100.0), 1)

    def _camera_motion_score(self, gray: np.ndarray) -> float:
        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            return 85.0

        diff = cv2.absdiff(self._prev_gray, gray)
        mean_motion = float(np.mean(diff))
        # Stationary cameras usually < 4; handheld panning 8–20; chaotic > 25.
        return round(_score_range(mean_motion, low=4.0, high=28.0, invert=True), 1)

    def _resolution_score(self, frame_width: int, frame_height: int) -> float:
        short_edge = min(frame_width, frame_height)
        if short_edge >= 1080:
            return 100.0
        if short_edge >= 720:
            return 85.0
        if short_edge >= 480:
            return 68.0
        if short_edge >= 360:
            return 50.0
        return 30.0
