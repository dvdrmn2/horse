from __future__ import annotations

from collections import deque
from dataclasses import replace

from config.landmark_spec import ViewName
from detection.detector import HorseDetector
from detection.pose_estimator import PoseEstimator
from landmarks.quality import LandmarkQualityController
from landmarks.view import estimate_camera_view, estimate_identity_confidence
from models.horse import HorseInstance
from models.session import FrameRecord, SessionHistory
from tracking.tracker import HorseTracker


class VideoSessionProcessor:
    """Frame → detect → track → pose → quality → session history."""

    def __init__(
        self,
        detector: HorseDetector,
        pose_estimator: PoseEstimator,
        *,
        source_video: str,
        fps: float,
        view_smoothing_window: int = 15,
    ):
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.tracker = HorseTracker()
        self.quality = LandmarkQualityController()
        self.session = SessionHistory(source_video=source_video, fps=fps)
        self._view_history: deque[ViewName] = deque(maxlen=view_smoothing_window)

    def _smooth_view(self, raw_view: ViewName) -> ViewName:
        if raw_view != "unknown":
            self._view_history.append(raw_view)
        if not self._view_history:
            return raw_view
        return max(set(self._view_history), key=list(self._view_history).count)

    def process_frame(self, frame, frame_index: int) -> FrameRecord:
        timestamp_sec = frame_index / self.session.fps if self.session.fps > 0 else None

        detections = self.detector.detect(frame)
        detections = self.pose_estimator.estimate(frame, detections)
        assignments = self.tracker.assign(detections, frame_index)

        horses = [
            HorseInstance(
                track_id=track_id,
                bbox=detection.bbox,
                detection_confidence=detection.confidence,
                class_id=detection.class_id,
                landmarks=detection.landmarks,
            )
            for detection, track_id in assignments
        ]

        frame_record = FrameRecord(
            frame_index=frame_index,
            timestamp_sec=timestamp_sec,
            horses=horses,
        )

        raw_view = estimate_camera_view(horses)
        frame_record.view = self._smooth_view(raw_view)

        for horse in frame_record.horses:
            horse.identity_confidence = estimate_identity_confidence(horse, frame_record.view)
            if horse.landmarks is None:
                continue

            updated_landmarks = []
            for landmark in horse.landmarks.landmarks:
                if landmark.visible:
                    updated_landmarks.append(
                        replace(
                            landmark,
                            identity_confidence=horse.identity_confidence,
                        )
                    )
                else:
                    updated_landmarks.append(landmark)

            horse.landmarks = horse.landmarks.__class__(
                landmarks=updated_landmarks,
                schema=horse.landmarks.schema,
            )

        frame_record = self.quality.refine(frame_record)
        self.session.add(frame_record)
        return frame_record
