from __future__ import annotations

from dataclasses import replace

from detection.detector import HorseDetector
from detection.pose_estimator import PoseEstimator
from landmarks.quality import LandmarkQualityController
from quality.pose_confidence import compute_frame_pose_confidence
from quality.recording import RecordingQualityAssessor
from landmarks.view import (
    ViewEstimate,
    ViewScores,
    ViewSmoother,
    classify_view_scores,
    compute_view_scores,
    estimate_identity_confidence,
    extract_view_features,
)
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
        frame_width: int = 0,
        frame_height: int = 0,
    ):
        self.detector = detector
        self.pose_estimator = pose_estimator
        self.tracker = HorseTracker()
        self.quality = LandmarkQualityController()
        self.recording_quality = RecordingQualityAssessor()
        self.session = SessionHistory(source_video=source_video, fps=fps)
        self._view_smoother = ViewSmoother(window=view_smoothing_window)
        self.frame_width = frame_width
        self.frame_height = frame_height

    def _estimate_frame_view(self, horses: list[HorseInstance]) -> ViewEstimate:
        if not horses:
            return classify_view_scores(ViewScores())

        ranked = sorted(
            horses,
            key=lambda horse: len(horse.landmarks.visible_landmarks) if horse.landmarks else 0,
            reverse=True,
        )
        for horse in ranked:
            raw_scores = compute_view_scores(extract_view_features(horse))
            if sum(raw_scores.normalized().values()) > 0:
                return self._view_smoother.update(raw_scores)
        return self._view_smoother.update(compute_view_scores(extract_view_features(ranked[0])))

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

        view_estimate = self._estimate_frame_view(horses)
        frame_record.view = view_estimate.view
        frame_record.view_confidence = view_estimate.confidence
        frame_record.view_distribution = view_estimate.distribution

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

        height, width = frame.shape[:2]
        frame_width = self.frame_width or width
        frame_height = self.frame_height or height
        recording = self.recording_quality.assess_frame(
            frame,
            frame_record.horses,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        frame_record.recording_quality = recording.overall()
        frame_record.recording_quality_factors = recording.as_dict()
        frame_record.pose_confidence = compute_frame_pose_confidence(frame_record)

        self.session.add(frame_record)
        return frame_record
