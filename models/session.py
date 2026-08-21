from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from config.landmark_spec import ViewName
from config.recording_protocol import RECOMMENDED_PROTOCOL
from landmarks.view import aggregate_view_distribution, dominant_view_from_distribution
from models.horse import HorseInstance


def _count_values(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_sec: float | None
    horses: list[HorseInstance]
    view: ViewName = "unknown"
    view_confidence: float = 0.0
    view_distribution: dict[str, float] = field(default_factory=dict)
    view_classification: str = "insufficient"
    view_label: str = "unknown"
    view_evidence: dict[str, dict] = field(default_factory=dict)
    observable_landmark_count: int = 0
    reliable_landmark_count: int = 0
    recording_quality: float = 0.0
    recording_quality_factors: dict[str, float] = field(default_factory=dict)
    pose_confidence: float = 0.0


@dataclass
class VideoMetadata:
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frame_count: int = 0
    duration_sec: float = 0.0

    def as_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "duration_sec": self.duration_sec,
        }


@dataclass
class SessionHistory:
    source_video: str
    fps: float
    video_metadata: VideoMetadata = field(default_factory=VideoMetadata)
    frames: list[FrameRecord] = field(default_factory=list)

    def add(self, frame: FrameRecord) -> None:
        self.frames.append(frame)

    def finalize(self, *, frame_count: int, width: int, height: int) -> None:
        self.video_metadata.width = width
        self.video_metadata.height = height
        self.video_metadata.fps = self.fps
        self.video_metadata.frame_count = frame_count
        self.video_metadata.duration_sec = frame_count / self.fps if self.fps > 0 else 0.0

    def _quality_summaries(
        self,
        *,
        dominant_view: str,
        view_confidence: float,
        view_distribution: dict[str, float],
    ) -> tuple[dict, dict, dict, dict, dict]:
        from quality.analysis_confidence import ConfidenceDimensions, build_analysis_confidence
        from quality.pose_confidence import summarize_pose_confidence
        from quality.tracking_confidence import summarize_tracking_confidence

        horse_frames = self.frames_with_horses()
        if not horse_frames:
            empty = {"overall": 0.0}
            gait_stub = {"overall": None, "available": False, "reason": "gait classification not yet implemented"}
            confidence = build_analysis_confidence(
                ConfidenceDimensions(0.0, 0.0, 0.0, 0.0),
                dominant_view=dominant_view,
            ).as_dict()
            return empty, empty, gait_stub, confidence, {"overall": 0.0, "available": True}

        recording_scores = [frame.recording_quality for frame in horse_frames]
        factor_keys = ("motion_blur", "horse_visibility", "occlusion", "camera_motion", "resolution")
        factor_means = {
            key: sum(frame.recording_quality_factors.get(key, 0.0) for frame in horse_frames) / len(horse_frames)
            for key in factor_keys
        }
        recording_summary = {
            "overall": round(sum(recording_scores) / len(recording_scores), 1),
            "factors": {key: round(value, 1) for key, value in factor_means.items()},
        }

        pose_summary = summarize_pose_confidence(self.frames)
        tracking_summary = summarize_tracking_confidence(self)
        gait_summary = {
            "overall": None,
            "available": False,
            "reason": "gait classification not yet implemented",
        }

        confidence = build_analysis_confidence(
            ConfidenceDimensions(
                recording_quality=recording_summary["overall"],
                pose_confidence=pose_summary["overall"],
                view_confidence=round(view_confidence * 100.0, 1),
                tracking_confidence=tracking_summary["overall"],
                gait_confidence=None,
            ),
            recording_factors=factor_means,
            dominant_view=dominant_view,
            view_distribution=view_distribution,
            track_count=tracking_summary.get("track_count", 1),
        ).as_dict()

        return recording_summary, pose_summary, gait_summary, confidence, tracking_summary

    def get_track_ids(self) -> list[int]:
        return sorted({horse.track_id for frame in self.frames for horse in frame.horses})

    def get_trajectory(
        self,
        track_id: int,
        landmark_name: str,
        *,
        include_outliers: bool = False,
        include_unreliable: bool = True,
    ) -> list[dict]:
        points = []

        for frame in self.frames:
            horse = next((item for item in frame.horses if item.track_id == track_id), None)
            if horse is None or horse.landmarks is None:
                continue

            landmark = horse.landmarks.get(landmark_name)
            if landmark is None:
                continue

            has_validated = landmark.x is not None and landmark.y is not None
            has_raw = landmark.raw_x is not None and landmark.raw_y is not None
            if not has_validated and not has_raw:
                continue
            if not has_validated:
                if not include_outliers:
                    continue
            elif not landmark.visible and not include_unreliable:
                continue
            elif landmark.outlier and not include_outliers:
                continue
            elif not include_unreliable and not landmark.is_reliable:
                continue

            points.append(
                {
                    "frame": frame.frame_index,
                    "timestamp_sec": frame.timestamp_sec,
                    "x": landmark.x,
                    "y": landmark.y,
                    "confidence": landmark.confidence,
                    "raw_x": landmark.raw_x,
                    "raw_y": landmark.raw_y,
                    "raw_confidence": landmark.raw_confidence,
                    "effective_confidence": landmark.effective_confidence,
                    "status": landmark.status.value,
                    "source": landmark.source.value,
                    "visible": landmark.visible,
                    "reliable": landmark.is_reliable,
                    "outlier": landmark.outlier,
                    "view": frame.view,
                }
            )

        return points

    def landmark_coverage(self, track_id: int) -> dict[str, int]:
        counts: dict[str, int] = {}

        for frame in self.frames:
            horse = next((item for item in frame.horses if item.track_id == track_id), None)
            if horse is None or horse.landmarks is None:
                continue

            for landmark in horse.landmarks.landmarks:
                if landmark.x is None or landmark.y is None:
                    continue
                if not landmark.visible:
                    continue
                counts[landmark.name] = counts.get(landmark.name, 0) + 1

        return counts

    def landmark_status_breakdown(self, track_id: int) -> dict[str, dict[str, int]]:
        """Count frames per landmark status for one track."""

        breakdown: dict[str, dict[str, int]] = {}

        for frame in self.frames:
            horse = next((item for item in frame.horses if item.track_id == track_id), None)
            if horse is None or horse.landmarks is None:
                continue

            for landmark in horse.landmarks.landmarks:
                status_key = landmark.status.value
                landmark_counts = breakdown.setdefault(landmark.name, {})
                landmark_counts[status_key] = landmark_counts.get(status_key, 0) + 1

        return breakdown

    def build_trajectory_index(self) -> dict[str, dict[str, list[dict]]]:
        index: dict[str, dict[str, list[dict]]] = {}

        for track_id in self.get_track_ids():
            track_key = f"track_{track_id}"
            index[track_key] = {}

            landmark_names = set()
            for frame in self.frames:
                horse = next((item for item in frame.horses if item.track_id == track_id), None)
                if horse and horse.landmarks:
                    landmark_names.update(horse.landmarks.schema)

            for landmark_name in sorted(landmark_names):
                index[track_key][landmark_name] = self.get_trajectory(track_id, landmark_name)

        return index

    def track_frame_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for frame in self.frames:
            for horse in frame.horses:
                counts[horse.track_id] = counts.get(horse.track_id, 0) + 1
        return counts

    def track_landmark_observations(self, track_id: int) -> int:
        total = 0
        for frame in self.frames:
            horse = next((item for item in frame.horses if item.track_id == track_id), None)
            if horse is None or horse.landmarks is None:
                continue
            total += sum(
                1
                for landmark in horse.landmarks.landmarks
                if landmark.visible and landmark.x is not None and landmark.y is not None
            )
        return total

    def track_stats(self) -> dict[int, dict[str, int]]:
        stats: dict[int, dict[str, int]] = {}
        for track_id in self.get_track_ids():
            stats[track_id] = {
                "frames": self.track_frame_counts().get(track_id, 0),
                "landmark_observations": self.track_landmark_observations(track_id),
            }
        return stats

    def frames_with_horses(self) -> list[FrameRecord]:
        return [frame for frame in self.frames if frame.horses]

    def dominant_view_for_track(self, track_id: int) -> ViewName:
        views = [
            frame.view
            for frame in self.frames
            if any(horse.track_id == track_id for horse in frame.horses)
            and frame.view != "unknown"
        ]
        if not views:
            return "unknown"
        return max(set(views), key=views.count)

    def summary(self) -> dict:
        if not self.frames:
            return {"frames": 0}

        horse_frames = self.frames_with_horses()
        reliable_counts = [frame.reliable_landmark_count for frame in horse_frames] or [0]

        views = [
            frame.view
            for frame in horse_frames
            if frame.view_classification == "confident" and frame.view != "unknown"
        ]
        dominant_view = max(set(views), key=views.count) if views else "unknown"

        from landmarks.view import ViewEstimate, ViewScores

        frame_estimates = [
            ViewEstimate(
                view=frame.view,
                confidence=frame.view_confidence,
                scores=ViewScores(
                    side=frame.view_distribution.get("side", 0.0),
                    front=frame.view_distribution.get("front", 0.0),
                    rear=frame.view_distribution.get("rear", 0.0),
                    oblique=frame.view_distribution.get("oblique", 0.0),
                ),
                distribution=frame.view_distribution,
            )
            for frame in horse_frames
            if frame.view_distribution and frame.view_classification == "confident"
        ]
        view_distribution = aggregate_view_distribution(frame_estimates)
        dominant_view, view_confidence = dominant_view_from_distribution(view_distribution)
        if dominant_view == "unknown":
            dominant_view = max(set(views), key=views.count) if views else "unknown"

        recording_summary, pose_summary, gait_summary, confidence, tracking_summary = self._quality_summaries(
            dominant_view=dominant_view,
            view_confidence=view_confidence,
            view_distribution=view_distribution,
        )

        view_summary = {
            "dominant": dominant_view,
            "confidence": round(view_confidence, 3),
            "distribution": view_distribution,
            "classification_counts": _count_values(frame.view_classification for frame in horse_frames),
            "label_counts": _count_values(frame.view_label for frame in horse_frames if frame.view_label),
        }

        return {
            "frames": len(self.frames),
            "frames_with_horses": len(horse_frames),
            "tracks": self.get_track_ids(),
            "track_stats": self.track_stats(),
            "view": view_summary,
            # Backward-compatible aliases for existing scripts.
            "dominant_view": dominant_view,
            "view_confidence": view_confidence,
            "view_distribution": view_distribution,
            "avg_reliable_landmarks": sum(reliable_counts) / len(reliable_counts),
            "max_reliable_landmarks": max(reliable_counts),
            "recording_quality": recording_summary,
            "pose_confidence": pose_summary,
            "gait_confidence": gait_summary,
            "tracking_confidence": tracking_summary,
            "confidence": confidence,
        }

    def export_json(self, path: str | Path) -> None:
        payload = {
            "pipeline_version": "validation-v6",
            "source_video": self.source_video,
            "fps": self.fps,
            "video_metadata": self.video_metadata.as_dict(),
            "recording_protocol": {
                "trainer_message": RECOMMENDED_PROTOCOL["trainer_message"],
            },
            "summary": self.summary(),
            "frames": [
                {
                    "frame_index": frame.frame_index,
                    "timestamp_sec": frame.timestamp_sec,
                    "view": frame.view,
                    "view_confidence": frame.view_confidence,
                    "view_distribution": frame.view_distribution,
                    "view_classification": frame.view_classification,
                    "view_label": frame.view_label,
                    "view_evidence": frame.view_evidence,
                    "observable_landmark_count": frame.observable_landmark_count,
                    "reliable_landmark_count": frame.reliable_landmark_count,
                    "recording_quality": frame.recording_quality,
                    "recording_quality_factors": frame.recording_quality_factors,
                    "pose_confidence": frame.pose_confidence,
                    "horses": [
                        {
                            "track_id": horse.track_id,
                            "bbox": horse.bbox,
                            "detection_confidence": horse.detection_confidence,
                            "identity_confidence": horse.identity_confidence,
                            "landmarks": [
                                {
                                    "name": landmark.name,
                                    "x": landmark.x,
                                    "y": landmark.y,
                                    "confidence": landmark.confidence,
                                    "raw_x": landmark.raw_x,
                                    "raw_y": landmark.raw_y,
                                    "raw_confidence": landmark.raw_confidence,
                                    "effective_confidence": landmark.effective_confidence,
                                    "status": landmark.status.value,
                                    "source": landmark.source.value,
                                    "visible": landmark.visible,
                                    "occluded": landmark.occluded,
                                    "outlier": landmark.outlier,
                                    "identity_confidence": landmark.identity_confidence,
                                }
                                for landmark in (horse.landmarks.landmarks if horse.landmarks else [])
                            ],
                        }
                        for horse in frame.horses
                    ],
                }
                for frame in self.frames
            ],
            "trajectories": self.build_trajectory_index(),
        }

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
