from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from config.landmark_spec import ViewName
from models.horse import HorseInstance


@dataclass
class FrameRecord:
    frame_index: int
    timestamp_sec: float | None
    horses: list[HorseInstance]
    view: ViewName = "unknown"
    observable_landmark_count: int = 0
    reliable_landmark_count: int = 0


@dataclass
class SessionHistory:
    source_video: str
    fps: float
    frames: list[FrameRecord] = field(default_factory=list)

    def add(self, frame: FrameRecord) -> None:
        self.frames.append(frame)

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

        views = [frame.view for frame in horse_frames if frame.view != "unknown"]
        dominant_view = max(set(views), key=views.count) if views else "unknown"

        return {
            "frames": len(self.frames),
            "frames_with_horses": len(horse_frames),
            "tracks": self.get_track_ids(),
            "track_stats": self.track_stats(),
            "dominant_view": dominant_view,
            "avg_reliable_landmarks": sum(reliable_counts) / len(reliable_counts),
            "max_reliable_landmarks": max(reliable_counts),
        }

    def export_json(self, path: str | Path) -> None:
        payload = {
            "pipeline_version": "validation-v2",
            "source_video": self.source_video,
            "fps": self.fps,
            "summary": self.summary(),
            "frames": [
                {
                    "frame_index": frame.frame_index,
                    "timestamp_sec": frame.timestamp_sec,
                    "view": frame.view,
                    "observable_landmark_count": frame.observable_landmark_count,
                    "reliable_landmark_count": frame.reliable_landmark_count,
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
