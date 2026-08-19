from __future__ import annotations

from dataclasses import dataclass, field

from models.detection import Detection


def bbox_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


@dataclass
class _TrackState:
    track_id: int
    bbox: list[float]
    last_frame: int
    misses: int = 0


@dataclass
class HorseTracker:
    """Assign stable track IDs to horse detections across frames."""

    iou_threshold: float = 0.3
    max_misses: int = 15
    _tracks: dict[int, _TrackState] = field(default_factory=dict)
    _next_track_id: int = 1

    def assign(self, detections: list[Detection], frame_index: int) -> list[tuple[Detection, int]]:
        if not detections:
            self._increment_misses(frame_index)
            return []

        unmatched_tracks = set(self._tracks.keys())
        assignments: list[tuple[Detection, int]] = []

        for detection in detections:
            best_track_id = None
            best_iou = 0.0

            for track_id in unmatched_tracks:
                iou = bbox_iou(detection.bbox, self._tracks[track_id].bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_track_id = track_id

            if best_track_id is not None and best_iou >= self.iou_threshold:
                track = self._tracks[best_track_id]
                track.bbox = detection.bbox
                track.last_frame = frame_index
                track.misses = 0
                unmatched_tracks.remove(best_track_id)
                assignments.append((detection, best_track_id))
            else:
                track_id = self._next_track_id
                self._next_track_id += 1
                self._tracks[track_id] = _TrackState(
                    track_id=track_id,
                    bbox=detection.bbox,
                    last_frame=frame_index,
                )
                assignments.append((detection, track_id))

        for track_id in unmatched_tracks:
            self._tracks[track_id].misses += 1

        self._prune_stale_tracks()
        return assignments

    def _increment_misses(self, frame_index: int) -> None:
        for track in self._tracks.values():
            track.misses += 1
            track.last_frame = frame_index
        self._prune_stale_tracks()

    def _prune_stale_tracks(self) -> None:
        stale_ids = [track_id for track_id, track in self._tracks.items() if track.misses > self.max_misses]
        for track_id in stale_ids:
            del self._tracks[track_id]
