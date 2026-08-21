from __future__ import annotations

from models.session import SessionHistory


def summarize_tracking_confidence(session: SessionHistory) -> dict:
    """How consistently we tracked the primary horse through the session."""

    horse_frames = session.frames_with_horses()
    if not horse_frames:
        return {"overall": 0.0, "available": True, "track_count": 0, "primary_track_coverage": 0.0}

    track_stats = session.track_stats()
    if not track_stats:
        return {"overall": 0.0, "available": True, "track_count": 0, "primary_track_coverage": 0.0}

    primary_frames = max(value["frames"] for value in track_stats.values())
    coverage = primary_frames / len(horse_frames)
    track_count = len(track_stats)

    score = 100.0 * coverage
    if track_count > 2:
        score *= 0.75
    elif track_count > 1:
        score *= 0.88

    mean_detection = 0.0
    detections = [
        horse.detection_confidence
        for frame in horse_frames
        for horse in frame.horses
    ]
    if detections:
        mean_detection = sum(detections) / len(detections)
        score = score * 0.7 + mean_detection * 100.0 * 0.3

    return {
        "overall": round(score, 1),
        "available": True,
        "track_count": track_count,
        "primary_track_coverage": round(coverage, 3),
        "mean_detection_confidence": round(mean_detection, 3),
    }
