from __future__ import annotations

from statistics import mean

from models.session import FrameRecord


def compute_frame_pose_confidence(frame: FrameRecord) -> float:
    """How much we trust pose extraction on this frame (0–100)."""

    best_score = 0.0
    for horse in frame.horses:
        if horse.landmarks is None:
            continue

        scored_landmarks = [
            landmark.effective_confidence or landmark.confidence or 0.0
            for landmark in horse.landmarks.landmarks
            if landmark.visible and landmark.x is not None and landmark.y is not None
        ]
        if not scored_landmarks:
            continue

        reliable = [
            landmark
            for landmark in horse.landmarks.landmarks
            if landmark.is_reliable and landmark.x is not None and landmark.y is not None
        ]
        mean_effective = mean(scored_landmarks)
        reliability_ratio = len(reliable) / max(len(scored_landmarks), 1)
        frame_score = 100.0 * mean_effective * (0.65 + 0.35 * reliability_ratio)
        best_score = max(best_score, frame_score)

    return round(best_score, 1)


def summarize_pose_confidence(frames: list[FrameRecord]) -> dict[str, float]:
    horse_frames = [frame for frame in frames if frame.horses]
    if not horse_frames:
        return {"overall": 0.0, "mean": 0.0, "p10": 0.0}

    scores = [compute_frame_pose_confidence(frame) for frame in horse_frames]
    ordered = sorted(scores)
    p10_index = max(0, int(round(0.10 * (len(ordered) - 1))))
    return {
        "overall": round(mean(scores), 1),
        "mean": round(mean(scores), 1),
        "p10": round(ordered[p10_index], 1),
    }
