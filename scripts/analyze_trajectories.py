#!/usr/bin/env python3
"""Analyze temporal stability of detected landmarks in exported trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import CANONICAL_LANDMARKS, KEYPOINT_SCORE_THRESHOLD
from landmarks.mapping import ANIMALPOSE_CANONICAL_COVERAGE
from landmarks.temporal import format_metrics, summarize_trajectory

DEFAULT_LANDMARKS = (
    "withers",
    "left_stifle",
    "right_stifle",
    "left_carpus",
    "right_carpus",
    "left_front_hoof",
    "right_front_hoof",
    "left_hind_hoof",
    "right_hind_hoof",
    "nose",
    "neck",
    "tail_head",
)


def _points_from_frames(frames: list[dict], track_id: int, landmark_name: str) -> list[dict]:
    points = []

    for frame in frames:
        horse = next(
            (item for item in frame.get("horses", []) if item.get("track_id") == track_id),
            None,
        )
        if horse is None:
            continue

        landmark = next(
            (item for item in horse.get("landmarks", []) if item.get("name") == landmark_name),
            None,
        )
        if landmark is None or landmark.get("x") is None or landmark.get("y") is None:
            continue

        status = landmark.get("status")
        if status in {"unmapped", "unavailable", "not_visible"}:
            continue

        confidence = landmark.get("confidence")
        effective = landmark.get("effective_confidence", confidence)
        reliable = (
            status in {"detected", "derived"}
            and effective is not None
            and effective >= KEYPOINT_SCORE_THRESHOLD
            and not landmark.get("outlier", False)
        )

        points.append(
            {
                "frame": frame.get("frame_index"),
                "timestamp_sec": frame.get("timestamp_sec"),
                "x": landmark["x"],
                "y": landmark["y"],
                "confidence": confidence,
                "effective_confidence": effective,
                "status": status,
                "source": landmark.get("source"),
                "visible": landmark.get("visible", False),
                "reliable": reliable,
                "outlier": landmark.get("outlier", False),
                "view": frame.get("view"),
                "bbox": horse.get("bbox"),
            }
        )

    return points


def _status_breakdown_from_frames(frames: list[dict], track_id: int, landmark_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}

    for frame in frames:
        horse = next(
            (item for item in frame.get("horses", []) if item.get("track_id") == track_id),
            None,
        )
        if horse is None:
            continue

        landmark = next(
            (item for item in horse.get("landmarks", []) if item.get("name") == landmark_name),
            None,
        )
        if landmark is None:
            continue

        status = landmark.get("status", "unknown")
        counts[status] = counts.get(status, 0) + 1

    return counts


def print_landmark_report(
    landmark_name: str,
    metrics,
    status_counts: dict[str, int],
) -> None:
    source = "AnimalPose" if landmark_name in ANIMALPOSE_CANONICAL_COVERAGE else "canonical-only"
    print(f"{landmark_name:22s}  ({source})")
    print(f"  trajectory frames : {metrics.frames}")
    print(f"  status breakdown  : {status_counts or {'(no frame data)': 0}}")

    if metrics.frames == 0:
        print("  (no trajectory points)")
        print()
        return

    for line in format_metrics(metrics):
        print(line)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze landmark trajectory stability.")
    parser.add_argument("path", type=Path, help="Path to *.trajectories.json")
    parser.add_argument("--track", type=int, required=True, help="Track ID to analyze")
    parser.add_argument(
        "--landmarks",
        nargs="*",
        default=list(DEFAULT_LANDMARKS),
        help="Landmarks to analyze (defaults to AnimalPose-friendly set)",
    )
    args = parser.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    frames = payload.get("frames", [])
    summary = payload.get("summary", {})
    track_stats = summary.get("track_stats", {}).get(str(args.track)) or summary.get("track_stats", {}).get(args.track, {})

    print(f"Analyzing {args.path.name} — track_{args.track}")
    print(f"Track frames: {track_stats.get('frames', '?')}")
    print(f"Dominant view: {summary.get('dominant_view', 'unknown')}")
    print("Note: large raw velocity is expected for hooves; use spike count and bbox velocity.")
    print()

    for landmark_name in args.landmarks:
        points = _points_from_frames(frames, args.track, landmark_name)
        metrics = summarize_trajectory(points, landmark_name)
        status_counts = _status_breakdown_from_frames(frames, args.track, landmark_name)
        print_landmark_report(landmark_name, metrics, status_counts)

    unmapped = [
        name
        for name in CANONICAL_LANDMARKS
        if name not in ANIMALPOSE_CANONICAL_COVERAGE and name != "spine_mid"
    ]
    print("Canonical landmarks not provided by AnimalPose:")
    print(", ".join(unmapped))


if __name__ == "__main__":
    main()
