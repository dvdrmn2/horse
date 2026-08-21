#!/usr/bin/env python3
"""Explain per-frame view classification on exported trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from landmarks.view import estimate_view_from_horse
from models.horse import HorseInstance
from models.landmark import Landmark, LandmarkSet
from models.landmark_types import LandmarkSource, LandmarkStatus


def horse_from_frame(frame: dict, track_id: int | None = None) -> HorseInstance | None:
    horses = frame.get("horses", [])
    if not horses:
        return None
    if track_id is None:
        horse_data = horses[0]
    else:
        horse_data = next((item for item in horses if item.get("track_id") == track_id), None)
    if horse_data is None:
        return None

    landmarks = []
    for index, item in enumerate(horse_data.get("landmarks", [])):
        landmarks.append(
            Landmark(
                name=item["name"],
                index=index,
                x=item.get("x"),
                y=item.get("y"),
                confidence=item.get("confidence"),
                status=LandmarkStatus(item.get("status", "unavailable")),
                source=LandmarkSource(item.get("source", "none")),
                visible=item.get("visible", False),
                outlier=item.get("outlier", False),
            )
        )

    return HorseInstance(
        track_id=horse_data["track_id"],
        bbox=horse_data["bbox"],
        detection_confidence=horse_data.get("detection_confidence", 1.0),
        class_id=0,
        landmarks=LandmarkSet(
            landmarks=landmarks,
            schema=tuple(item["name"] for item in horse_data.get("landmarks", [])),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug view classification on trajectories.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--track", type=int, default=1)
    parser.add_argument("--sample", type=int, default=5, help="Sample frames per classified view")
    args = parser.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    examples: dict[str, list[str]] = {}

    for frame in payload.get("frames", []):
        horse = horse_from_frame(frame, args.track)
        if horse is None:
            continue
        estimate = estimate_view_from_horse(horse)
        counts[estimate.view] += 1
        reasons[estimate.reason] += 1
        bucket = examples.setdefault(estimate.view, [])
        if len(bucket) < args.sample:
            bucket.append(f"frame {frame['frame_index']}: {estimate.reason}")

    print(f"File: {args.path}")
    print(f"Track: {args.track}")
    print(f"Stored dominant view: {payload.get('summary', {}).get('dominant_view')}")
    print()
    print("Recomputed view counts:")
    for view, count in counts.most_common():
        print(f"  {view:8s} {count}")
    print()
    print("Top reasons:")
    for reason, count in reasons.most_common(8):
        print(f"  [{count:4d}] {reason}")
    print()
    for view, samples in sorted(examples.items()):
        print(f"Examples ({view}):")
        for line in samples:
            print(f"  {line}")
        print()


if __name__ == "__main__":
    main()
