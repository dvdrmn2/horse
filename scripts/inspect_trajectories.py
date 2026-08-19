#!/usr/bin/env python3
"""Inspect exported horse pose trajectories and landmark coverage."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import CANONICAL_LANDMARKS
from landmarks.mapping import ANIMALPOSE_CANONICAL_COVERAGE, ANIMALPOSE_UNMAPPED_CANONICAL


def print_track_overview(payload: dict) -> None:
    summary = payload.get("summary", {})
    trajectories = payload.get("trajectories", {})
    track_stats = summary.get("track_stats", {})

    print("Tracks in session:")
    for track_id in summary.get("tracks", []):
        stats = track_stats.get(track_id) or track_stats.get(str(track_id), {})
        track_key = f"track_{track_id}"
        point_count = sum(len(points) for points in trajectories.get(track_key, {}).values())
        print(
            f"  track {track_id}: {stats.get('frames', '?')} frames, "
            f"{point_count} trajectory points"
        )
    print()


def _status_breakdown_for_track(payload: dict, track_id: int) -> dict[str, Counter]:
    breakdown: dict[str, Counter] = {}

    for frame in payload.get("frames", []):
        horse = next((item for item in frame.get("horses", []) if item.get("track_id") == track_id), None)
        if horse is None:
            continue

        for landmark in horse.get("landmarks", []):
            name = landmark.get("name")
            status = landmark.get("status", "unknown")
            if name is None:
                continue
            breakdown.setdefault(name, Counter())[status] += 1

    return breakdown


def _dominant_status(counter: Counter) -> str:
    if not counter:
        return "missing"
    return counter.most_common(1)[0][0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect exported horse pose trajectories.")
    parser.add_argument("path", type=Path, help="Path to *.trajectories.json")
    parser.add_argument(
        "--track",
        type=int,
        help="Track ID to inspect (use track overview to choose when multiple horses appear)",
    )
    parser.add_argument(
        "--all-tracks",
        action="store_true",
        help="Print coverage summary for every track",
    )
    parser.add_argument("--landmark", default="left_stifle", help="Landmark name")
    args = parser.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})

    print(f"Summary: {summary}")
    print_track_overview(payload)

    if args.all_tracks:
        for track_id in summary.get("tracks", []):
            args_copy = argparse.Namespace(**vars(args))
            args_copy.track = track_id
            _inspect_one_track(payload, args_copy)
            print()
        return

    if args.track is None:
        print("Multiple horses may appear in one video. Re-run with --track N or --all-tracks.")
        return

    _inspect_one_track(payload, args)


def _inspect_one_track(payload: dict, args: argparse.Namespace) -> None:
    trajectories = payload.get("trajectories", {})
    track_key = f"track_{args.track}"
    points = trajectories.get(track_key, {}).get(args.landmark, [])
    status_breakdown = _status_breakdown_for_track(payload, args.track)

    print(f"Inspecting: {track_key}")

    if args.landmark not in ANIMALPOSE_CANONICAL_COVERAGE and args.landmark != "spine_mid":
        print(
            f"Note: '{args.landmark}' is NOT produced directly by AnimalPose.\n"
            f"Unmapped canonical landmarks: {', '.join(ANIMALPOSE_UNMAPPED_CANONICAL)}\n"
            f"Try e.g. --landmark left_stifle, nose, withers, left_hind_hoof, or spine_mid (derived).\n"
        )

    print(f"{track_key}.{args.landmark}: {len(points)} trajectory points")
    for point in points[:10]:
        print(point)
    if len(points) > 10:
        print(f"... {len(points) - 10} more")

    print()
    print(f"Status breakdown for {track_key} (frames per landmark status):")
    for landmark_name in CANONICAL_LANDMARKS:
        counter = status_breakdown.get(landmark_name, Counter())
        dominant = _dominant_status(counter)
        detected = counter.get("detected", 0)
        derived = counter.get("derived", 0)
        unmapped = counter.get("unmapped", 0)
        not_visible = counter.get("not_visible", 0)
        low_conf = counter.get("low_confidence", 0)

        if landmark_name in ANIMALPOSE_CANONICAL_COVERAGE:
            adapter = "AnimalPose"
        elif landmark_name == "spine_mid":
            adapter = "derived"
        else:
            adapter = "unmapped"

        marker = " " if detected or derived else "-"
        print(
            f"  {marker} {landmark_name:22s}  "
            f"detected={detected:3d}  derived={derived:3d}  "
            f"unmapped={unmapped:3d}  not_visible={not_visible:3d}  "
            f"low_conf={low_conf:3d}  ({adapter}, dominant={dominant})"
        )


if __name__ == "__main__":
    main()
