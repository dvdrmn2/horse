#!/usr/bin/env python3
"""Inspect exported horse pose trajectories and landmark coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    trajectories = payload.get("trajectories", {})

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

    print(f"Inspecting: {track_key}")

    if args.landmark not in ANIMALPOSE_CANONICAL_COVERAGE:
        print(
            f"Note: '{args.landmark}' is NOT produced by the AnimalPose backend.\n"
            f"Unmapped canonical landmarks: {', '.join(ANIMALPOSE_UNMAPPED_CANONICAL)}\n"
            f"Try e.g. --landmark left_stifle, nose, withers, or left_hind_hoof.\n"
        )

    print(f"{track_key}.{args.landmark}: {len(points)} points")
    for point in points[:10]:
        print(point)
    if len(points) > 10:
        print(f"... {len(points) - 10} more")

    print()
    print(f"Coverage for {track_key} (visible frames per landmark):")
    all_landmarks = sorted(
        {
            landmark_name
            for track_data in trajectories.values()
            for landmark_name in track_data.keys()
        }
    )
    track_data = trajectories.get(track_key, {})
    for landmark_name in all_landmarks:
        count = len(track_data.get(landmark_name, []))
        source = "AnimalPose" if landmark_name in ANIMALPOSE_CANONICAL_COVERAGE else "unmapped"
        marker = " " if count else "-"
        print(f"  {marker} {landmark_name:22s} {count:4d}  ({source})")


if __name__ == "__main__":
    main()
