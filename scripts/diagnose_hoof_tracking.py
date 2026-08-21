#!/usr/bin/env python3
"""Diagnose hoof tracking: confidence timeline and left/right limb identity."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.landmark_profiles import get_landmark_profile
from landmarks.temporal import Point2D, compute_kinematics, detect_spike_indices, get_motion_profile

FRONT_CHAIN = (
    ("left_elbow", "left_carpus"),
    ("left_carpus", "left_front_hoof"),
    ("right_elbow", "right_carpus"),
    ("right_carpus", "right_front_hoof"),
)


def _lm(frame: dict, track_id: int, name: str) -> dict | None:
    horse = next((item for item in frame.get("horses", []) if item.get("track_id") == track_id), None)
    if horse is None:
        return None
    return next((item for item in horse.get("landmarks", []) if item.get("name") == name), None)


def _point(lm: dict | None) -> tuple[float, float] | None:
    if lm is None:
        return None
    x = lm.get("raw_x", lm.get("x"))
    y = lm.get("raw_y", lm.get("y"))
    if x is None or y is None:
        return None
    return float(x), float(y)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def load_series(payload: dict, track_id: int, landmark_name: str) -> list[dict]:
    rows = []
    for frame in payload.get("frames", []):
        lm = _lm(frame, track_id, landmark_name)
        if lm is None:
            continue
        raw = _point(lm)
        if raw is None:
            continue
        rows.append(
            {
                "frame": frame["frame_index"],
                "raw_x": raw[0],
                "raw_y": raw[1],
                "raw_confidence": lm.get("raw_confidence", lm.get("confidence")),
                "validated": lm.get("x") is not None and lm.get("y") is not None,
                "outlier": lm.get("outlier", False),
                "visible": lm.get("visible", False),
            }
        )
    return rows


def spike_frames(rows: list[dict], landmark_name: str) -> list[int]:
    points = [Point2D(row["raw_x"], row["raw_y"], frame=row["frame"]) for row in rows]
    samples = compute_kinematics(points)
    indices = detect_spike_indices(samples, get_motion_profile(landmark_name))
    return [rows[index]["frame"] for index in indices]


def print_confidence_timeline(rows: list[dict], *, sample_every: int = 20) -> None:
    print("Frame   RawConf  Valid  Outlier  RawX   RawY")
    for index, row in enumerate(rows):
        if index % sample_every != 0 and index != len(rows) - 1:
            continue
        conf = row["raw_confidence"]
        conf_text = f"{conf:.2f}" if conf is not None else " n/a"
        print(
            f"{row['frame']:5d}  {conf_text:>7s}  "
            f"{'yes' if row['validated'] else ' no '}     "
            f"{'yes' if row['outlier'] else ' no '}      "
            f"{row['raw_x']:6.0f} {row['raw_y']:6.0f}"
        )


def inspect_limb_identity(payload: dict, track_id: int, frame_index: int) -> None:
    frame = next(item for item in payload["frames"] if item["frame_index"] == frame_index)
    print(f"\nLimb geometry at frame {frame_index}:")
    for start_name, end_name in FRONT_CHAIN:
        start = _point(_lm(frame, track_id, start_name))
        end = _point(_lm(frame, track_id, end_name))
        if start and end:
            print(f"  {start_name:18s} → {end_name:18s}: {_dist(start, end):6.1f}px")
        else:
            print(f"  {start_name:18s} → {end_name:18s}: missing")

    left_hoof = _point(_lm(frame, track_id, "left_front_hoof"))
    right_hoof = _point(_lm(frame, track_id, "right_front_hoof"))
    if left_hoof and right_hoof:
        print(f"  left/right hoof separation: {_dist(left_hoof, right_hoof):.1f}px")
        if left_hoof[0] > right_hoof[0]:
            print("  note: left hoof is to the right of right hoof in image coordinates")


def summarize_confidence_at_spikes(rows: list[dict], spike_frame_list: list[int]) -> None:
    by_frame = {row["frame"]: row for row in rows}
    low_conf_spikes = 0
    high_conf_spikes = 0
    for frame_index in spike_frame_list:
        row = by_frame.get(frame_index)
        if row is None:
            continue
        conf = row.get("raw_confidence") or 0.0
        if conf < 0.3:
            low_conf_spikes += 1
        elif conf >= 0.5:
            high_conf_spikes += 1
    print(
        f"Spike confidence: low (<0.3)={low_conf_spikes}, "
        f"high (>=0.5)={high_conf_spikes}, total spikes={len(spike_frame_list)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose hoof tracking and identity.")
    parser.add_argument("path", type=Path)
    parser.add_argument("--track", type=int, default=1)
    parser.add_argument("--landmark", default="right_front_hoof")
    parser.add_argument("--compare", default="left_front_hoof")
    parser.add_argument("--inspect-spikes", type=int, default=3)
    args = parser.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    profile = get_landmark_profile(args.landmark)
    rows = load_series(payload, args.track, args.landmark)
    compare_rows = load_series(payload, args.track, args.compare)
    spikes = spike_frames(rows, args.landmark)

    print(f"Diagnosing {args.landmark} on track {args.track}")
    print(f"Profile: motion={profile.motion}, identity_swap_risk={profile.identity_swap_risk}")
    print(f"Frames with raw model output: {len(rows)}")
    print(f"Kinematic spikes: {len(spikes)}")
    summarize_confidence_at_spikes(rows, spikes)
    print()
    print_confidence_timeline(rows, sample_every=max(1, len(rows) // 20))

    print()
    print(f"Compare landmark {args.compare}: {len(compare_rows)} raw frames, "
          f"{len(spike_frames(compare_rows, args.compare))} spikes")

    for frame_index in spikes[: args.inspect_spikes]:
        inspect_limb_identity(payload, args.track, frame_index)


if __name__ == "__main__":
    main()
