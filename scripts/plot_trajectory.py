#!/usr/bin/env python3
"""Plot a landmark trajectory from exported session JSON."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from landmarks.body_frame import extract_body_points
from landmarks.temporal import (
    Point2D,
    compute_kinematics,
    detect_spike_indices,
    get_motion_profile,
    normalize_to_bbox,
)


def load_points(payload: dict, track_id: int, landmark_name: str) -> list[dict]:
    points = extract_body_points(payload.get("frames", []), track_id, landmark_name)
    for point in points:
        if point.get("bbox"):
            point["bbox_x"], point["bbox_y"] = normalize_to_bbox(point["x"], point["y"], point["bbox"])
        else:
            point["bbox_x"] = None
            point["bbox_y"] = None
    return points


def render_ascii_plot(points: list[dict], x_key: str, y_key: str, width: int = 72, height: int = 24) -> str:
    if not points:
        return "(no points)"

    xs = [point[x_key] for point in points]
    ys = [point[y_key] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)

    grid = [[" " for _ in range(width)] for _ in range(height)]

    for index, point in enumerate(points):
        col = int((point[x_key] - min_x) / span_x * (width - 1))
        row = int((point[y_key] - min_y) / span_y * (height - 1))
        marker = "*" if point.get("outlier") else str(index % 10)
        grid[row][col] = marker

    lines = ["+" + "-" * width + "+"]
    for row in grid:
        lines.append("|" + "".join(row) + "|")
    lines.append("+" + "-" * width + "+")
    lines.append(f"x: {min_x:.0f}..{max_x:.0f}   y: {min_y:.0f}..{max_y:.0f}   frames: {len(points)}")
    return "\n".join(lines)


def write_svg(
    path: Path,
    points: list[dict],
    *,
    title: str,
    x_key: str,
    y_key: str,
    spike_indices: set[int],
) -> None:
    if not points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>", encoding="utf-8")
        return

    xs = [point[x_key] for point in points]
    ys = [point[y_key] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1e-6)
    span_y = max(max_y - min_y, 1e-6)

    width, height, margin = 900, 520, 40

    def scale_x(value: float) -> float:
        return margin + (value - min_x) / span_x * (width - 2 * margin)

    def scale_y(value: float) -> float:
        return height - margin - (value - min_y) / span_y * (height - 2 * margin)

    polyline = " ".join(f"{scale_x(point[x_key]):.1f},{scale_y(point[y_key]):.1f}" for point in points)
    circles = []
    for index, point in enumerate(points):
        color = "#d62728" if index in spike_indices else ("#888888" if point.get("outlier") else "#1f77b4")
        radius = 5 if index in spike_indices else 3
        circles.append(
            f"<circle cx='{scale_x(point[x_key]):.1f}' cy='{scale_y(point[y_key]):.1f}' "
            f"r='{radius}' fill='{color}' />"
        )

    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
  <rect x='0' y='0' width='{width}' height='{height}' fill='#111' />
  <text x='{margin}' y='24' fill='#eee' font-size='18'>{title}</text>
  <polyline fill='none' stroke='#ff7f0e' stroke-width='2' points='{polyline}' />
  {''.join(circles)}
</svg>
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a landmark trajectory from session JSON.")
    parser.add_argument("path", type=Path, help="Path to *.trajectories.json")
    parser.add_argument("--track", type=int, required=True)
    parser.add_argument("--landmark", default="left_front_hoof")
    parser.add_argument(
        "--space",
        choices=("video", "bbox", "body"),
        default="video",
        help="Coordinate space for the plot",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional SVG output path (defaults to output/<track>_<landmark>_<space>.svg)",
    )
    args = parser.parse_args()

    payload = json.loads(args.path.read_text(encoding="utf-8"))
    points = load_points(payload, args.track, args.landmark)
    profile = get_motion_profile(args.landmark)
    samples = compute_kinematics(
        [Point2D(point["x"], point["y"], frame=point["frame"]) for point in points]
    )
    spike_indices = set(detect_spike_indices(samples, profile))

    if args.space == "bbox":
        x_key, y_key = "bbox_x", "bbox_y"
        space_label = "bbox-normalized"
    elif args.space == "body":
        x_key, y_key = "body_long", "body_lat"
        space_label = "body-axis-normalized"
        points = [
            point
            for point in points
            if point.get(x_key) is not None
            and point.get(y_key) is not None
            and point.get("body_frame_quality") != "unavailable"
        ]
        qualities = Counter(point.get("body_frame_quality") for point in points)
        print(f"body frame quality: {dict(qualities)}")
    else:
        x_key, y_key = "x", "y"
        space_label = "video pixels"

    print(f"track_{args.track}.{args.landmark} — {space_label}")
    print(f"motion profile: {profile.value}")
    print(f"points: {len(points)}   kinematic spikes: {len(spike_indices)}")
    print()
    print(render_ascii_plot(points, x_key, y_key))
    print()

    if points:
        velocities = [sample.velocity for sample in samples]
        print(
            "velocity px/frame:",
            f"mean={sum(velocities)/len(velocities):.1f}",
            f"max={max(velocities):.1f}",
        )
        if spike_indices:
            print("spike frames:", ", ".join(str(points[index]["frame"]) for index in sorted(spike_indices)))

    output_path = args.output or Path("output") / f"track_{args.track}_{args.landmark}_{args.space}.svg"
    write_svg(
        output_path,
        points,
        title=f"track {args.track} {args.landmark} ({space_label})",
        x_key=x_key,
        y_key=y_key,
        spike_indices=spike_indices,
    )
    print(f"Wrote SVG: {output_path}")


if __name__ == "__main__":
    main()
