#!/usr/bin/env python3
"""Diagnose rear-view classification: per-frame features and scores over time."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from landmarks.view import ViewDiagnosticSnapshot, ViewScores, ViewSmoother, build_view_diagnostics
from models.horse import HorseInstance
from models.landmark import Landmark, LandmarkSet
from models.landmark_types import LandmarkSource, LandmarkStatus

FEATURE_KEYS = (
    "stifle_separation_px",
    "stifle_separation_ratio",
    "body_width_px",
    "body_length_px",
    "width_length_ratio",
    "head_visibility",
    "tail_visibility",
    "tail_head_absent",
    "left_right_symmetry",
    "visible_landmarks",
)


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
                raw_x=item.get("raw_x"),
                raw_y=item.get("raw_y"),
                raw_confidence=item.get("raw_confidence"),
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


def _primary_track(summary: dict) -> int | None:
    stats = summary.get("track_stats", {})
    if not stats:
        return None
    best_track = None
    best_score = -1
    for key, value in stats.items():
        track_id = int(key)
        score = value.get("landmark_observations", 0)
        if score > best_score:
            best_track = track_id
            best_score = score
    return best_track


def _parse_video_path(trajectory_path: Path) -> tuple[str, str, str]:
    parts = trajectory_path.parts
    view = gait = slug = "unknown"
    if "validation" in parts:
        idx = parts.index("validation")
        if idx + 3 < len(parts):
            view = parts[idx + 1]
            gait = parts[idx + 2]
            slug = parts[idx + 3]
    return view, gait, slug


@dataclass
class FrameDiagnostic:
    frame_index: int
    raw: ViewDiagnosticSnapshot
    smoothed_distribution: dict[str, float]
    smoothed_view: str


def analyze_video(
    trajectory_path: Path,
    *,
    track_id: int | None = None,
    smooth_window: int = 15,
) -> list[FrameDiagnostic]:
    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    primary_track = track_id if track_id is not None else _primary_track(payload.get("summary", {}))
    smoother = ViewSmoother(window=smooth_window)
    diagnostics: list[FrameDiagnostic] = []

    for frame in payload.get("frames", []):
        horse = horse_from_frame(frame, primary_track)
        if horse is None:
            continue

        raw = build_view_diagnostics(horse, frame_index=frame.get("frame_index"))
        smoothed = smoother.update(
            ViewScores(
                side=raw.raw_scores["side"],
                front=raw.raw_scores["front"],
                rear=raw.raw_scores["rear"],
                oblique=raw.raw_scores["oblique"],
            )
        )
        diagnostics.append(
            FrameDiagnostic(
                frame_index=raw.frame_index or 0,
                raw=raw,
                smoothed_distribution=smoothed.distribution,
                smoothed_view=smoothed.view,
            )
        )

    return diagnostics


def _mean_feature(frames: list[FrameDiagnostic], key: str) -> float | None:
    values = [
        item.raw.features[key]
        for item in frames
        if key in item.raw.features and not math.isnan(item.raw.features[key])
    ]
    if not values:
        return None
    return statistics.mean(values)


def _pick_example(
    frames: list[FrameDiagnostic],
    *,
    label: str,
    reverse: bool = False,
) -> FrameDiagnostic | None:
    if not frames:
        return None

    def sort_key(item: FrameDiagnostic) -> float:
        return item.raw.distribution.get(label, 0.0)

    return sorted(frames, key=sort_key, reverse=not reverse)[0]


def _bucket_frames(frames: list[FrameDiagnostic]) -> dict[str, list[FrameDiagnostic]]:
    buckets: dict[str, list[FrameDiagnostic]] = defaultdict(list)
    for item in frames:
        distribution = item.raw.distribution
        winner = max(("side", "front", "rear", "oblique"), key=lambda name: distribution.get(name, 0.0))
        buckets[winner].append(item)
    return buckets


def render_video_section(
    trajectory_path: Path,
    frames: list[FrameDiagnostic],
) -> list[str]:
    view, gait, slug = _parse_video_path(trajectory_path)
    buckets = _bucket_frames(frames)
    counts = Counter(item.raw.dominant_view for item in frames)

    lines = [
        f"## {view}/{gait}/{slug}",
        "",
        f"Frames analyzed: {len(frames)}",
        f"Dominant labels: {', '.join(f'{name}={count}' for name, count in counts.most_common())}",
        f"Bucket counts (raw distribution winner): rear={len(buckets['rear'])}, "
        f"oblique={len(buckets['oblique'])}, side={len(buckets['side'])}, "
        f"front={len(buckets['front'])}",
        "",
        "### Mean features by bucket",
        "",
        "| Feature | rear bucket | oblique bucket | delta |",
        "|---------|-------------|----------------|-------|",
    ]

    for key in FEATURE_KEYS:
        rear_mean = _mean_feature(buckets["rear"], key)
        oblique_mean = _mean_feature(buckets["oblique"], key)
        if rear_mean is None and oblique_mean is None:
            continue
        delta = None
        if rear_mean is not None and oblique_mean is not None:
            delta = rear_mean - oblique_mean
        lines.append(
            f"| {key} | {_fmt(rear_mean)} | {_fmt(oblique_mean)} | {_fmt(delta)} |"
        )

    lines.extend(["", "### Example frames", ""])

    oblique_example = _pick_example(buckets["oblique"], label="oblique")
    rear_example = _pick_example(buckets["rear"], label="rear", reverse=True)

    if oblique_example:
        lines.extend(["#### Typical oblique-classified frame", "", "```", oblique_example.raw.render(), "```", ""])
    if rear_example:
        lines.extend(["#### Typical rear-classified frame", "", "```", rear_example.raw.render(), "```", ""])

    if oblique_example and rear_example:
        lines.extend(["#### Feature contrast (rear − oblique example)", ""])
        for key in FEATURE_KEYS:
            oblique_value = oblique_example.raw.features.get(key, float("nan"))
            rear_value = rear_example.raw.features.get(key, float("nan"))
            if math.isnan(oblique_value) and math.isnan(rear_value):
                continue
            delta = rear_value - oblique_value if not math.isnan(oblique_value) and not math.isnan(rear_value) else float("nan")
            lines.append(f"- **{key}**: oblique={_fmt(oblique_value)}, rear={_fmt(rear_value)}, delta={_fmt(delta)}")

    smoothing_changes = sum(1 for item in frames if item.raw.dominant_view != item.smoothed_view)
    lines.extend(
        [
            "",
            "### Temporal smoothing",
            "",
            f"- Frames where smoothing changes the discrete label: {smoothing_changes}/{len(frames)}",
        ]
    )

    return lines


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float) and math.isnan(value):
        return "n/a"
    return f"{value:.3f}"


def snapshot_to_dict(item: FrameDiagnostic) -> dict:
    return {
        "frame_index": item.frame_index,
        "raw_features": item.raw.features,
        "raw_scores": item.raw.raw_scores,
        "raw_distribution": item.raw.distribution,
        "raw_dominant_view": item.raw.dominant_view,
        "smoothed_distribution": item.smoothed_distribution,
        "smoothed_view": item.smoothed_view,
        "cues": list(item.raw.cues),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose rear-view feature/score behavior.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=ROOT / "output" / "validation",
        help="Validation output root",
    )
    parser.add_argument("--view", default="rear", help="Only analyze videos from this folder view")
    parser.add_argument("--track", type=int, default=None, help="Override primary track id")
    parser.add_argument("--smooth-window", type=int, default=15, help="Temporal smoothing window")
    parser.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Analyze a single trajectories.json instead of all videos under root",
    )
    args = parser.parse_args()

    if args.video:
        trajectory_paths = [args.video.expanduser().resolve()]
    else:
        root = args.root.expanduser().resolve()
        trajectory_paths = sorted(root.glob(f"{args.view}/*/*/trajectories.json"))

    if not trajectory_paths:
        raise SystemExit("No trajectories found")

    all_lines = [
        "# Rear View Feature Diagnostic",
        "",
        "Read-only dump of geometric features and view scores. Classifier logic is unchanged.",
        "Bucket labels use the raw per-frame score winner (rear vs oblique vs other).",
        "",
    ]

    json_payload = {}
    for path in trajectory_paths:
        frames = analyze_video(path, track_id=args.track, smooth_window=args.smooth_window)
        _, gait, slug = _parse_video_path(path)
        key = f"{args.view}/{gait}/{slug}"
        json_payload[key] = [snapshot_to_dict(item) for item in frames]
        all_lines.extend(render_video_section(path, frames))
        all_lines.append("")

    output_root = trajectory_paths[0].parents[3] if "validation" in trajectory_paths[0].parts else trajectory_paths[0].parent
    json_path = output_root / "rear_view_diagnostic.json"
    md_path = output_root / "rear_view_diagnostic.md"
    json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
    md_path.write_text("\n".join(all_lines), encoding="utf-8")

    print("\n".join(all_lines))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
