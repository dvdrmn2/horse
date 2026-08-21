#!/usr/bin/env python3
"""Analyze temporal view-label transitions and smoothing flip patterns."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diagnose_rear_view import analyze_video, _parse_video_path


def _transition_pattern(labels: list[str]) -> str:
    if len(labels) < 2:
        return "static"
    changes = sum(1 for index in range(1, len(labels)) if labels[index] != labels[index - 1])
    rate = changes / (len(labels) - 1)
    if rate < 0.05:
        return "stable_runs"
    if rate < 0.20:
        return "moderate_transitions"
    return "noisy_flip_flop"


def _run_lengths(labels: list[str]) -> list[tuple[str, int]]:
    if not labels:
        return []
    runs: list[tuple[str, int]] = []
    current = labels[0]
    length = 1
    for label in labels[1:]:
        if label == current:
            length += 1
        else:
            runs.append((current, length))
            current = label
            length = 1
    runs.append((current, length))
    return runs


def analyze_transitions(trajectory_path: Path, *, track_id: int | None = None) -> dict:
    frames = analyze_video(trajectory_path, track_id=track_id)
    raw_labels = [item.raw.view_label or item.raw.dominant_view for item in frames]
    smoothed_labels = [item.smoothed_label or item.smoothed_view for item in frames]
    raw_classes = [item.raw.classification for item in frames]
    smoothed_classes = [item.smoothed_classification for item in frames]

    raw_runs = _run_lengths(raw_labels)
    smoothed_runs = _run_lengths(smoothed_labels)
    smoothing_changes = sum(
        1 for item in frames if item.raw.dominant_view != item.smoothed_view
    )

    transitions = Counter()
    for index in range(1, len(smoothed_labels)):
        transitions[(smoothed_labels[index - 1], smoothed_labels[index])] += 1

    view, gait, slug = _parse_video_path(trajectory_path)
    return {
        "video": f"{view}/{gait}/{slug}",
        "frames": len(frames),
        "raw_label_counts": dict(Counter(raw_labels)),
        "smoothed_label_counts": dict(Counter(smoothed_labels)),
        "raw_classification_counts": dict(Counter(raw_classes)),
        "smoothed_classification_counts": dict(Counter(smoothed_classes)),
        "smoothing_label_changes": smoothing_changes,
        "raw_transition_pattern": _transition_pattern(raw_labels),
        "smoothed_transition_pattern": _transition_pattern(smoothed_labels),
        "median_raw_run_length": _median_run(smoothed_runs),
        "median_smoothed_run_length": _median_run(smoothed_runs),
        "top_transitions": [
            {"from": left, "to": right, "count": count}
            for (left, right), count in transitions.most_common(8)
        ],
        "longest_smoothed_runs": sorted(smoothed_runs, key=lambda item: item[1], reverse=True)[:6],
    }


def _median_run(runs: list[tuple[str, int]]) -> float:
    if not runs:
        return 0.0
    lengths = sorted(length for _, length in runs)
    mid = len(lengths) // 2
    if len(lengths) % 2:
        return float(lengths[mid])
    return (lengths[mid - 1] + lengths[mid]) / 2.0


def render_markdown(reports: list[dict]) -> str:
    lines = [
        "# View Transition Analysis",
        "",
        "Inspects whether temporal smoothing produces stable runs or noisy flip-flops.",
        "",
        "| Video | Frames | Raw pattern | Smoothed pattern | Smoothing changes | Insufficient | Ambiguous (smoothed) |",
        "|-------|--------|-------------|------------------|-------------------|--------------|----------------------|",
    ]
    for report in reports:
        raw_counts = report["raw_classification_counts"]
        smooth_counts = report["smoothed_classification_counts"]
        lines.append(
            f"| {report['video']} | {report['frames']} | {report['raw_transition_pattern']} | "
            f"{report['smoothed_transition_pattern']} | {report['smoothing_label_changes']} | "
            f"{raw_counts.get('insufficient', 0)} | {smooth_counts.get('ambiguous', 0)} |"
        )

    lines.extend(["", "## Detail", ""])
    for report in reports:
        lines.append(f"### {report['video']}")
        lines.append(f"- Raw labels: {report['raw_label_counts']}")
        lines.append(f"- Smoothed labels: {report['smoothed_label_counts']}")
        lines.append(f"- Raw classifications: {report['raw_classification_counts']}")
        lines.append(f"- Smoothed classifications: {report['smoothed_classification_counts']}")
        lines.append(f"- Median smoothed run length: {report['median_smoothed_run_length']:.1f} frames")
        if report["top_transitions"]:
            lines.append("- Top transitions:")
            for item in report["top_transitions"]:
                lines.append(f"  - {item['from']} → {item['to']}: {item['count']}")
        if report["longest_smoothed_runs"]:
            lines.append("- Longest smoothed runs:")
            for label, length in report["longest_smoothed_runs"]:
                lines.append(f"  - {label}: {length} frames")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze view label transitions over time.")
    parser.add_argument("root", type=Path, nargs="?", default=ROOT / "output" / "validation")
    parser.add_argument("--view", default=None)
    parser.add_argument("--track", type=int, default=None)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    paths = sorted(root.glob(f"{args.view}/*/*/trajectories.json")) if args.view else sorted(
        root.glob("*/*/*/trajectories.json")
    )
    if not paths:
        raise SystemExit("No trajectories found")

    reports = [analyze_transitions(path, track_id=args.track) for path in paths]
    markdown = render_markdown(reports)
    output_root = paths[0].parents[3] if "validation" in paths[0].parts else paths[0].parent
    json_path = output_root / "view_transition_analysis.json"
    md_path = output_root / "view_transition_analysis.md"
    json_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
