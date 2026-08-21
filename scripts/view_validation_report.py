#!/usr/bin/env python3
"""Quantitative view-classification validation against folder ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from landmarks.view import ViewSmoother, compute_view_scores, extract_view_features
from models.horse import HorseInstance
from models.landmark import Landmark, LandmarkSet
from models.landmark_types import LandmarkSource, LandmarkStatus

GROUND_TRUTH_VIEWS = ("side", "front", "rear")
PREDICTED_VIEWS = ("side", "front", "rear", "oblique", "unknown")


@dataclass
class FrameViewResult:
    frame_index: int
    view: str
    confidence: float
    distribution: dict[str, float]


@dataclass
class VideoViewResult:
    actual_view: str
    gait: str
    slug: str
    path: str
    primary_track: int | None
    horse_frames: int
    dominant_view: str
    dominant_confidence: float
    view_distribution: dict[str, float]
    frame_counts: dict[str, int] = field(default_factory=dict)
    frame_results: list[FrameViewResult] = field(default_factory=list)


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


def analyze_trajectory_views(
    trajectory_path: Path,
    *,
    track_id: int | None = None,
    smooth_window: int = 15,
    use_stored: bool = False,
) -> VideoViewResult:
    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    frames = payload.get("frames", [])
    actual_view, gait, slug = _parse_video_path(trajectory_path)
    primary_track = track_id if track_id is not None else _primary_track(summary)

    frame_results: list[FrameViewResult] = []
    smoother = ViewSmoother(window=smooth_window)

    for frame in frames:
        horse = horse_from_frame(frame, primary_track)
        if horse is None:
            continue

        if use_stored and frame.get("view_distribution"):
            estimate_view = frame.get("view", "unknown")
            estimate_confidence = float(frame.get("view_confidence", 0.0))
            distribution = dict(frame.get("view_distribution", {}))
        else:
            raw_scores = compute_view_scores(extract_view_features(horse))
            estimate = smoother.update(raw_scores)
            estimate_view = estimate.view
            estimate_confidence = estimate.confidence
            distribution = estimate.distribution

        frame_results.append(
            FrameViewResult(
                frame_index=frame.get("frame_index", len(frame_results)),
                view=estimate_view,
                confidence=estimate_confidence,
                distribution=distribution,
            )
        )

    frame_counts = Counter(result.view for result in frame_results)
    total_frames = len(frame_results)

    if use_stored and summary.get("view_distribution"):
        view_distribution = dict(summary.get("view_distribution", {}))
        dominant_view = summary.get("dominant_view", "unknown")
        dominant_confidence = float(summary.get("view_confidence", 0.0))
    else:
        aggregate = {"side": 0.0, "front": 0.0, "rear": 0.0, "oblique": 0.0}
        for result in frame_results:
            for key in aggregate:
                aggregate[key] += result.distribution.get(key, 0.0)
        total = sum(aggregate.values()) or 1.0
        view_distribution = {key: value / total for key, value in aggregate.items()}
        dominant_view = max(view_distribution, key=view_distribution.get)
        dominant_confidence = view_distribution[dominant_view]

    return VideoViewResult(
        actual_view=actual_view,
        gait=gait,
        slug=slug,
        path=str(trajectory_path),
        primary_track=primary_track,
        horse_frames=total_frames,
        dominant_view=dominant_view,
        dominant_confidence=dominant_confidence,
        view_distribution=view_distribution,
        frame_counts=dict(frame_counts),
        frame_results=frame_results,
    )


@dataclass
class ConfusionMatrix:
    counts: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def add(self, actual: str, predicted: str) -> None:
        self.counts[actual][predicted] += 1

    def accuracy_for(self, actual: str) -> float | None:
        row = self.counts.get(actual, {})
        total = sum(row.values())
        if total == 0:
            return None
        return row.get(actual, 0) / total

    def overall_accuracy(self) -> float | None:
        correct = 0
        total = 0
        for actual in GROUND_TRUTH_VIEWS:
            row = self.counts.get(actual, {})
            correct += row.get(actual, 0)
            total += sum(row.values())
        if total == 0:
            return None
        return correct / total


def build_confusion(results: list[VideoViewResult], *, frame_level: bool) -> ConfusionMatrix:
    matrix = ConfusionMatrix()
    for result in results:
        if frame_level:
            for frame in result.frame_results:
                matrix.add(result.actual_view, frame.view)
        else:
            matrix.add(result.actual_view, result.dominant_view)
    return matrix


def render_confusion_table(matrix: ConfusionMatrix, *, title: str) -> list[str]:
    lines = [f"### {title}", ""]
    header = "| Actual \\ Predicted | " + " | ".join(PREDICTED_VIEWS) + " | Total |"
    sep = "|" + "|".join(["---"] * (len(PREDICTED_VIEWS) + 2)) + "|"
    lines.extend([header, sep])

    for actual in GROUND_TRUTH_VIEWS:
        row = matrix.counts.get(actual, {})
        cells = [str(row.get(predicted, 0)) for predicted in PREDICTED_VIEWS]
        total = sum(row.values())
        lines.append(f"| {actual} | " + " | ".join(cells) + f" | {total} |")

    lines.append("")
    for actual in GROUND_TRUTH_VIEWS:
        accuracy = matrix.accuracy_for(actual)
        if accuracy is None:
            lines.append(f"- **{actual} accuracy**: n/a (no samples)")
        else:
            lines.append(f"- **{actual} accuracy**: {accuracy:.1%}")

    overall = matrix.overall_accuracy()
    if overall is not None:
        lines.append(f"- **overall accuracy**: {overall:.1%}")
    lines.append("")
    return lines


def result_to_dict(result: VideoViewResult) -> dict:
    return {
        "actual_view": result.actual_view,
        "gait": result.gait,
        "slug": result.slug,
        "path": result.path,
        "primary_track": result.primary_track,
        "horse_frames": result.horse_frames,
        "dominant_view": result.dominant_view,
        "dominant_confidence": result.dominant_confidence,
        "view_distribution": result.view_distribution,
        "frame_counts": result.frame_counts,
        "frame_results": [
            {
                "frame_index": frame.frame_index,
                "view": frame.view,
                "confidence": frame.confidence,
                "distribution": frame.distribution,
            }
            for frame in result.frame_results
        ],
    }


def render_markdown(results: list[VideoViewResult]) -> str:
    dominant_matrix = build_confusion(results, frame_level=False)
    frame_matrix = build_confusion(results, frame_level=True)

    lines = [
        "# View Classification Validation",
        "",
        "Ground truth is the validation folder (`side/`, `front/`, `rear/`).",
        "Predictions are recomputed from landmarks with score-based smoothing.",
        "",
        "## Per-video summary",
        "",
        "| Actual | Gait | Slug | Track | Horse frames | Dominant | Confidence | side | front | rear | oblique |",
        "|--------|------|------|-------|--------------|----------|------------|------|-------|------|---------|",
    ]

    for result in results:
        dist = result.view_distribution
        lines.append(
            f"| {result.actual_view} | {result.gait} | {result.slug} | "
            f"{result.primary_track or '-'} | {result.horse_frames} | {result.dominant_view} | "
            f"{result.dominant_confidence:.2f} | "
            f"{dist.get('side', 0):.0%} | {dist.get('front', 0):.0%} | "
            f"{dist.get('rear', 0):.0%} | {dist.get('oblique', 0):.0%} |"
        )

    lines.extend(["", "## Frame-level view counts", ""])
    for result in results:
        counts_text = ", ".join(f"{view}={count}" for view, count in sorted(result.frame_counts.items()))
        lines.append(f"- **{result.actual_view}/{result.gait}/{result.slug}**: {counts_text}")

    lines.extend(["", "## Confusion matrix (dominant view per video)", ""])
    lines.extend(render_confusion_table(dominant_matrix, title="Video-level"))
    lines.extend(render_confusion_table(frame_matrix, title="Frame-level"))

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate view classification against folder ground truth.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=ROOT / "output" / "validation",
        help="Validation output root",
    )
    parser.add_argument("--track", type=int, default=None, help="Override primary track id")
    parser.add_argument("--smooth-window", type=int, default=15, help="Temporal smoothing window")
    parser.add_argument(
        "--use-stored",
        action="store_true",
        help="Use view fields stored in trajectories instead of recomputing",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    trajectory_paths = sorted(root.glob("*/*/*/trajectories.json"))
    if not trajectory_paths:
        raise SystemExit(f"No trajectories found under {root}")

    results = [
        analyze_trajectory_views(
            path,
            track_id=args.track,
            smooth_window=args.smooth_window,
            use_stored=args.use_stored,
        )
        for path in trajectory_paths
    ]

    markdown = render_markdown(results)
    json_path = root / "view_validation_report.json"
    md_path = root / "view_validation_report.md"
    json_path.write_text(
        json.dumps([result_to_dict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    md_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
