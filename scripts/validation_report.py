#!/usr/bin/env python3
"""Summarize validation study results across view/gait videos."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.config import CANONICAL_LANDMARKS
from landmarks.body_frame import body_velocity, extract_body_points
from landmarks.mapping import ANIMALPOSE_CANONICAL_COVERAGE
from landmarks.temporal import Point2D, compute_kinematics, detect_spike_indices, get_motion_profile, summarize_trajectory

KEY_LANDMARKS = (
    "withers",
    "left_stifle",
    "left_carpus",
    "left_front_hoof",
    "right_front_hoof",
    "left_hind_hoof",
    "right_hind_hoof",
    "tail_head",
)


@dataclass
class VideoReport:
    view: str
    gait: str
    slug: str
    path: str
    primary_track: int | None
    frames: int
    frames_with_horses: int
    dominant_view: str
    view_match: bool
    avg_reliable_landmarks: float
    detected_landmark_rate: float
    issues: list[str]
    hoof_metrics: dict[str, dict]


def _load_points(frames: list[dict], track_id: int, landmark_name: str) -> list[dict]:
    points = extract_body_points(frames, track_id, landmark_name)
    return points


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


def _detected_rate(frames: list[dict], track_id: int) -> float:
    total_slots = 0
    detected = 0
    for frame in frames:
        horse = next((item for item in frame.get("horses", []) if item.get("track_id") == track_id), None)
        if horse is None:
            continue
        for landmark in horse.get("landmarks", []):
            if landmark.get("name") not in ANIMALPOSE_CANONICAL_COVERAGE:
                continue
            total_slots += 1
            if landmark.get("status") in {"detected", "low_confidence"}:
                detected += 1
    return detected / total_slots if total_slots else 0.0


def _body_metrics(points: list[dict]) -> dict:
    body_points = [
        (point["body_long"], point["body_lat"])
        for point in points
        if point.get("body_long") is not None
        and point.get("body_lat") is not None
        and point.get("body_frame_quality") in {None, "reliable", "low_confidence"}
    ]
    if len(body_points) < 2:
        return {"body_frames": len(body_points), "body_mean_velocity": None, "body_p95_velocity": None}

    velocities = [
        body_velocity(body_points[index - 1][0], body_points[index - 1][1], body_points[index][0], body_points[index][1])
        for index in range(1, len(body_points))
    ]
    ordered = sorted(velocities)
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "body_frames": len(body_points),
        "body_mean_velocity": sum(velocities) / len(velocities),
        "body_p95_velocity": ordered[p95_index],
    }


def analyze_video(trajectory_path: Path, *, expected_view: str | None = None) -> VideoReport:
    payload = json.loads(trajectory_path.read_text(encoding="utf-8"))
    summary = payload.get("summary", {})
    frames = payload.get("frames", [])
    track_id = _primary_track(summary)

    parts = trajectory_path.parts
    view = expected_view or "unknown"
    gait = "unknown"
    slug = trajectory_path.parent.name
    if "validation" in parts:
        idx = parts.index("validation")
        if idx + 2 < len(parts):
            view = parts[idx + 1]
            gait = parts[idx + 2]
            slug = parts[idx + 3]

    issues: list[str] = []
    dominant_view = summary.get("dominant_view", "unknown")
    view_match = dominant_view == view or dominant_view == "unknown"
    if not view_match:
        issues.append(f"view_mismatch(expected={view}, dominant={dominant_view})")

    frames_with_horses = summary.get("frames_with_horses", 0)
    total_frames = summary.get("frames", 0)
    if total_frames and frames_with_horses / total_frames < 0.25:
        issues.append("low_horse_coverage")

    track_stats = summary.get("track_stats", {})
    if len(track_stats) > 2:
        issues.append("multi_horse_tracks")

    hoof_metrics: dict[str, dict] = {}
    if track_id is not None:
        for landmark_name in KEY_LANDMARKS:
            points = _load_points(frames, track_id, landmark_name)
            metrics = summarize_trajectory(points, landmark_name)
            body = _body_metrics(points)
            samples = compute_kinematics(
                [Point2D(point["x"], point["y"], frame=point.get("frame")) for point in points]
            )
            spikes = detect_spike_indices(samples, get_motion_profile(landmark_name))
            hoof_metrics[landmark_name] = {
                "frames": metrics.frames,
                "spikes": metrics.spike_count,
                "qc_outlier_rate": metrics.legacy_outlier_rate,
                "mean_velocity_px": metrics.mean_velocity,
                **body,
            }
            if landmark_name.endswith("_hoof") and metrics.spike_count > 2:
                issues.append(f"{landmark_name}_spikes={metrics.spike_count}")
            if landmark_name == "withers" and metrics.legacy_outlier_rate and metrics.legacy_outlier_rate > 0.15:
                issues.append(f"withers_qc_outliers={metrics.legacy_outlier_rate:.0%}")

        detected_rate = _detected_rate(frames, track_id)
        if detected_rate < 0.35:
            issues.append(f"low_detection_rate={detected_rate:.0%}")
    else:
        detected_rate = 0.0
        issues.append("no_primary_track")

    return VideoReport(
        view=view,
        gait=gait,
        slug=slug,
        path=str(trajectory_path),
        primary_track=track_id,
        frames=total_frames,
        frames_with_horses=frames_with_horses,
        dominant_view=dominant_view,
        view_match=view_match,
        avg_reliable_landmarks=float(summary.get("avg_reliable_landmarks", 0.0)),
        detected_landmark_rate=detected_rate,
        issues=issues,
        hoof_metrics=hoof_metrics,
    )


def render_markdown(reports: list[VideoReport]) -> str:
    lines = [
        "# Validation Study Report",
        "",
        "| View | Gait | Slug | Track | Horse frames | View OK | Detected | Reliable avg | Issues |",
        "|------|------|------|-------|--------------|---------|----------|--------------|--------|",
    ]

    for report in reports:
        issue_text = ", ".join(report.issues) if report.issues else "none"
        lines.append(
            f"| {report.view} | {report.gait} | {report.slug} | "
            f"{report.primary_track or '-'} | {report.frames_with_horses}/{report.frames} | "
            f"{'yes' if report.view_match else 'no'} | {report.detected_landmark_rate:.0%} | "
            f"{report.avg_reliable_landmarks:.1f} | {issue_text} |"
        )

    lines.extend(["", "## Hoof body-frame motion", ""])
    for report in reports:
        lines.append(f"### {report.view}/{report.gait}/{report.slug}")
        for hoof in ("left_front_hoof", "right_front_hoof", "left_hind_hoof", "right_hind_hoof"):
            metrics = report.hoof_metrics.get(hoof, {})
            if not metrics:
                continue
            lines.append(
                f"- **{hoof}**: frames={metrics.get('frames', 0)}, "
                f"spikes={metrics.get('spikes', 0)}, "
                f"qc_outliers={metrics.get('qc_outlier_rate', 0):.0%}, "
                f"body_mean_vel={metrics.get('body_mean_velocity')}, "
                f"body_p95_vel={metrics.get('body_p95_velocity')}"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize validation trajectories.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=ROOT / "output" / "validation",
        help="Validation output root",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    trajectory_paths = sorted(root.glob("*/*/*/trajectories.json"))
    if not trajectory_paths:
        raise SystemExit(f"No trajectories found under {root}")

    reports = [analyze_video(path) for path in trajectory_paths]
    payload = [report.__dict__ for report in reports]

    json_path = root / "validation_report.json"
    md_path = root / "validation_report.md"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown(reports), encoding="utf-8")

    print(render_markdown(reports))
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
