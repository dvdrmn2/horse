#!/usr/bin/env python3
"""Audit view features for missing-vs-zero conflation across trajectories."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from landmarks.view import _visible_point, build_view_diagnostics, extract_view_features
from scripts.diagnose_rear_view import horse_from_frame, _primary_track, _parse_video_path

AUDIT_FEATURES = (
    "stifle_separation",
    "body_horizontal_span",
    "body_vertical_span",
    "eye_separation",
    "bilateral_symmetry",
    "tail_visible",
    "withers_visible",
)


def audit_trajectory(path: Path, *, track_id: int | None = None) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    primary_track = track_id if track_id is not None else _primary_track(payload.get("summary", {}))

    availability_counts: dict[str, Counter] = {key: Counter() for key in AUDIT_FEATURES}
    stifle_states = Counter()
    body_span_states = Counter()
    misleading_zero_frames: list[dict] = []
    frames_analyzed = 0

    for frame in payload.get("frames", []):
        horse = horse_from_frame(frame, primary_track)
        if horse is None:
            continue

        frames_analyzed += 1
        features = extract_view_features(horse)
        diagnostics = build_view_diagnostics(horse, frame_index=frame.get("frame_index"))
        availability = features.availability_summary()

        for key in AUDIT_FEATURES:
            availability_counts[key]["available" if availability.get(key, False) else "unavailable"] += 1

        left_stifle = features.stifle_sep.available
        if left_stifle:
            stifle_states["both_stifles"] += 1
        else:
            has_left = _visible_point(horse, "left_stifle") is not None
            has_right = _visible_point(horse, "right_stifle") is not None
            if has_left or has_right:
                stifle_states["single_stifle_only"] += 1
            else:
                stifle_states["no_stifles"] += 1

        if features.body_horizontal_span.available and features.body_vertical_span.available:
            body_span_states["both_spans"] += 1
        elif availability.get("body_horizontal_span") or availability.get("body_vertical_span"):
            body_span_states["partial_axis"] += 1
        else:
            body_span_states["no_axis"] += 1

        if not features.stifle_sep.available:
            has_left = _visible_point(horse, "left_stifle") is not None
            has_right = _visible_point(horse, "right_stifle") is not None
            if has_left ^ has_right:
                misleading_zero_frames.append(
                    {
                        "frame": frame.get("frame_index"),
                        "issue": "stifle_separation_would_have_been_zero_single_stifle",
                    }
                )
        if not features.body_horizontal_span.available and len(
            [point for point in (
                _visible_point(horse, "nose"),
                _visible_point(horse, "neck"),
                _visible_point(horse, "withers"),
                _visible_point(horse, "tail_head"),
            ) if point]
        ) < 2:
            misleading_zero_frames.append(
                {
                    "frame": frame.get("frame_index"),
                    "issue": "body_span_would_have_been_zero_insufficient_axis_landmarks",
                }
            )

    view, gait, slug = _parse_video_path(path)
    return {
        "video": f"{view}/{gait}/{slug}",
        "path": str(path),
        "frames_analyzed": frames_analyzed,
        "availability_rates": {
            key: {
                "available": counts["available"],
                "unavailable": counts["unavailable"],
                "available_pct": round(100.0 * counts["available"] / frames_analyzed, 1) if frames_analyzed else 0.0,
            }
            for key, counts in availability_counts.items()
        },
        "stifle_states": dict(stifle_states),
        "body_span_states": dict(body_span_states),
        "misleading_zero_frame_count": len(misleading_zero_frames),
        "misleading_zero_examples": misleading_zero_frames[:8],
    }


def render_markdown(reports: list[dict]) -> str:
    lines = [
        "# View Feature Availability Audit",
        "",
        "Checks whether key view features are explicitly unavailable instead of silently using 0.",
        "",
        "| Video | Frames | Stifle avail % | Body span avail % | Symmetry avail % | Single stifle | No axis | Misleading-zero frames |",
        "|-------|--------|----------------|-------------------|------------------|---------------|---------|------------------------|",
    ]

    for report in reports:
        rates = report["availability_rates"]
        stifle = report["stifle_states"]
        body = report["body_span_states"]
        lines.append(
            f"| {report['video']} | {report['frames_analyzed']} | "
            f"{rates['stifle_separation']['available_pct']:.0f}% | "
            f"{rates['body_horizontal_span']['available_pct']:.0f}% | "
            f"{rates['bilateral_symmetry']['available_pct']:.0f}% | "
            f"{stifle.get('single_stifle_only', 0)} | "
            f"{body.get('no_axis', 0)} | "
            f"{report['misleading_zero_frame_count']} |"
        )

    lines.extend(["", "## Detail", ""])
    for report in reports:
        lines.append(f"### {report['video']}")
        lines.append(f"- Frames analyzed: {report['frames_analyzed']}")
        lines.append(f"- Stifle states: {report['stifle_states']}")
        lines.append(f"- Body span states: {report['body_span_states']}")
        for key, payload in report["availability_rates"].items():
            lines.append(
                f"- **{key}**: {payload['available']} available / {payload['unavailable']} unavailable "
                f"({payload['available_pct']:.1f}% available)"
            )
        if report["misleading_zero_examples"]:
            lines.append("- Misleading-zero examples:")
            for example in report["misleading_zero_examples"]:
                lines.append(f"  - frame {example['frame']}: {example['issue']}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit view feature availability vs zero conflation.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=ROOT / "output" / "validation",
    )
    parser.add_argument("--view", default=None, help="Only audit videos under this folder view")
    parser.add_argument("--track", type=int, default=None)
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if args.view:
        paths = sorted(root.glob(f"{args.view}/*/*/trajectories.json"))
    else:
        paths = sorted(root.glob("*/*/*/trajectories.json"))

    if not paths:
        raise SystemExit(f"No trajectories found under {root}")

    reports = [audit_trajectory(path, track_id=args.track) for path in paths]
    markdown = render_markdown(reports)

    output_root = paths[0].parents[3] if "validation" in paths[0].parts else paths[0].parent
    json_path = output_root / "view_feature_audit.json"
    md_path = output_root / "view_feature_audit.md"
    json_path.write_text(json.dumps(reports, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")

    print(markdown)
    print(f"\nWrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
