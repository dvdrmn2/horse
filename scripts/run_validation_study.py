#!/usr/bin/env python3
"""Run the pose pipeline over a validation video set."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from discover_validation_videos import ValidationVideo, discover_videos


def run_one_local(video: ValidationVideo, output_root: Path, *, write_video: bool) -> Path:
    out_dir = output_root / video.view / video.gait / video.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    output_video = out_dir / "annotated.mp4" if write_video else ""
    trajectory_path = out_dir / "trajectories.json"

    env = os.environ.copy()
    env["INPUT_VIDEO"] = str(video.path)
    env["HEADLESS"] = "1"
    env["OUTPUT_TRAJECTORIES"] = str(trajectory_path)
    if output_video:
        env["OUTPUT_VIDEO"] = str(output_video)
    else:
        env.pop("OUTPUT_VIDEO", None)

    subprocess.run(
        [sys.executable, str(ROOT / "main.py")],
        cwd=ROOT,
        env=env,
        check=True,
    )
    return trajectory_path


def run_one_docker(
    video: ValidationVideo,
    output_root: Path,
    *,
    write_video: bool,
    videos_mount: Path,
) -> Path:
    out_dir = output_root / video.view / video.gait / video.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    container_video = Path("/videos") / video.path.relative_to(videos_mount)
    container_output = Path("/app/output/validation") / video.view / video.gait / video.slug
    trajectory_path = out_dir / "trajectories.json"

    cmd = [
        "docker",
        "compose",
        "run",
        "--rm",
        "-v",
        f"{videos_mount.resolve()}:/videos:ro",
        "-e",
        f"INPUT_VIDEO={container_video}",
        "-e",
        f"OUTPUT_TRAJECTORIES={container_output / 'trajectories.json'}",
        "-e",
        "HEADLESS=1",
    ]

    if write_video:
        cmd.extend(["-e", f"OUTPUT_VIDEO={container_output / 'annotated.mp4'}"])
    else:
        cmd.extend(["-e", "OUTPUT_VIDEO="])

    cmd.append("horse")

    subprocess.run(cmd, cwd=ROOT, check=True)
    return trajectory_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation study over view/gait video folders.")
    parser.add_argument(
        "--videos-dir",
        type=Path,
        default=Path("/Users/dvd_rmn/Desktop/videos"),
        help="Root containing side/walk, rear/trot, etc.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "validation",
        help="Where to write trajectories and annotated videos",
    )
    parser.add_argument("--docker", action="store_true", help="Run each video through docker compose")
    parser.add_argument("--skip-video", action="store_true", help="Skip annotated video output (faster)")
    parser.add_argument("--view", action="append", help="Only process this view (repeatable)")
    parser.add_argument("--gait", action="append", help="Only process this gait (repeatable)")
    parser.add_argument("--limit", type=int, default=0, help="Process at most N videos")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have trajectories.json (do not use after pipeline changes)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Delete each video output directory before processing (use to rebuild stale results)",
    )
    args = parser.parse_args()

    if args.force and args.skip_existing:
        raise SystemExit("Use either --force or --skip-existing, not both.")

    videos_dir = args.videos_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(videos_dir)
    if args.view:
        allowed = {value.lower() for value in args.view}
        videos = [video for video in videos if video.view in allowed]
    if args.gait:
        allowed = {value.lower() for value in args.gait}
        videos = [video for video in videos if video.gait in allowed]
    if args.limit:
        videos = videos[: args.limit]

    if not videos:
        raise SystemExit(f"No validation videos found under {videos_dir}")

    manifest = []
    print(f"Processing {len(videos)} videos → {output_dir}")

    for index, video in enumerate(videos, start=1):
        out_dir = output_dir / video.view / video.gait / video.slug
        trajectory_path = out_dir / "trajectories.json"
        if args.force and out_dir.exists():
            shutil.rmtree(out_dir)
        if args.skip_existing and trajectory_path.is_file():
            print(f"[{index}/{len(videos)}] skip existing {video.relative_key}")
            manifest.append(
                {
                    **asdict(video),
                    "path": str(video.path),
                    "trajectories": str(trajectory_path),
                    "skipped": True,
                }
            )
            continue

        print(f"[{index}/{len(videos)}] {video.relative_key}  ({video.path.name})")
        if args.docker:
            trajectory_path = run_one_docker(
                video,
                output_dir,
                write_video=not args.skip_video,
                videos_mount=videos_dir,
            )
        else:
            trajectory_path = run_one_local(
                video,
                output_dir,
                write_video=not args.skip_video,
            )

        manifest.append(
            {
                **asdict(video),
                "path": str(video.path),
                "trajectories": str(trajectory_path),
            }
        )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {manifest_path}")
    print("Generate report with: python3 scripts/validation_report.py output/validation")


if __name__ == "__main__":
    main()
