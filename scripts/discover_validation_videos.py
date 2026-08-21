#!/usr/bin/env python3
"""Discover validation videos organized as {view}/{gait}/*.{mp4,mov,avi}."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
KNOWN_VIEWS = {"side", "rear", "front", "oblique"}
KNOWN_GAITS = {"walk", "trot", "run", "canter", "gallop"}


@dataclass(frozen=True)
class ValidationVideo:
    path: Path
    view: str
    gait: str
    slug: str

    @property
    def relative_key(self) -> str:
        return f"{self.view}/{self.gait}/{self.slug}"


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-._")
    return slug[:80] or "video"


def discover_videos(root: Path) -> list[ValidationVideo]:
    videos: list[ValidationVideo] = []

    if not root.is_dir():
        return videos

    for view_dir in sorted(root.iterdir()):
        if not view_dir.is_dir() or view_dir.name.startswith("."):
            continue
        view = view_dir.name.lower()
        if view not in KNOWN_VIEWS:
            continue

        for gait_dir in sorted(view_dir.iterdir()):
            if not gait_dir.is_dir() or gait_dir.name.startswith("."):
                continue
            gait = gait_dir.name.lower()
            if gait not in KNOWN_GAITS:
                continue

            for path in sorted(gait_dir.iterdir()):
                if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                videos.append(
                    ValidationVideo(
                        path=path.resolve(),
                        view=view,
                        gait=gait,
                        slug=slugify(path.stem),
                    )
                )

    return videos


def main() -> None:
    parser = argparse.ArgumentParser(description="List validation videos under a root directory.")
    parser.add_argument(
        "root",
        type=Path,
        nargs="?",
        default=Path("/Users/dvd_rmn/Desktop/videos"),
        help="Root directory containing view/gait folders",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON manifest")
    args = parser.parse_args()

    videos = discover_videos(args.root.expanduser().resolve())
    if args.json:
        print(json.dumps([asdict(video) | {"path": str(video.path)} for video in videos], indent=2))
        return

    print(f"Found {len(videos)} validation videos under {args.root}:")
    for video in videos:
        print(f"  {video.relative_key:40s}  {video.path.name}")


if __name__ == "__main__":
    main()
