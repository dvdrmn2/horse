#!/usr/bin/env python3
"""Compare pose backends on the same video (AnimalPose vs SuperAnimal).

SuperAnimal requires optional deps — see requirements-optional-superanimal.txt.
This script documents the benchmark workflow for Phase 1C.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark pose backends on one video.")
    parser.add_argument("--video", default=os.getenv("INPUT_VIDEO", "datasets/raw/test.mp4"))
    parser.add_argument(
        "--backends",
        nargs="+",
        default=["horse10_mmpose"],
        help="Backends to compare: horse10_mmpose superanimal template",
    )
    args = parser.parse_args()

    if not Path(args.video).is_file():
        raise FileNotFoundError(args.video)

    print("Phase 1C benchmark scaffold")
    print(f"Video: {args.video}")
    print(f"Backends requested: {args.backends}")
    print()
    print("Run each backend with Docker and compare output trajectories:")
    for backend in args.backends:
        output = Path("output") / f"annotated_{backend}.mp4"
        traj = output.with_suffix(".trajectories.json")
        print(
            f"  POSE_BACKEND={backend} OUTPUT_VIDEO={output} "
            f"OUTPUT_TRAJECTORIES={traj} docker compose run --rm horse"
        )
    print()
    print("SuperAnimal is not wired yet — backend raises NotImplementedError until installed.")


if __name__ == "__main__":
    main()
