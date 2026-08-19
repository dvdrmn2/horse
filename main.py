import os
from pathlib import Path

import cv2

from config.config import POSE_BACKEND
from utils.video import VideoReader
from utils.drawing import draw_boxes, draw_landmarks
from detection.detector import HorseDetector
from detection.pose_estimator import PoseEstimator

INPUT_VIDEO = os.getenv("INPUT_VIDEO", "datasets/raw/test.mp4")
OUTPUT_VIDEO = os.getenv("OUTPUT_VIDEO", "")
HEADLESS = os.environ.get("HEADLESS", "").lower() in ("1", "true", "yes")


def create_pose_estimator() -> PoseEstimator:
    try:
        estimator = PoseEstimator(POSE_BACKEND)
        print(f"Pose backend: {POSE_BACKEND} ({estimator.backend_name})")
        if POSE_BACKEND == "template":
            print(
                "Skeleton disabled (template backend). "
                "Set POSE_BACKEND=horse10_mmpose for pose landmarks."
            )
        elif POSE_BACKEND == "horse10_mmpose":
            # Fail fast: load MMPose weights before processing the video.
            estimator.backend._load_model()
            print("MMPose animal pose model loaded.")
        return estimator
    except (ImportError, NotImplementedError, ValueError, FileNotFoundError) as error:
        print(f"Pose backend '{POSE_BACKEND}' unavailable: {error}")
        print("Falling back to template backend (detection only, no skeleton).")
        return PoseEstimator("template")


def main() -> None:
    if not Path(INPUT_VIDEO).is_file():
        raise FileNotFoundError(f"Input video not found: {INPUT_VIDEO}")

    if not Path("yolov8x.pt").is_file():
        raise FileNotFoundError(
            "Missing yolov8x.pt in project root. "
            "Download the YOLOv8x weights and place them beside main.py."
        )

    video = VideoReader(INPUT_VIDEO)
    detector = HorseDetector()
    pose_estimator = create_pose_estimator()

    writer = None
    if OUTPUT_VIDEO:
        Path(OUTPUT_VIDEO).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = video.fps if video.fps and video.fps > 0 else 25.0
        writer = cv2.VideoWriter(
            OUTPUT_VIDEO,
            fourcc,
            fps,
            (video.width, video.height),
        )
        print(f"Writing output to: {OUTPUT_VIDEO}")

    frame_count = 0

    while True:
        success, frame = video.read()

        if not success:
            break

        detections = detector.detect(frame)
        detections = pose_estimator.estimate(frame, detections)

        frame = draw_boxes(frame, detections)
        frame = draw_landmarks(frame, detections)

        if writer is not None:
            writer.write(frame)

        if not HEADLESS:
            cv2.imshow("Horse Detection", frame)
            if cv2.waitKey(1) == ord("q"):
                break

        frame_count += 1

    video.release()

    if writer is not None:
        writer.release()
        print(f"Saved {frame_count} frames to {OUTPUT_VIDEO}")

    if not HEADLESS:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
