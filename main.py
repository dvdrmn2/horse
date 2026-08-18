import cv2

from config.config import POSE_BACKEND
from utils.video import VideoReader
from utils.drawing import draw_boxes, draw_landmarks
from detection.detector import HorseDetector
from detection.pose_estimator import PoseEstimator

video = VideoReader("datasets/raw/test.mp4")

detector = HorseDetector()

try:
    pose_estimator = PoseEstimator(POSE_BACKEND)
    print(f"Pose backend: {POSE_BACKEND} ({pose_estimator.backend_name})")
except (ImportError, NotImplementedError, ValueError) as error:
    print(f"Pose backend '{POSE_BACKEND}' unavailable: {error}")
    print("Falling back to template backend (detection only, no skeleton).")
    pose_estimator = PoseEstimator("template")

while True:

    success, frame = video.read()

    if not success:
        break

    detections = detector.detect(frame)
    detections = pose_estimator.estimate(frame, detections)

    frame = draw_boxes(frame, detections)
    frame = draw_landmarks(frame, detections)

    cv2.imshow("Horse Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
