import cv2

from config.config import CANONICAL_SKELETON
from models.horse import HorseInstance
from models.landmark_types import LandmarkStatus
from models.session import FrameRecord


def draw_horse_box(frame, horse: HorseInstance):
    x1, y1, x2, y2 = map(int, horse.bbox)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"horse #{horse.track_id} {horse.detection_confidence:.2f}",
        (x1, max(y1 - 8, 0)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )

    return frame


def draw_horse_landmarks(frame, horse: HorseInstance):
    if horse.landmarks is None:
        return frame

    landmark_map = {
        landmark.name: landmark for landmark in horse.landmarks.landmarks if landmark.is_drawable
    }

    for start_name, end_name in CANONICAL_SKELETON:
        start = landmark_map.get(start_name)
        end = landmark_map.get(end_name)
        if start is None or end is None:
            continue
        if not start.is_reliable or not end.is_reliable:
            continue

        cv2.line(
            frame,
            (int(start.x), int(start.y)),
            (int(end.x), int(end.y)),
            (255, 128, 0),
            2,
        )

    for landmark in horse.landmarks.landmarks:
        if not landmark.is_drawable:
            continue

        if landmark.status is LandmarkStatus.DERIVED:
            color = (200, 64, 255)
        elif landmark.is_reliable:
            color = (0, 128, 255)
        else:
            color = (160, 160, 160)
        cv2.circle(frame, (int(landmark.x), int(landmark.y)), 4, color, -1)

    return frame


def draw_frame_overlay(frame, frame_record: FrameRecord):
    for horse in frame_record.horses:
        frame = draw_horse_box(frame, horse)
        frame = draw_horse_landmarks(frame, horse)

    cv2.putText(
        frame,
        f"view={frame_record.view} reliable={frame_record.reliable_landmark_count}",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )
    return frame


# Backward-compatible helpers for legacy Detection-based code paths.
def draw_boxes(frame, detections):
    for det in detections:
        x1, y1, x2, y2 = map(int, det.bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"horse {det.confidence:.2f}",
            (x1, max(y1 - 8, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return frame


def draw_landmarks(frame, detections):
    for det in detections:
        if det.landmarks is None:
            continue

        landmark_map = {
            landmark.name: landmark for landmark in det.landmarks.visible_landmarks
        }

        for start_name, end_name in CANONICAL_SKELETON:
            start = landmark_map.get(start_name)
            end = landmark_map.get(end_name)
            if start is None or end is None:
                continue

            cv2.line(
                frame,
                (int(start.x), int(start.y)),
                (int(end.x), int(end.y)),
                (255, 128, 0),
                2,
            )

        for landmark in det.landmarks.visible_landmarks:
            cv2.circle(frame, (int(landmark.x), int(landmark.y)), 4, (0, 128, 255), -1)

    return frame
