import cv2

from config.config import CANONICAL_SKELETON


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
