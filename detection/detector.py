from ultralytics import YOLO
from models.detection import Detection

class HorseDetector:

    def __init__(self):
        self.model = YOLO("yolov8x.pt")

    def detect(self, frame):

        results = self.model(frame)

        detections = []

        for box in results[0].boxes:

            cls = int(box.cls[0])

            if cls != 17:
                continue

            confidence = float(box.conf[0])

            if confidence < 0.5:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                Detection(
                    bbox=[x1, y1, x2, y2],
                    confidence=confidence,
                    class_id=cls
                )
            )

        return detections