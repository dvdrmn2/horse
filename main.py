from utils.video import VideoReader
from detection.detector import HorseDetector

video = VideoReader("datasets/raw/test.mp4")

detector = HorseDetector()

while True:

    success, frame = video.read()

    if not success:
        break

    detections = detector.detect(frame)

    frame = draw_boxes(frame, detections)

    cv2.imshow("Horse Detection", frame)

    if cv2.waitKey(1) == ord("q"):
        break

video.release()
cv2.destroyAllWindows()