import cv2

class VideoReader:

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)

        if not self.cap.isOpened():
            raise Exception(f"Could not open {path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        reported_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.estimated_frame_count = reported_count if reported_count > 0 else None

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()
