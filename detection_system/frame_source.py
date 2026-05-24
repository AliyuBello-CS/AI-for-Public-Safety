# frame_source.py
import cv2


class FrameSource:
    """
    Handles:
    - Opening a video source (webcam or file)
    - Providing frames
    """

    def __init__(self, source=0):
        """
        source: 0 for webcam, or path to video file.
        """
        self.source = source
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.source)
        return self.cap.isOpened()

    def read(self):
        """
        Returns (success, frame).
        """
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None