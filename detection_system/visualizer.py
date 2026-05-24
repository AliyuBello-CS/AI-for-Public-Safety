# visualizer.py
import cv2


class Visualizer:
    """
    Responsible for:
    - Drawing boxes and labels on frames
    - Coloring by confidence
    - Showing frames in a window
    """

    def __init__(self, window_name: str = "YOLO11 Demo"):
        self.window_name = window_name

    @staticmethod
    def description(score: float) -> str:
        """
        score is confidence * 100.
        """
        if score < 25:
            return "SAFE"
        elif score < 50:
            return "MEDIUM"
        elif score < 85:
            return "HIGH"
        else:
            return "CRITICAL"

    @staticmethod
    def color_for_score(score: float):
        """
        Map score (0–100) to BGR color:
        - < 50  -> yellow
        - < 85  -> green
        - >= 85 -> red
        """
        if score < 50:
            return (0, 255, 255)  # yellow
        elif score < 85:
            return (0, 255, 0)    # green
        else:
            return (0, 0, 255)    # red

    def draw_detections(self, frame, detections, fps: float | None = None):
        """
        Draw bounding boxes, labels, confidence, description, and FPS.
        """
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            conf = det["confidence"]

            score = conf * 100.0
            desc = self.description(score)
            color = self.color_for_score(score)

            # Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Text: label + conf + description
            text = f"{label} {conf:.2f} {desc}"
            cv2.putText(
                frame,
                text,
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )

        if fps is not None:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return frame

    def show(self, frame):
        cv2.imshow(self.window_name, frame)

    def close(self):
        cv2.destroyAllWindows()