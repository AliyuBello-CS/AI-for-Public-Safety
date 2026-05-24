   # detector.py (ADD THIS BELOW your current Yolo11Detector)
import cv2
import torch
from ultralytics import YOLO

class DualYoloDetector:
    """
    Runs two YOLO models per frame (e.g., person + weapon),
    merges detections, and returns one annotated frame.
    """

    def __init__(
        self,
        person_model_path: str,
        weapon_model_path: str,
        person_conf: float = 0.25,
        weapon_conf: float = 0.25,
    ):
        self.person_conf = person_conf
        self.weapon_conf = weapon_conf

        self.person_model = self._load_model(person_model_path, tag="PERSON")
        self.weapon_model = self._load_model(weapon_model_path, tag="WEAPON")

    def _load_model(self, path: str, tag: str):
        model = YOLO(path)
        try:
            if torch.backends.mps.is_available():
                print(f"🔋 Using MPS (Apple GPU) for {tag} model")
                model.to("mps")
            else:
                print(f"⚙️ MPS not available, using CPU for {tag} model")
        except Exception as e:
            print(f"⚠️ Could not move {tag} model to MPS: {e}")
        return model

    def _extract(self, results, names, source: str):
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        dets = []
        for box, conf, cls in zip(boxes, confs, classes):
            x1, y1, x2, y2 = box.astype(int)
            label = names.get(cls, str(cls))
            dets.append({
                "label": label,
                "confidence": float(conf),
                "box": (x1, y1, x2, y2),
                "source": source,   # "person" or "weapon"
            })
        return dets

    def _draw(self, frame, detections):
        out = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            label = det["label"]
            conf = det["confidence"]
            source = det.get("source")

            # person = green, weapon = red
            if source == "person":
                color = (0, 255, 0)
                prefix = "P"
            else:
                color = (0, 0, 255)
                prefix = "W"

            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                out,
                f"{prefix}:{label} {conf:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
        return out

    def predict(self, frame):
        person_res = self.person_model(frame, conf=self.person_conf, verbose=False)[0]
        weapon_res = self.weapon_model(frame, conf=self.weapon_conf, verbose=False)[0]

        det_person = self._extract(person_res, self.person_model.names, "person")
        det_weapon = self._extract(weapon_res, self.weapon_model.names, "weapon")

        detections = det_person + det_weapon
        annotated_frame = self._draw(frame, detections)

        return detections, annotated_frame