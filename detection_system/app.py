# app.py
import time
import cv2

from detector import DualYoloDetector
from frame_source import FrameSource
from visualizer import Visualizer  # overlays + display


class DetectionApp:
    """
    Two-model Detection App:
    - Model A: person
    - Model B: weapon
    - Merges detections into one list (with det["source"] = "person"/"weapon")
    - On weapon alert:
        - find closest person
        - crop person from RAW frame
        - ask visual_intelligence to describe clothing
        - include in email
    """

    def __init__(
        self,
        detector: DualYoloDetector,
        frame_source: FrameSource,
        visualizer: Visualizer,
        notifier=None,
        visual_intelligence=None,  # <-- NEW
        alert_classes=None,
        alert_cooldown_sec: float = 10.0,
        alert_conf_threshold: float = 0.6,
        one_alert_per_run: bool = False,
        debug: bool = True,
        require_person_for_alert: bool = True,  # <-- NEW: only alert if person+weapon
    ):
        self.detector = detector
        self.frame_source = frame_source
        self.visualizer = visualizer
        self.notifier = notifier
        self.visual_intelligence = visual_intelligence
        self.debug = debug
        self.require_person_for_alert = require_person_for_alert

        if alert_classes is None:
            alert_classes = ["Handgun", "Knife", "Heavy-weapon", "Violence"]

        self.alert_classes = [c.lower() for c in alert_classes]
        self.alert_cooldown_sec = alert_cooldown_sec
        self.alert_conf_threshold = alert_conf_threshold
        self.one_alert_per_run = one_alert_per_run

        self._last_alert_time = 0.0
        self._alert_sent_this_run = False

    def run(self):
        """Main loop: capture → detect → overlay → alert → display."""
        if not self.frame_source.open():
            print("❌ Could not open video source.")
            return

        print("✅ Press 'q' to quit.")
        prev_time = time.time()

        while True:
            ok, frame = self.frame_source.read()
            if not ok:
                print("End of video or cannot read frame.")
                break

            now = time.time()
            fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
            prev_time = now

            # detector.predict must return (detections, annotated_frame)
            detections, annotated_frame = self.detector.predict(frame)

            if self.debug and detections:
                preview = []
                for d in detections[:12]:
                    preview.append((d.get("source"), d.get("label"), round(d.get("confidence", 0.0), 2)))
                print(preview)

            # Use YOLO's annotated frame + add overlay text
            frame_out = annotated_frame.copy()
            frame_out = self.visualizer.draw_detections(frame_out, detections, fps=fps)

            # IMPORTANT: pass RAW frame for cropping clothing
            self._maybe_send_alert(detections, raw_frame=frame, frame_with_boxes=frame_out)

            self.visualizer.show(frame_out)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.frame_source.release()
        self.visualizer.close()

    @staticmethod
    def _center(box):
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    @staticmethod
    def _safe_crop(img, box, pad=12):
        """Crop box from img with padding, clipped to frame bounds."""
        h, w = img.shape[:2]
        x1, y1, x2, y2 = box

        x1 = max(0, int(x1 - pad))
        y1 = max(0, int(y1 - pad))
        x2 = min(w, int(x2 + pad))
        y2 = min(h, int(y2 + pad))

        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2]

    def _pick_main_weapon_threat(self, detections):
        """Pick highest-confidence weapon detection that matches alert classes + threshold."""
        threats = []
        for det in detections:
            label = (det.get("label") or "").lower()
            conf = float(det.get("confidence", 0.0))

            # prefer weapon-model detections only (if available)
            src = det.get("source")
            if src is not None and src != "weapon":
                continue

            if label in self.alert_classes and conf >= self.alert_conf_threshold:
                threats.append(det)

        if not threats:
            return None
        return max(threats, key=lambda d: d.get("confidence", 0.0))

    def _find_closest_person(self, detections, weapon_det):
        """Find the person detection closest to the weapon (by center distance)."""
        persons = [d for d in detections if d.get("source") == "person"]
        if not persons:
            return None

        wcx, wcy = self._center(weapon_det["box"])

        def dist2(p):
            pcx, pcy = self._center(p["box"])
            return (pcx - wcx) ** 2 + (pcy - wcy) ** 2

        return min(persons, key=dist2)

    def _maybe_send_alert(self, detections, raw_frame, frame_with_boxes):
        """
        Send an email alert if:
        - weapon label is in alert_classes
        - confidence >= threshold
        - cooldown passed
        - optionally only one alert per run
        - optionally require a person detection too
        """
        if self.notifier is None:
            return

        if self.one_alert_per_run and self._alert_sent_this_run:
            return

        now = time.time()
        if now - self._last_alert_time < self.alert_cooldown_sec:
            return

        weapon_det = self._pick_main_weapon_threat(detections)
        if weapon_det is None:
            return

        # If you want person+weapon required
        person_det = self._find_closest_person(detections, weapon_det)
        if self.require_person_for_alert and person_det is None:
            return

        main_label = weapon_det["label"]
        main_conf = float(weapon_det.get("confidence", 0.0))

        clothing_text = None
        if self.visual_intelligence is not None and person_det is not None:
            crop = self._safe_crop(raw_frame, person_det["box"], pad=14)
            if crop is not None:
                try:
                    clothing_text = self.visual_intelligence.describe_clothing(crop)
                except Exception as e:
                    clothing_text = f"Clothing description failed: {e}"
            else:
                clothing_text = "Clothing description unavailable (invalid crop)."

        # Build extra message for the email
        extra_lines = [f"Weapon: {main_label} ({main_conf:.2f})"]
        if person_det is not None:
            extra_lines.append(f"Person detected (closest to weapon).")
        if clothing_text:
            extra_lines.append(f"Clothing: {clothing_text}")

        extra_text = "\n".join(extra_lines)

        print(f"⚠️ Alert triggered: {main_label} ({main_conf:.2f})")
        print(extra_text)

        # Call notifier in a backward-compatible way
        try:
            # If your notifier.notify supports extra_text
            self.notifier.notify(main_label, frame=frame_with_boxes, extra_text=extra_text)
        except TypeError:
            # Older notifier signature: notify(label, frame)
            self.notifier.notify(main_label, frame=frame_with_boxes)

        self._last_alert_time = now
        self._alert_sent_this_run = True