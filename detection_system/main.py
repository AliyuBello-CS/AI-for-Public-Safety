# main.py
import time
import cv2

from detector import DualYoloDetector
from frame_source import FrameSource
from visualizer import Visualizer
from app import DetectionApp
from notifier import Notifier
from visual_intelligence import VisualIntelligence  # <-- NEW
from fusion import FusionEngine

# --- CONFIG ---
MODE = "both"   # "either" or "both"
WINDOW_SEC = 1.0

ALERT_CLASSES = ["handgun", "knife", "heavy-weapon", "violence"]
ALERT_CONF = 0.35
COOLDOWN_SEC = 10.0
 
# --- Model paths ---
PERSON_MODEL_PATH = "/Users/hydarbello/Desktop/gunbest-2.pt"
WEAPON_MODEL_PATH = "/Users/hydarbello/Desktop/wea_best-2.pt"
VIDEO_SOURCE = "/Users/hydarbello/Desktop/TH_SYS/New_____/video_2_3.mp4"
def safe_crop(img, box, pad=14):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1 - pad)); y1 = max(0, int(y1 - pad))
    x2 = min(w, int(x2 + pad)); y2 = min(h, int(y2 + pad))
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def weapon_ok(det):
    label = (det.get("label") or "").lower()
    conf = float(det.get("confidence", 0.0))
    return (label in ALERT_CLASSES) and (conf >= ALERT_CONF)


def main():
    # 1) Cameras
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)

    if not cap0.isOpened():
        print("❌ Camera 0 failed to open")
        return
    if not cap1.isOpened():
        print("❌ Camera 1 failed to open")
        return

    # 2) Detector + notifier + VI
    detector = DualYoloDetector(
        person_model_path=PERSON_MODEL_PATH,
        weapon_model_path=WEAPON_MODEL_PATH,
        person_conf=0.25,
        weapon_conf=0.25,
    )
    notifier = Notifier()
    vi = VisualIntelligence(model="gpt-4o-mini")  # or your offline version
    fusion = FusionEngine(mode=MODE, window_sec=WINDOW_SEC)

    last_alert_time = 0.0

    print(f"✅ Multi-cam running. Fusion mode = {MODE}. Press 'q' to quit.")

    while True:
        r0, f0 = cap0.read()
        r1, f1 = cap1.read()

        if r0:
            det0, ann0 = detector.predict(f0)
            fusion.update_camera("cam0", det0, f0)
            cv2.imshow("CAM0", ann0)

        if r1:
            det1, ann1 = detector.predict(f1)
            fusion.update_camera("cam1", det1, f1)
            cv2.imshow("CAM1", ann1)

        # Alert gating
        now = time.time()
        if now - last_alert_time >= COOLDOWN_SEC:
            triggered, weapon_det, weapon_cam = fusion.should_alert(weapon_ok)
            if triggered:
                # choose best person view for clothing
                person_cam, person_det, person_frame = fusion.best_person_view()

                clothing_text = None
                if person_det is not None and person_frame is not None:
                    crop = safe_crop(person_frame, person_det["box"])
                    if crop is not None:
                        try:
                            clothing_text = vi.describe_clothing(crop)
                        except Exception as e:
                            clothing_text = f"Clothing description failed: {e}"
                    else:
                        clothing_text = "Clothing unclear (bad crop)."
                else:
                    clothing_text = "No person view available."

                wlabel = weapon_det.get("label")
                wconf = float(weapon_det.get("confidence", 0.0))

                extra_text = (
                    f"Fusion mode: {MODE}\n"
                    f"Weapon: {wlabel} ({wconf:.2f}) from {weapon_cam}\n"
                    f"Person view used: {person_cam}\n"
                    f"Clothing: {clothing_text}"
                )

                # attach the weapon camera annotated frame if available
                # simplest: attach whichever frame you last showed for that cam
                attach_frame = None
                if weapon_cam == "cam0" and r0:
                    attach_frame = ann0
                elif weapon_cam == "cam1" and r1:
                    attach_frame = ann1

                notifier.notify(wlabel, frame=attach_frame, extra_text=extra_text)
                last_alert_time = now

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()