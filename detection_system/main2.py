from detector import DualYoloDetector
from frame_source import FrameSource
from visualizer import Visualizer
from app import DetectionApp
from notifier import Notifier
from visual_intelligence import VisualIntelligence  # <-- NEW
import cv2

# --- Model paths ---
PERSON_MODEL_PATH = "/Users/hydarbello/Desktop/gunbest-2.pt"
WEAPON_MODEL_PATH = "/Users/hydarbello/Desktop/wea_best-2.pt"

# --- Video source ---
VIDEO_SOURCE = "/Users/hydarbello/Desktop/TH_SYS/New_____/video_2_3.mp4"  # or 0 for webcam



def main():
    # 1) Dual model detector
    detector = DualYoloDetector(
        person_model_path=PERSON_MODEL_PATH,
        weapon_model_path=WEAPON_MODEL_PATH,
        person_conf=0.25,
        weapon_conf=0.25,
    )

    # 2) Video + UI + notifier
    frame_source = FrameSource(VIDEO_SOURCE)
    visualizer = Visualizer("YOLO Dual-Model System")
    notifier = Notifier()

    # 3) Visual Intelligence (clothing description)
    # Requires OPENAI_API_KEY in your environment if you're using the OpenAI-based module
    vi = VisualIntelligence(model="gpt-4o-mini")

    # 4) App
    app = DetectionApp(
        detector=detector,
        frame_source=frame_source,
        visualizer=visualizer,
        notifier=notifier,
        visual_intelligence=vi,          # <-- NEW
        require_person_for_alert=True,   # <-- only alert when weapon + person exist
        alert_classes=["Handgun", "Knife", "Heavy-weapon", "Violence"],
        alert_conf_threshold=0.6,
        alert_cooldown_sec=10.0,
        one_alert_per_run=False,
        debug=True,
    )

    app.run()


if __name__ == "__main__":
    main()