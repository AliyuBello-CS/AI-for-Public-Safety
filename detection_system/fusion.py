# fusion.py
import time

def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)

class FusionEngine:
    """
    mode:
      - "either": alert if any camera has a qualifying weapon detection
      - "both": alert only if 2 cameras confirm within window_sec
    """
    def __init__(self, mode="either", window_sec=1.0):
        assert mode in ("either", "both")
        self.mode = mode
        self.window_sec = window_sec
        self.last_weapon = {}   # cam_id -> (t, weapon_det)
        self.last_person = {}   # cam_id -> (t, person_det, raw_frame)

    def update_camera(self, cam_id, detections, raw_frame):
        now = time.time()

        # Pick best weapon (already filtered by your app's alert_classes/threshold later)
        weapons = [d for d in detections if d.get("source") == "weapon"]
        if weapons:
            best_w = max(weapons, key=lambda d: d.get("confidence", 0.0))
            self.last_weapon[cam_id] = (now, best_w)

        # Pick best person (largest box)
        persons = [d for d in detections if d.get("source") == "person"]
        if persons:
            best_p = max(persons, key=lambda d: box_area(d["box"]))
            self.last_person[cam_id] = (now, best_p, raw_frame)

    def _weapons_in_window(self):
        now = time.time()
        active = {}
        for cam_id, (t, wdet) in self.last_weapon.items():
            if now - t <= self.window_sec:
                active[cam_id] = wdet
        return active

    def should_alert(self, weapon_ok_fn):
        """
        weapon_ok_fn(det) -> bool: checks label/conf threshold
        Returns:
          (triggered: bool, weapon_det: dict or None, weapon_cam: str or None)
        """
        active = self._weapons_in_window()
        # filter weapons by your criteria
        active_ok = {cid: d for cid, d in active.items() if weapon_ok_fn(d)}
        if not active_ok:
            return False, None, None

        if self.mode == "either":
            # choose best confidence among cameras
            weapon_cam, weapon_det = max(active_ok.items(), key=lambda kv: kv[1].get("confidence", 0.0))
            return True, weapon_det, weapon_cam

        # mode == "both"
        if len(active_ok) >= 2:
            # choose best confidence as primary weapon det
            weapon_cam, weapon_det = max(active_ok.items(), key=lambda kv: kv[1].get("confidence", 0.0))
            return True, weapon_det, weapon_cam

        return False, None, None

    def best_person_view(self):
        """
        Returns (cam_id, person_det, raw_frame) for best available person crop within window.
        Chooses largest person box among cams.
        """
        now = time.time()
        candidates = []
        for cam_id, (t, pdet, frame) in self.last_person.items():
            if now - t <= self.window_sec:
                candidates.append((cam_id, pdet, frame))
        if not candidates:
            return None, None, None
        return max(candidates, key=lambda x: box_area(x[1]["box"]))