from collections import deque, Counter


# BGR colors for each shot type (for overlay banner)
SHOT_COLORS = {
    "Stance (Ready)":    (80, 80, 80),
    "Forward Defensive": (180, 130, 0),
    "Drive":             (0, 180, 0),
    "Lofted Drive":      (0, 220, 80),
    "Pull / Hook Shot":  (0, 100, 220),
    "Sweep Shot":        (180, 0, 200),
    "Cut Shot":          (0, 0, 200),
    "Back Foot Punch":   (0, 180, 180),
}


class ShotClassifier:
    """
    Rule-based shot classifier using MediaPipe pose angles and positions.

    Angles expected (degrees):
        front_knee   – hip → knee → ankle on the leading leg
        back_knee    – hip → knee → ankle on the trailing leg (side view only)
        front_elbow  – wrist → elbow → shoulder on the top-hand arm

    Positions expected (image pixel y-coords, larger = lower in frame):
        front_wrist_y, front_shoulder_y, front_hip_y
    """

    def __init__(self, window_size=20):
        self.history = deque(maxlen=window_size)
        self.current_shot = "Stance (Ready)"

    def classify(self, angles: dict, positions: dict) -> str:
        raw = self._classify_raw(angles, positions)
        self.history.append(raw)
        if self.history:
            self.current_shot = Counter(self.history).most_common(1)[0][0]
        return self.current_shot

    def _classify_raw(self, angles: dict, positions: dict) -> str:
        fk  = angles.get('front_knee',  160)
        bk  = angles.get('back_knee',   160)
        fe  = angles.get('front_elbow', 120)

        wy  = positions.get('front_wrist_y',    999)
        sy  = positions.get('front_shoulder_y', 500)
        hy  = positions.get('front_hip_y',      600)

        # image coords: smaller y = higher in the frame
        hand_above_shoulder = wy < sy
        hand_above_hip      = wy < hy

        # --- Shot rules (order matters) ---

        # Sweep: extreme front-knee bend
        if fk < 125:
            return "Sweep Shot"

        # Pull / Hook: hands above shoulder, back foot planted
        if hand_above_shoulder and bk > 148:
            return "Pull / Hook Shot"

        # Cut: weight on back foot, hands high
        if bk > 155 and hand_above_shoulder:
            return "Cut Shot"

        # Drive family: front leg extends, elbow follows through
        if fk > 158 and fe > 125:
            return "Lofted Drive" if hand_above_shoulder else "Drive"

        # Forward defensive: front foot forward, hands low
        if fk > 148 and not hand_above_shoulder and fe < 120:
            return "Forward Defensive"

        # Back foot punch: weight fully transferred back
        if bk > 158 and fk > 152:
            return "Back Foot Punch"

        return "Stance (Ready)"

    def get_color(self) -> tuple:
        return SHOT_COLORS.get(self.current_shot, (80, 80, 80))
