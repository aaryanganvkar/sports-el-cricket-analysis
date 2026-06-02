"""
headless_analyzer.py

Runs the full CricketLytics analysis pipeline on a video file without
opening any display window.

Uses mp.solutions.pose — the same API as the working backup — which is
still fully available in mediapipe >= 0.10.x. Annotated frames
(skeleton + angle labels + shot overlay) are captured at even intervals
and returned as base64 JPEG strings so the dashboard can show a
flipbook without a second HTTP request.
"""

from __future__ import annotations

import base64
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from src.Analysis.shot_classifier import ShotClassifier
from src.Analysis.posture_advisor import PostureAdvisor
from src.Analysis.overlay import draw_analysis_overlay


# ── MediaPipe solutions (same as backup) ─────────────────────────────
_mp_drawing = mp.solutions.drawing_utils
_mp_pose    = mp.solutions.pose

_POSE_LM_DRAW  = _mp_drawing.DrawingSpec(thickness=3, circle_radius=2,  color=(0, 0, 255))
_POSE_CON_DRAW = _mp_drawing.DrawingSpec(thickness=1, circle_radius=1,  color=(0, 220, 0))

# Visibility / presence threshold — mirrors get_idx_to_coordinates in backup
_VIS_THRESHOLD  = 0.45
_PRES_THRESHOLD = 0.45

# How many annotated frames to embed in the JSON response
_MAX_PREVIEW_FRAMES = 14

# Ideal angle ranges per joint (degrees)
_IDEAL_RANGES = {
    "front_knee":  (130, 175),
    "back_knee":   (130, 175),
    "front_elbow": (90,  145),
}

# Landmark indices — same as backup RightHandedBatting / LeftHandedBatting
_LANDMARKS = {
    "right": dict(
        shoulder=11, elbow=13, wrist=15,
        hip=23,      knee=25,  ankle=27,
        front_foot=29, back_foot=30,
        back_hip=24,   back_knee=26, back_ankle=28,
        opp_shoulder=12,
    ),
    "left": dict(
        shoulder=12, elbow=14, wrist=16,
        hip=24,      knee=26,  ankle=28,
        front_foot=30, back_foot=29,
        back_hip=23,   back_knee=25, back_ankle=27,
        opp_shoulder=11,
    ),
}

_GOOD_MESSAGES = frozenset({"Good technique!", "Good technique for this shot!"})


# ── Geometry helpers (identical to backup utils.py) ───────────────────
def _dot(vA, vB):
    return vA[0] * vB[0] + vA[1] * vB[1]


def _angle_deg(lineA, lineB) -> float:
    """Angle at shared vertex of two line segments (degrees)."""
    vA = (lineA[0][0] - lineA[1][0], lineA[0][1] - lineA[1][1])
    vB = (lineB[0][0] - lineB[1][0], lineB[0][1] - lineB[1][1])
    dot_p = _dot(vA, vB)
    magA  = _dot(vA, vA) ** 0.5
    magB  = _dot(vB, vB) ** 0.5
    if magA < 1e-6 or magB < 1e-6:
        return 0.0
    cos_  = max(-1.0, min(1.0, dot_p / (magA * magB)))
    ang   = math.degrees(math.acos(cos_)) % 360
    ang   = 180 - ang
    return (360 - ang) if ang - 180 >= 0 else ang


def _get_idx(results, h: int, w: int) -> dict[int, tuple[int, int]]:
    """
    Mirror of backup's get_idx_to_coordinates:
    Returns {landmark_index: (x_px, y_px)} for all visible landmarks.
    Filters by visibility and presence thresholds, just like the backup.
    """
    idx: dict[int, tuple[int, int]] = {}
    if not results.pose_landmarks:
        return idx
    for i, lm in enumerate(results.pose_landmarks.landmark):
        # Respect visibility / presence thresholds
        if lm.HasField("visibility") and lm.visibility < _VIS_THRESHOLD:
            continue
        if lm.HasField("presence") and lm.presence < _PRES_THRESHOLD:
            continue
        if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
            continue
        idx[i] = (
            min(int(lm.x * w), w - 1),
            min(int(lm.y * h), h - 1),
        )
    return idx


def _frame_to_b64(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 72])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _compute_technique_score(all_angles: list[dict], shot_history: list[str]) -> int:
    """
    Technique score 0-100 via three components:
      • Angle fitness   (50 pts) – % of frames where each joint is in ideal range
      • Shot recognition (25 pts) – proportion of non-Stance frames
      • Form stability  (25 pts) – inverse of angle variance
    """
    if not all_angles:
        return 0

    # Angle fitness
    fitness_parts = []
    for key, (lo, hi) in _IDEAL_RANGES.items():
        vals = [a[key] for a in all_angles if key in a]
        if vals:
            fitness_parts.append(sum(1 for v in vals if lo <= v <= hi) / len(vals))
    fitness = (sum(fitness_parts) / len(fitness_parts) * 50) if fitness_parts else 25.0

    # Shot recognition
    actual    = [s for s in shot_history if s != "Stance (Ready)"]
    shot_score = (len(actual) / len(shot_history) * 25) if shot_history else 0.0

    # Form stability
    stab_parts = []
    for key in _IDEAL_RANGES:
        vals = [a[key] for a in all_angles if key in a]
        if len(vals) < 2:
            continue
        mean_v = sum(vals) / len(vals)
        std    = (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5
        stab_parts.append(max(0.0, 1.0 - std / 35.0))
    stability = (sum(stab_parts) / len(stab_parts) * 25) if stab_parts else 12.5

    return min(100, max(0, round(fitness + shot_score + stability)))


# ── Path to the bundled pose model (only needed if Tasks API used) ─────
_HERE = Path(__file__).parent.parent.parent  # repo root


class HeadlessAnalyzer:
    """
    Runs the batting analysis pipeline on every frame of a video and
    returns a dict of aggregated statistics — no GUI involved.
    Uses mp.solutions.pose (the same API the backup uses) for reliable
    landmark detection.
    """

    def __init__(self) -> None:
        # mp.solutions.pose.Pose — identical to the backup
        self._pose = _mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    # ------------------------------------------------------------------
    def analyze(
        self,
        video_path: str,
        hand: str,
        view: str,
        intended_shot: str = "auto",
    ) -> dict:
        hand = hand.lower()
        view = view.lower()
        lm   = _LANDMARKS.get(hand, _LANDMARKS["right"])

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        classifier = ShotClassifier()
        advisor    = PostureAdvisor()

        all_angles:     list[dict] = []
        all_tips:       list[str]  = []
        shot_history:   list[str]  = []
        angle_timeline: list[dict] = []
        preview_frames: list[str]  = []

        # Modulo-based capture — works even when CAP_PROP_FRAME_COUNT == 0
        capture_interval = (
            max(1, total_frames // _MAX_PREVIEW_FRAMES)
            if total_frames > _MAX_PREVIEW_FRAMES else 10
        )

        frame_idx = 0

        while cap.isOpened():
            ok, bgr = cap.read()
            if not ok or bgr is None:
                break

            h, w = bgr.shape[:2]

            # ── Pose detection (mirror of backup processing loop) ─────
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self._pose.process(rgb)
            rgb.flags.writeable = True

            idx            = _get_idx(results, h, w)
            angles_data    = {}
            positions_data = {}

            # ── Front knee ───────────────────────────────────────────
            try:
                a = _angle_deg(
                    (idx[lm["hip"]],  idx[lm["knee"]]),
                    (idx[lm["knee"]], idx[lm["ankle"]]),
                )
                angles_data["front_knee"] = round(a, 1)
            except (KeyError, ZeroDivisionError):
                pass

            # ── Back knee (side view only) ────────────────────────────
            if view == "side":
                try:
                    a = _angle_deg(
                        (idx[lm["back_hip"]],  idx[lm["back_knee"]]),
                        (idx[lm["back_knee"]], idx[lm["back_ankle"]]),
                    )
                    angles_data["back_knee"] = round(a, 1)
                except (KeyError, ZeroDivisionError):
                    pass

            # ── Top-hand elbow ────────────────────────────────────────
            try:
                a = _angle_deg(
                    (idx[lm["wrist"]],  idx[lm["elbow"]]),
                    (idx[lm["elbow"]], idx[lm["shoulder"]]),
                )
                angles_data["front_elbow"] = round(a, 1)
            except (KeyError, ZeroDivisionError):
                pass

            # ── Positions ─────────────────────────────────────────────
            try:
                positions_data["front_wrist_y"]    = idx[lm["wrist"]][1]
                positions_data["front_shoulder_y"] = idx[lm["shoulder"]][1]
                positions_data["front_hip_y"]      = idx[lm["hip"]][1]
            except KeyError:
                pass
            try:
                positions_data["foot_spread"] = abs(
                    idx[lm["front_foot"]][0] - idx[lm["back_foot"]][0]
                )
                positions_data["shoulder_width"] = abs(
                    idx[lm["shoulder"]][0] - idx[lm["opp_shoulder"]][0]
                )
            except KeyError:
                pass

            # ── Classify + advise ─────────────────────────────────────
            if angles_data:
                shot = classifier.classify(angles_data, positions_data)
                tips = advisor.advise(angles_data, positions_data, shot, intended_shot)
                all_angles.append(angles_data)
                all_tips.extend(tips)
                shot_history.append(shot)
                if frame_idx % 5 == 0:
                    angle_timeline.append({"frame": frame_idx, **angles_data})

            # ── Capture annotated preview frame ───────────────────────
            if frame_idx % capture_interval == 0 and len(preview_frames) < _MAX_PREVIEW_FRAMES:
                annotated = bgr.copy()

                # Draw skeleton (using the same drawing_utils as backup)
                if results.pose_landmarks:
                    _mp_drawing.draw_landmarks(
                        annotated,
                        results.pose_landmarks,
                        _mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=_POSE_LM_DRAW,
                        connection_drawing_spec=_POSE_CON_DRAW,
                    )

                    # Angle labels next to joints
                    _draw_angle_labels(annotated, angles_data, idx, lm, view)

                # Shot / feedback overlay (identical to backup)
                shot_now  = classifier.current_shot
                color_now = classifier.get_color()
                tips_now  = advisor.advise(
                    angles_data, positions_data, shot_now, intended_shot
                )
                draw_analysis_overlay(annotated, shot_now, color_now, tips_now, intended_shot)

                # Frame watermark
                cv2.putText(
                    annotated, f"Frame {frame_idx}",
                    (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (160, 160, 160), 1,
                )
                b64 = _frame_to_b64(annotated)
                if b64:
                    preview_frames.append(b64)

            frame_idx += 1

        cap.release()
        try:
            self._pose.close()
        except Exception:
            pass

        return _aggregate(
            total_frames if total_frames > 0 else frame_idx,
            all_angles, all_tips, shot_history,
            angle_timeline, hand, view, intended_shot, preview_frames,
        )


def _draw_angle_labels(
    image: np.ndarray,
    angles_data: dict,
    idx: dict,
    lm: dict,
    view: str,
) -> None:
    """Draw angle values next to relevant joints."""
    try:
        if "front_knee" in angles_data:
            cv2.putText(image, f"{angles_data['front_knee']:.1f}\u00b0",
                        (idx[lm["knee"]][0] + 12, idx[lm["knee"]][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    except KeyError:
        pass
    try:
        if "back_knee" in angles_data and view == "side":
            cv2.putText(image, f"{angles_data['back_knee']:.1f}\u00b0",
                        (idx[lm["back_knee"]][0] + 12, idx[lm["back_knee"]][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    except KeyError:
        pass
    try:
        if "front_elbow" in angles_data:
            cv2.putText(image, f"{angles_data['front_elbow']:.1f}\u00b0",
                        (idx[lm["elbow"]][0] + 10, idx[lm["elbow"]][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    except KeyError:
        pass


# ── Aggregation ───────────────────────────────────────────────────────
def _aggregate(
    total_frames:   int,
    all_angles:     list[dict],
    all_tips:       list[str],
    shot_history:   list[str],
    angle_timeline: list[dict],
    hand:           str,
    view:           str,
    intended_shot:  str,
    preview_frames: list[str],
) -> dict:
    frames_analyzed = len(all_angles)
    shot_dist       = dict(Counter(shot_history))

    actual_shots = [s for s in shot_history if s != "Stance (Ready)"]
    dominant_shot = (
        Counter(actual_shots).most_common(1)[0][0]
        if actual_shots
        else (Counter(shot_history).most_common(1)[0][0] if shot_history else "Unknown")
    )

    avg_angles: dict = {}
    for key in ("front_knee", "back_knee", "front_elbow"):
        vals = [a[key] for a in all_angles if key in a]
        if vals:
            avg_angles[key] = round(sum(vals) / len(vals), 1)

    seen:  set  = set()
    tips:  list = []
    for tip in all_tips:
        if tip not in _GOOD_MESSAGES and tip not in seen:
            seen.add(tip)
            tips.append(tip)
    if not tips:
        tips = ["Great technique overall — keep it up!"]

    return {
        "total_frames":      total_frames,
        "frames_analyzed":   frames_analyzed,
        "shot_distribution": shot_dist,
        "dominant_shot":     dominant_shot,
        "average_angles":    avg_angles,
        "angle_timeline":    angle_timeline,
        "coaching_tips":     tips[:8],
        "technique_score":   _compute_technique_score(all_angles, shot_history),
        "hand":              hand,
        "view":              view,
        "intended_shot":     intended_shot,
        "preview_frames":    preview_frames,
    }
