from collections import deque

import cv2
import mediapipe as mp

from src.Batting.Batting import Batting
from src.Analysis.shot_classifier import ShotClassifier
from src.Analysis.posture_advisor import PostureAdvisor
from src.Analysis.overlay import draw_analysis_overlay
from src.utils import *


class RightHandedBatting(Batting):
    """
    Analyses right-handed batting from a recorded video.

    Landmarks used (left body side – dominant for right-handers):
        11 left shoulder  |  13 left elbow   |  15 left wrist
        23 left hip       |  25 left knee     |  27 left ankle
        29 left foot index|  30 right foot index
        11/12 shoulders (for stance-width reference)
    """

    def __init__(self, video_path: str, intended_shot: str = 'auto'):
        self.video_path = video_path
        self.intended_shot = intended_shot
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.pose_landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=5, circle_radius=2, color=(0, 0, 255))
        self.pose_connection_drawing_spec = self.mp_drawing.DrawingSpec(
            thickness=1, circle_radius=1, color=(0, 255, 0))
        self.PRESENCE_THRESHOLD = 0.5
        self.VISIBILITY_THRESHOLD = 0.5

    # ──────────────────────────────────────────────────────────────────
    def front_batting(self):
        cap        = cv2.VideoCapture(self.video_path)
        classifier = ShotClassifier()
        advisor    = PostureAdvisor()

        with self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:

            while cap.isOpened():
                success, image = cap.read()
                if not success or image is None:
                    break

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.pose_landmark_drawing_spec,
                    connection_drawing_spec=self.pose_connection_drawing_spec)

                idx = get_idx_to_coordinates(image, results)
                angles_data   = {}
                positions_data = {}

                # Batsman label
                if 0 in idx:
                    cv2.putText(image, "Batsman : Right Handed",
                                (idx[0][0] - 120, idx[0][1] - 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)

                # ── Front (left) knee angle ──────────────────────────
                try:
                    l1 = np.linspace(idx[23], idx[25], 100)
                    l2 = np.linspace(idx[25], idx[27], 100)
                    cv2.line(image,
                             (int(l1[99][0]), int(l1[99][1])),
                             (int(l1[59][0]), int(l1[59][1])), (0, 0, 255), 6)
                    cv2.line(image,
                             (int(l2[0][0]),  int(l2[0][1])),
                             (int(l2[40][0]), int(l2[40][1])), (0, 0, 255), 6)
                    a = ang((idx[23], idx[25]), (idx[25], idx[27]))
                    angles_data['front_knee'] = a
                    cv2.putText(image, str(round(a, 2)),
                                (idx[25][0] + 20, idx[25][1]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    c, r, sa, ea = convert_arc(l1[90], l2[10], sagitta=15)
                    draw_ellipse(image, c, (r, r), -1, sa, ea, 255)
                except Exception:
                    pass

                # ── Top-hand elbow angle (wrist-elbow-shoulder) ──────
                try:
                    l1 = np.linspace(idx[15], idx[13], 100)
                    l2 = np.linspace(idx[13], idx[11], 100)
                    cv2.line(image,
                             (int(l1[99][0]), int(l1[99][1])),
                             (int(l1[59][0]), int(l1[59][1])), (0, 0, 255), 6)
                    cv2.line(image,
                             (int(l2[0][0]),  int(l2[0][1])),
                             (int(l2[40][0]), int(l2[40][1])), (0, 0, 255), 6)
                    a = ang((idx[15], idx[13]), (idx[13], idx[11]))
                    angles_data['front_elbow'] = a
                    cv2.putText(image, str(round(a, 2)),
                                (idx[13][0] + 10, idx[13][1]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    c, r, sa, ea = convert_arc(l1[90], l2[10], sagitta=15)
                    draw_ellipse(image, c, (r, r), -1, sa, ea, 255)
                except Exception:
                    pass

                # ── Shoulder-hip-knee (body lean) ────────────────────
                try:
                    l1 = np.linspace(idx[11], idx[23], 100)
                    l2 = np.linspace(idx[23], idx[25], 100)
                    cv2.line(image,
                             (int(l1[99][0]), int(l1[99][1])),
                             (int(l1[75][0]), int(l1[75][1])), (0, 0, 255), 6)
                    cv2.line(image,
                             (int(l2[0][0]),  int(l2[0][1])),
                             (int(l2[30][0]), int(l2[30][1])), (0, 0, 255), 6)
                    a = ang((idx[11], idx[23]), (idx[23], idx[25]))
                    cv2.putText(image, str(round(a, 2)),
                                (idx[23][0] + 20, idx[23][1]),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 0), 2)
                    c, r, sa, ea = convert_arc(l1[90], l2[10], sagitta=15)
                    draw_ellipse(image, c, (r, r), -1, sa, ea, 255)
                except Exception:
                    pass

                # ── Foot-to-hand distance ────────────────────────────
                try:
                    lf = idx[29]
                    lh = idx[17]
                    cv2.line(image, (lf[0], lf[1]+50), (lh[0], lf[1]+50), (255,255,0), 2)
                    cv2.line(image, (lf[0], lf[1]+50), (lf[0], lf[1]+60), (255,255,0), 2)
                    cv2.line(image, (lh[0], lf[1]+50), (lh[0], lf[1]+60), (255,255,0), 2)
                    cv2.putText(image,
                                str(max(0, lf[0] - lh[0])) + " px",
                                (lf[0] - 50, lf[1] + 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                except Exception:
                    pass

                # ── Collect positions for analysis ───────────────────
                try:
                    positions_data['front_wrist_y']    = idx[15][1]
                    positions_data['front_shoulder_y'] = idx[11][1]
                    positions_data['front_hip_y']      = idx[23][1]
                except Exception:
                    pass
                try:
                    positions_data['foot_spread']    = abs(idx[30][0] - idx[29][0])
                    positions_data['shoulder_width'] = abs(idx[12][0] - idx[11][0])
                except Exception:
                    pass

                # ── Ball tracking (green) ────────────────────────────
                try:
                    hsv    = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                    kernel = np.ones((5, 5), np.uint8)
                    mask   = cv2.inRange(hsv,
                                         np.array([29, 86, 6]),
                                         np.array([64, 255, 255]))
                    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                    mask   = cv2.dilate(mask, kernel, iterations=1)
                    cnts, _ = cv2.findContours(mask.copy(),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)[-2:]
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        ((x, y), radius) = cv2.minEnclosingCircle(c)
                        M = cv2.moments(c)
                        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"]))
                        if radius > 10:
                            cv2.circle(image, (int(x), int(y)), int(radius), (0,255,255), 2)
                            cv2.circle(image, center, 5, (0, 0, 255), -1)
                            cv2.putText(image, "Ball",
                                        (int(x) + int(radius) + 4, int(y)),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        0.45, (0, 255, 255), 1, cv2.LINE_AA)
                except Exception:
                    pass

                # ── Shot classification & posture feedback ───────────
                shot   = classifier.classify(angles_data, positions_data)
                tips   = advisor.advise(angles_data, positions_data, shot, self.intended_shot)
                draw_analysis_overlay(image, shot, classifier.get_color(), tips, self.intended_shot)

                cv2.imshow('CricketLytics', rescale_frame(image, percent=100))
                if cv2.waitKey(25) & 0xFF == 27:
                    break
                if cv2.waitKey(1) == ord('p'):
                    cv2.waitKey(-1)

        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()

    # ──────────────────────────────────────────────────────────────────
    def side_batting(self):
        cap        = cv2.VideoCapture(self.video_path)
        classifier = ShotClassifier()
        advisor    = PostureAdvisor()

        with self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:

            while cap.isOpened():
                success, image = cap.read()
                if not success or image is None:
                    break

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                self.mp_drawing.draw_landmarks(
                    image, results.pose_landmarks,
                    self.mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.pose_landmark_drawing_spec,
                    connection_drawing_spec=self.pose_connection_drawing_spec)

                idx = get_idx_to_coordinates(image, results)
                angles_data    = {}
                positions_data = {}

                if 0 in idx:
                    cv2.putText(image, "Batsman : Right Handed",
                                (idx[0][0] - 120, idx[0][1] - 100),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)

                # ── Front (left) knee ────────────────────────────────
                try:
                    l1 = np.linspace(idx[23], idx[25], 100)
                    l2 = np.linspace(idx[25], idx[27], 100)
                    cv2.line(image,
                             (int(l1[99][0]), int(l1[99][1])),
                             (int(l1[59][0]), int(l1[59][1])), (0, 0, 255), 6)
                    cv2.line(image,
                             (int(l2[0][0]),  int(l2[0][1])),
                             (int(l2[40][0]), int(l2[40][1])), (0, 0, 255), 6)
                    a = ang((idx[23], idx[25]), (idx[25], idx[27]))
                    angles_data['front_knee'] = a
                    cv2.putText(image, str(round(a, 2)),
                                (idx[25][0] + 10, idx[25][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    c, r, sa, ea = convert_arc(l1[90], l2[10], sagitta=15)
                    draw_ellipse(image, c, (r, r), -1, sa, ea, 255)
                except Exception:
                    pass

                # ── Back (right) knee ────────────────────────────────
                try:
                    l1 = np.linspace(idx[24], idx[26], 100)
                    l2 = np.linspace(idx[26], idx[28], 100)
                    cv2.line(image,
                             (int(l1[99][0]), int(l1[99][1])),
                             (int(l1[59][0]), int(l1[59][1])), (0, 0, 255), 6)
                    cv2.line(image,
                             (int(l2[0][0]),  int(l2[0][1])),
                             (int(l2[40][0]), int(l2[40][1])), (0, 0, 255), 6)
                    a = ang((idx[24], idx[26]), (idx[26], idx[28]))
                    angles_data['back_knee'] = a
                    cv2.putText(image, str(round(a, 2)),
                                (idx[26][0] + 20, idx[26][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    c, r, sa, ea = convert_arc(l1[90], l2[10], sagitta=15)
                    draw_ellipse(image, c, (r, r), -1, sa, ea, 255)
                except Exception:
                    pass

                # ── Top-hand elbow ───────────────────────────────────
                try:
                    l1 = np.linspace(idx[15], idx[13], 100)
                    l2 = np.linspace(idx[13], idx[11], 100)
                    cv2.line(image,
                             (int(l1[99][0]), int(l1[99][1])),
                             (int(l1[59][0]), int(l1[59][1])), (0, 0, 255), 6)
                    cv2.line(image,
                             (int(l2[0][0]),  int(l2[0][1])),
                             (int(l2[40][0]), int(l2[40][1])), (0, 0, 255), 6)
                    a = ang((idx[15], idx[13]), (idx[13], idx[11]))
                    angles_data['front_elbow'] = a
                    cv2.putText(image, str(round(a, 2)),
                                (idx[13][0] + 10, idx[13][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    c, r, sa, ea = convert_arc(l1[90], l2[10], sagitta=15)
                    draw_ellipse(image, c, (r, r), -1, sa, ea, 255)
                except Exception:
                    pass

                # ── Body lean (shoulder-hip-knee) ────────────────────
                try:
                    l1 = np.linspace(idx[11], idx[23], 100)
                    l2 = np.linspace(idx[23], idx[25], 100)
                    cv2.line(image,
                             (int(l1[99][0]), int(l1[99][1])),
                             (int(l1[75][0]), int(l1[75][1])), (0, 0, 255), 6)
                    cv2.line(image,
                             (int(l2[0][0]),  int(l2[0][1])),
                             (int(l2[30][0]), int(l2[30][1])), (0, 0, 255), 6)
                    a = ang((idx[11], idx[23]), (idx[23], idx[25]))
                    cv2.putText(image, str(round(a, 2)),
                                (idx[23][0] + 10, idx[23][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    c, r, sa, ea = convert_arc(l1[90], l2[10], sagitta=15)
                    draw_ellipse(image, c, (r, r), -1, sa, ea, 255)
                except Exception:
                    pass

                # ── Foot-to-hand & foot-to-foot distances ────────────
                try:
                    lf = idx[29]
                    lh = idx[17]
                    cv2.line(image, (lf[0], lf[1]+50), (lh[0], lf[1]+50), (255,255,0), 2)
                    cv2.line(image, (lf[0], lf[1]+50), (lf[0], lf[1]+60), (255,255,0), 2)
                    cv2.line(image, (lh[0], lf[1]+50), (lh[0], lf[1]+60), (255,255,0), 2)
                    cv2.putText(image, str(max(0, lf[0]-lh[0])) + " px",
                                (lf[0]-100, lf[1]+80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                except Exception:
                    pass
                try:
                    lf = idx[29]
                    rf = idx[30]
                    cv2.line(image, (rf[0], rf[1]+50), (lf[0], rf[1]+50), (255,51,51), 2)
                    cv2.line(image, (rf[0], rf[1]+50), (rf[0], rf[1]+60), (255,51,51), 2)
                    cv2.line(image, (lf[0], rf[1]+50), (lf[0], rf[1]+60), (255,51,51), 2)
                    cv2.putText(image, str(max(0, rf[0]-lf[0])) + " px",
                                (rf[0]-100, rf[1]+80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 2)
                    positions_data['foot_spread'] = abs(rf[0] - lf[0])
                except Exception:
                    pass

                # ── Collect positions for analysis ───────────────────
                try:
                    positions_data['front_wrist_y']    = idx[15][1]
                    positions_data['front_shoulder_y'] = idx[11][1]
                    positions_data['front_hip_y']      = idx[23][1]
                    positions_data['shoulder_width']   = abs(idx[12][0] - idx[11][0])
                except Exception:
                    pass

                # ── Shot classification & posture feedback ───────────
                shot = classifier.classify(angles_data, positions_data)
                tips = advisor.advise(angles_data, positions_data, shot, self.intended_shot)
                draw_analysis_overlay(image, shot, classifier.get_color(), tips, self.intended_shot)

                cv2.imshow('CricketLytics', rescale_frame(image, percent=100))
                if cv2.waitKey(25) & 0xFF == 27:
                    break
                if cv2.waitKey(1) == ord('p'):
                    cv2.waitKey(-1)

        cap.release()
        cv2.destroyAllWindows()
        self.pose.close()
