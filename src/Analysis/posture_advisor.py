class PostureAdvisor:
    """
    Gives real-time posture feedback for batting.

    When an intended_shot is provided (not 'auto'), targeted coaching cues
    are returned based on the biomechanical requirements of that shot.
    Otherwise, generic angle-based checks are used.
    """

    # ideal angle ranges per shot: (front_knee_min, front_knee_max, elbow_min)
    _SHOT_IDEALS = {
        'drive':      (155, 180, 100),
        'defensive':  (140, 165,  80),
        'pull':       (145, 170,  95),
        'sweep':      ( 90, 135,  80),
        'cut':        (148, 175,  90),
        'back-punch': (150, 180, 100),
    }

    def advise(self, angles: dict, positions: dict,
               detected_shot: str, intended_shot: str = 'auto') -> list:
        if intended_shot and intended_shot != 'auto':
            return self._targeted(angles, positions, intended_shot)
        return self._generic(angles, positions, detected_shot)

    # ── Generic (auto mode) ───────────────────────────────────────────
    def _generic(self, angles, positions, shot_name):
        tips = []
        fk = angles.get('front_knee')
        bk = angles.get('back_knee')
        fe = angles.get('front_elbow')
        foot_spread    = positions.get('foot_spread', 0)
        shoulder_width = positions.get('shoulder_width', 0)

        if fk is not None:
            if fk > 172 and shot_name not in ("Back Foot Punch", "Cut Shot"):
                tips.append("Bend front knee more")
            elif fk < 115 and shot_name != "Sweep Shot":
                tips.append("Front knee over-flexed")
        if bk is not None and bk > 172:
            tips.append("Bend back knee slightly")
        if fe is not None and fe < 78:
            tips.append("Raise top elbow higher")
        if foot_spread > 0 and shoulder_width > 0:
            ratio = foot_spread / shoulder_width
            if ratio < 0.75:
                tips.append("Widen your stance")
            elif ratio > 2.3 and shot_name == "Stance (Ready)":
                tips.append("Narrow your stance slightly")

        return tips if tips else ["Good technique!"]

    # ── Shot-specific targeted advice ─────────────────────────────────
    def _targeted(self, angles, positions, shot):
        tips = []
        fk = angles.get('front_knee')
        bk = angles.get('back_knee')
        fe = angles.get('front_elbow')
        wy = positions.get('front_wrist_y',    999)
        sy = positions.get('front_shoulder_y', 500)
        hand_above = wy < sy

        if shot == 'drive':
            if fk is not None and fk < 155:
                tips.append("Drive: extend front leg — push into the ball")
            if fe is not None and fe < 100:
                tips.append("Drive: raise top elbow for full swing")
            if not hand_above:
                tips.append("Drive: follow through — hands above shoulder")

        elif shot == 'defensive':
            if fk is not None and fk > 168:
                tips.append("Defensive: bend front knee — get over the ball")
            if fe is not None and fe > 130:
                tips.append("Defensive: keep elbow compact and close")
            if hand_above:
                tips.append("Defensive: keep hands low — soft hands")

        elif shot == 'pull':
            if not hand_above:
                tips.append("Pull: get hands higher — hit on top of bounce")
            if bk is not None and bk > 168:
                tips.append("Pull: pivot on back foot — bend back knee")
            if fe is not None and fe < 90:
                tips.append("Pull: both elbows up for power")

        elif shot == 'sweep':
            if fk is not None and fk > 140:
                tips.append("Sweep: get lower — bend front knee more")
            if bk is not None and bk < 145:
                tips.append("Sweep: extend back leg for balance")
            if hand_above:
                tips.append("Sweep: keep hands low, bat horizontal")

        elif shot == 'cut':
            if bk is not None and bk < 148:
                tips.append("Cut: move back — plant weight on back foot")
            if fk is not None and fk < 148:
                tips.append("Cut: step back and across before cutting")
            if not hand_above:
                tips.append("Cut: get hands up to cut down on the ball")

        elif shot == 'back-punch':
            if bk is not None and bk < 150:
                tips.append("Back punch: plant back foot — don't collapse")
            if fe is not None and fe < 100:
                tips.append("Back punch: punch through — straighten elbow")
            if fk is not None and fk < 150:
                tips.append("Back punch: stay upright on back foot")

        return tips if tips else ["Good technique for this shot!"]
