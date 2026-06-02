import cv2

SHOT_LABELS = {
    'auto':       'Auto Detect',
    'drive':      'Drive',
    'defensive':  'Forward Defensive',
    'pull':       'Pull / Hook',
    'sweep':      'Sweep',
    'cut':        'Cut Shot',
    'back-punch': 'Back Foot Punch',
}


def draw_analysis_overlay(image, detected_shot: str, shot_color: tuple,
                          feedback: list, intended_shot: str = 'auto') -> None:
    """
    Draws two semi-transparent panels onto *image* (in-place):
      - Top banner  : intended shot (if set) + detected shot
      - Bottom panel: targeted posture feedback
    """
    h, w = image.shape[:2]
    overlay = image.copy()

    # ── Top banner ──────────────────────────────────────────────────────
    banner_h = 52
    cv2.rectangle(overlay, (0, 0), (w, banner_h), shot_color, -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    if intended_shot and intended_shot != 'auto':
        intended_label = SHOT_LABELS.get(intended_shot, intended_shot.title())
        # Left: intended shot
        cv2.putText(image, f"Intended: {intended_label}",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (220, 220, 220), 1, cv2.LINE_AA)
        # Right: detected shot
        cv2.putText(image, f"Detected: {detected_shot}",
                    (10, 44), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(image, f"Shot: {detected_shot}",
                    (10, 36), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)

    # ── Bottom feedback panel ───────────────────────────────────────────
    line_h  = 24
    pad     = 8
    panel_h = pad + 20 + len(feedback) * line_h + pad
    panel_w = 370

    overlay2 = image.copy()
    cv2.rectangle(overlay2, (0, h - panel_h), (panel_w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.65, image, 0.35, 0, image)

    header = (f"{SHOT_LABELS.get(intended_shot, 'Posture')} Tips"
              if intended_shot != 'auto' else "Posture Feedback")
    cv2.putText(image, header,
                (10, h - panel_h + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (160, 160, 160), 1, cv2.LINE_AA)

    for i, tip in enumerate(feedback):
        color = (0, 255, 120) if "Good" in tip else (0, 200, 255)
        cv2.putText(image, f"  {tip}",
                    (10, h - panel_h + 18 + pad + (i + 1) * line_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    color, 1, cv2.LINE_AA)
