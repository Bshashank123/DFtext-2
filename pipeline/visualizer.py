"""
Visualization Module — Fixed heatmap color scale

Key fix: heatmap uses ABSOLUTE score anchoring, not relative.
Previously normalized to max of current image → always showed red.
Now: 0.85+ = red, 0.5 = yellow, 0.35 = blue regardless of image.
"""

import cv2
import numpy as np


def draw_region_boxes(img_bgr, boxes_with_scores,
                      threshold_suspicious=0.65,
                      threshold_moderate=0.45):
    annotated = img_bgr.copy()

    for (box, score) in boxes_with_scores:
        x1, y1, x2, y2 = box

        if score > threshold_suspicious:
            color     = (0, 0, 220)
            label     = f"SUSPICIOUS {score:.2f}"
            thickness = 2
        elif score > threshold_moderate:
            color     = (0, 200, 255)
            label     = f"MODERATE {score:.2f}"
            thickness = 1
        else:
            color     = (0, 180, 0)
            label     = f"OK {score:.2f}"
            thickness = 1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label_y = max(y1 - 6, 12)
        cv2.putText(
            annotated, label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45, color, 1, cv2.LINE_AA
        )

    return annotated


def build_heatmap(gray_img, boxes_with_scores, max_score_anchor=0.85):
    """
    Absolute anchoring — score 0.85 = full red, 0.5 = yellow, 0.35 = blue.
    A clean document stays cool even if its max patch score is 0.4.
    """
    h, w = gray_img.shape[:2]
    heat = np.zeros((h, w), dtype=np.float32)

    for (box, score) in boxes_with_scores:
        x1, y1, x2, y2 = box
        heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], float(score))

    kernel_size = max(21, min(h, w) // 15)
    if kernel_size % 2 == 0:
        kernel_size += 1
    heat = cv2.GaussianBlur(heat, (kernel_size, kernel_size), 0)

    # Absolute scale — do NOT normalize to this image's max
    heat_anchored = np.clip(heat / max_score_anchor, 0.0, 1.0)

    heat_uint8    = (heat_anchored * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_JET)

    base_bgr = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
    overlay  = cv2.addWeighted(base_bgr, 0.55, heatmap_color, 0.45, 0)

    return overlay


def create_summary_panel(img_bgr, boxes_with_scores, image_score,
                         threshold_suspicious=0.65, threshold_moderate=0.45):
    annotated = draw_region_boxes(img_bgr, boxes_with_scores,
                                  threshold_suspicious, threshold_moderate)

    h, w     = img_bgr.shape[:2]
    target_h = min(h, 800)
    scale    = target_h / h
    target_w = int(w * scale)

    left  = cv2.resize(img_bgr,   (target_w, target_h))
    right = cv2.resize(annotated, (target_w, target_h))

    cv2.putText(left,  "ORIGINAL",          (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(right, "FORENSIC ANALYSIS", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    divider = np.full((target_h, 4, 3), 80, dtype=np.uint8)
    panel   = np.hstack([left, divider, right])

    bar_h = 50
    bar   = np.zeros((bar_h, panel.shape[1], 3), dtype=np.uint8)
    suspicious_count = sum(1 for (_, s) in boxes_with_scores if s > threshold_suspicious)

    if image_score > threshold_suspicious:
        verdict, bar_color = "AI EDIT LIKELY",    (0, 0, 180)
    elif image_score > threshold_moderate:
        verdict, bar_color = "SUSPICIOUS",         (0, 150, 220)
    else:
        verdict, bar_color = "APPEARS AUTHENTIC",  (0, 140, 0)

    bar[:] = bar_color
    cv2.putText(
        bar,
        f"VERDICT: {verdict}  |  Score: {image_score:.2f}  |  Suspicious: {suspicious_count}/{len(boxes_with_scores)}",
        (10, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA
    )

    return np.vstack([panel, bar])