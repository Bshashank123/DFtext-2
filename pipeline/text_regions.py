"""
Text Region Detection — Word-Level, Clean Boxes Only
"""

import cv2
import numpy as np

MAX_PATCHES = 60


def detect_text_regions(gray_img):
    try:
        boxes = _detect_words(gray_img)
    except Exception as e:
        print(f"  text_regions error: {e}")
        boxes = []

    if not boxes:
        boxes = _ink_grid_fallback(gray_img)

    if len(boxes) > MAX_PATCHES:
        step  = len(boxes) / MAX_PATCHES
        boxes = [boxes[int(i * step)] for i in range(MAX_PATCHES)]

    return boxes


def _detect_words(gray_img):
    h, w = gray_img.shape[:2]

    binary = cv2.adaptiveThreshold(
        gray_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=10
    )

    # Remove noise dots
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k_open, iterations=1)

    # Merge letters into words
    k_word = cv2.getStructuringElement(cv2.MORPH_RECT, (18, 2))
    word_mask = cv2.dilate(binary, k_word, iterations=1)

    k_vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 6))
    word_mask = cv2.dilate(word_mask, k_vert, iterations=1)

    contours, _ = cv2.findContours(
        word_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes    = []
    min_area = h * w * 0.0004
    max_area = h * w * 0.15

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)

        if bh < h * 0.008:
            continue
        if bh > h * 0.10:
            continue
        if bw < w * 0.015:
            continue

        aspect = bw / (bh + 1e-6)
        if aspect < 0.4 or aspect > 20:
            continue

        region = binary[y:y+bh, x:x+bw]
        fill   = region.sum() / (255 * bw * bh + 1e-6)
        if fill < 0.04 or fill > 0.90:
            continue

        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + bw)
        y2 = min(h, y + bh)
        boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda b: (b[1] // (h // 25 + 1), b[0]))
    return boxes


def _ink_grid_fallback(gray_img):
    h, w   = gray_img.shape[:2]
    binary = cv2.adaptiveThreshold(
        gray_img, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 25, 10
    )
    row_ink  = binary.sum(axis=1)
    ink_rows = np.where(row_ink > row_ink.max() * 0.05)[0]
    if len(ink_rows) == 0:
        return [(0, 0, w, h)]

    patch_h = max(40, h // 12)
    boxes   = []
    for y in range(int(ink_rows[0]), int(ink_rows[-1]), patch_h):
        y2     = min(h, y + patch_h)
        region = binary[y:y2, 0:w]
        if region.sum() / (255 * region.size + 1e-6) > 0.01:
            boxes.append((0, y, w, y2))
    return boxes[:MAX_PATCHES] if boxes else [(0, 0, w, h)]