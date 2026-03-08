"""
Patch Extraction Utilities
Extracts image patches around text regions with configurable padding.
Each patch is one "unit of analysis" for the forensic pipeline.
"""

import numpy as np
import cv2


def extract_patch(gray_img, box, padding=20):
    """
    Extract a single padded patch around a bounding box.

    Args:
        gray_img: Grayscale image (H x W numpy array)
        box: (x1, y1, x2, y2) bounding box
        padding: pixels to add around the box on each side

    Returns:
        patch: numpy array, or None if too small
    """
    h, w = gray_img.shape[:2]
    x1, y1, x2, y2 = box

    # Add padding and clamp
    px1 = max(0, x1 - padding)
    py1 = max(0, y1 - padding)
    px2 = min(w, x2 + padding)
    py2 = min(h, y2 + padding)

    if (px2 - px1) < 10 or (py2 - py1) < 10:
        return None

    return gray_img[py1:py2, px1:px2].copy()


def extract_patches(gray_img, boxes, padding=20, min_size=30):
    """
    Extract patches for all detected text regions.

    Args:
        gray_img: Grayscale image
        boxes: List of (x1, y1, x2, y2) bounding boxes
        padding: Margin around each box
        min_size: Minimum patch dimension in pixels

    Returns:
        patches: List of (patch, box) tuples
    """
    results = []

    for box in boxes:
        patch = extract_patch(gray_img, box, padding=padding)
        if patch is None:
            continue
        if patch.shape[0] < min_size or patch.shape[1] < min_size:
            continue
        results.append((patch, box))

    return results


def normalize_patch_size(patch, target_size=(64, 64)):
    """
    Resize patch to a fixed size for consistent feature extraction.
    Maintains aspect ratio with padding if needed.
    """
    return cv2.resize(patch, target_size, interpolation=cv2.INTER_AREA)
