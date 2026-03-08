"""
Image Preprocessing Module
Standardizes images without enhancing them.
Preserves generation artifacts while removing camera/lighting bias.
"""

import cv2
import numpy as np


def preprocess_image(img):
    """
    Standardize image for forensic analysis.

    Steps:
        1. Convert to grayscale
        2. Resize to fixed width (1200px)
        3. Normalize contrast (CLAHE)
        4. Very light denoising (bilateral - preserves edges)

    We do NOT beautify. We standardize.
    """
    # Step 1: Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Step 2: Resize to fixed width keeping aspect ratio
    target_width = 1200
    h, w = gray.shape
    if w != target_width:
        aspect_ratio = h / w
        new_h = int(target_width * aspect_ratio)
        gray = cv2.resize(gray, (target_width, new_h), interpolation=cv2.INTER_AREA)

    # Step 3: CLAHE contrast normalization
    # Removes lighting bias, keeps noise pattern intact
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    normalized = clahe.apply(gray)

    # Step 4: Very light bilateral denoising
    # Preserves edges but smooths extreme outliers
    denoised = cv2.bilateralFilter(normalized, d=3, sigmaColor=10, sigmaSpace=10)

    return denoised


def get_grayscale(img):
    """Helper: convert to grayscale only."""
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()
