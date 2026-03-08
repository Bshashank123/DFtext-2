"""
Image Forensics Module
Pixel-level forensic analysis: noise, strokes, paper texture.

This module operates on the whole image and returns aggregate scores.
For per-region analysis, use the patch pipeline (anomaly_detector.py).
"""

import cv2
import numpy as np
from scipy.stats import entropy

from pipeline.stroke_analysis import analyze_stroke_naturalness


# ==================== NOISE ANALYSIS ====================

def extract_noise_residual(gray_img):
    """Extract noise by subtracting smoothed version."""
    blurred = cv2.GaussianBlur(gray_img, (5, 5), 0)
    noise = gray_img.astype(np.float32) - blurred.astype(np.float32)
    return noise


def noise_variance_map(noise, block_size=32):
    """Measure noise variance across spatial blocks."""
    h, w = noise.shape
    variances = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = noise[y:y + block_size, x:x + block_size]
            variances.append(np.var(block))

    return np.array(variances)


def fft_energy_ratio(noise):
    """Analyze frequency spectrum of noise."""
    fft = np.fft.fft2(noise)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)

    h, w = magnitude.shape
    center = magnitude[h // 4:3 * h // 4, w // 4:3 * w // 4]
    high_freq = magnitude.sum() - center.sum()
    low_freq = center.sum()

    return high_freq / (low_freq + 1e-6)


def noise_authenticity_score(gray_img):
    """
    Whole-image noise analysis.
    Returns [0,1] where higher = more synthetic-like.
    """
    noise = extract_noise_residual(gray_img)
    variances = noise_variance_map(noise)

    if len(variances) == 0:
        return 0.5

    variance_std = np.std(variances)
    variance_mean = np.mean(variances)
    freq_ratio = fft_energy_ratio(noise)
    variance_cv = variance_std / (variance_mean + 1e-6)

    # Variance scoring
    if variance_std > 25:
        var_score = 0.0
    elif variance_std < 8:
        var_score = 1.0
    else:
        var_score = (25 - variance_std) / 17.0

    # Frequency scoring
    if freq_ratio > 1.2:
        freq_score = 0.0
    elif freq_ratio < 0.6:
        freq_score = 1.0
    else:
        freq_score = (1.2 - freq_ratio) / 0.6

    # CV scoring
    if variance_cv > 0.8:
        cv_score = 0.0
    elif variance_cv < 0.3:
        cv_score = 1.0
    else:
        cv_score = (0.8 - variance_cv) / 0.5

    return float(np.clip(0.45 * var_score + 0.35 * freq_score + 0.20 * cv_score, 0.0, 1.0))


# ==================== PAPER TEXTURE ====================

def extract_background_regions(gray_img, threshold=180):
    """Sample ink-free (paper background) regions."""
    background_mask = gray_img > threshold
    return gray_img[background_mask]


def texture_entropy(pixels):
    """Measure randomness in paper texture."""
    if len(pixels) < 100:
        return 0.0
    hist, _ = np.histogram(pixels, bins=30, density=True)
    return entropy(hist + 1e-6)


def analyze_paper_grain(gray_img):
    """
    Analyze paper grain patterns.
    Real paper: visible fiber structure.
    AI paper: too clean or artificial.
    Returns [0,1] where higher = more synthetic.
    """
    bg_pixels = extract_background_regions(gray_img, threshold=180)

    if len(bg_pixels) < 500:
        return 0.5

    tex_entropy = texture_entropy(bg_pixels)
    tex_variance = np.var(bg_pixels)

    if len(bg_pixels) > 1000:
        sample = bg_pixels[:1000]
        local_vars = [np.var(sample[i:i + 10]) for i in range(0, len(sample) - 10, 10)]
        texture_roughness = np.std(local_vars)
    else:
        texture_roughness = 0.0

    median_val = np.median(bg_pixels)
    uniformity_ratio = np.sum(np.abs(bg_pixels - median_val) < 5) / len(bg_pixels)

    # Entropy scoring
    if 2.0 < tex_entropy < 3.5:
        entropy_score = 0.0
    else:
        entropy_score = min(1.0, abs(tex_entropy - 2.75) / 1.5)

    # Variance scoring
    if 60 < tex_variance < 180:
        var_score = 0.0
    elif tex_variance < 30:
        var_score = 1.0
    elif tex_variance > 250:
        var_score = 0.8
    else:
        var_score = 0.4

    # Roughness scoring
    if texture_roughness > 12:
        rough_score = 0.0
    elif texture_roughness < 5:
        rough_score = 1.0
    else:
        rough_score = (12 - texture_roughness) / 7.0

    # Uniformity scoring
    if 0.25 < uniformity_ratio < 0.45:
        uniform_score = 0.0
    elif uniformity_ratio < 0.15 or uniformity_ratio > 0.6:
        uniform_score = 1.0
    else:
        uniform_score = 0.5

    score = 0.35 * entropy_score + 0.30 * var_score + 0.20 * rough_score + 0.15 * uniform_score
    return float(np.clip(score, 0.0, 1.0))


# ==================== FUSION ====================

def analyze_image_forensics(gray_img):
    """
    Run all image forensics modules on the full image.
    Returns dict of scores — used as global context.
    For region-specific detection, use the patch pipeline.
    """
    noise_score  = noise_authenticity_score(gray_img)
    stroke_score = analyze_stroke_naturalness(gray_img)
    paper_score  = analyze_paper_grain(gray_img)

    # Paper is the most reliable signal, stroke second
    image_forensics_score = (
        0.20 * noise_score +
        0.35 * stroke_score +
        0.45 * paper_score
    )

    return {
        'noise': noise_score,
        'stroke': stroke_score,
        'paper': paper_score,
        'image_forensics': float(np.clip(image_forensics_score, 0.0, 1.0))
    }
