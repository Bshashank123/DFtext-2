"""
Noise Analysis Module
Extracts and analyzes camera noise residuals per patch.

The core idea:
  Real photos have spatially non-uniform, sensor-dependent noise.
  AI-edited regions have DIFFERENT noise statistics than surrounding real paper.

  When AI tools modify a region:
  - They may resynthesize it
  - Even if visually seamless, the noise pattern changes
  - We detect this change via noise residual statistics

Per-patch workflow:
  1. Denoise patch
  2. Compute residual (original - denoised) = noise signal
  3. Extract statistics from residual
  4. These statistics become features for anomaly detection
"""

import cv2
import numpy as np
from scipy.stats import kurtosis, skew


def extract_noise_residual(patch):
    """
    Extract noise residual from a patch.

    Subtracting a denoised version isolates the noise signal.
    We use a gentle Gaussian blur as the denoiser.

    Args:
        patch: Grayscale numpy array

    Returns:
        noise: Float32 numpy array (same size as patch)
    """
    blurred = cv2.GaussianBlur(patch, (5, 5), 0)
    noise = patch.astype(np.float32) - blurred.astype(np.float32)
    return noise


def noise_statistics(noise):
    """
    Extract statistical features from a noise residual.

    Features:
        - variance: Overall noise power. Real photos have higher local variance
        - std: Standard deviation of noise
        - kurtosis: Tail behavior. Real sensor noise is more Gaussian
        - skewness: Asymmetry. AI-synthesized noise may be skewed
        - spatial_cv: Coefficient of variation across blocks.
                      Real: uneven. AI: too uniform.

    Args:
        noise: Float32 numpy array

    Returns:
        features: dict of noise statistics
    """
    flat = noise.flatten()

    if len(flat) < 25:
        return {
            'variance': 0.0,
            'std': 0.0,
            'kurtosis': 0.0,
            'skewness': 0.0,
            'spatial_cv': 0.0
        }

    var = float(np.var(flat))
    std = float(np.std(flat))

    try:
        kurt = float(kurtosis(flat))
    except Exception:
        kurt = 0.0

    try:
        skewness = float(skew(flat))
    except Exception:
        skewness = 0.0

    # Spatial CV: variance across 8x8 blocks
    spatial_cv = _spatial_coefficient_of_variation(noise, block_size=8)

    return {
        'variance': var,
        'std': std,
        'kurtosis': kurt,
        'skewness': skewness,
        'spatial_cv': spatial_cv
    }


def _spatial_coefficient_of_variation(noise, block_size=8):
    """
    Measure how uneven the noise is across spatial blocks.

    Real photos: noise varies across regions (high CV).
    AI-generated: noise is more uniform (low CV).
    """
    h, w = noise.shape
    block_variances = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = noise[y:y + block_size, x:x + block_size]
            block_variances.append(np.var(block))

    if not block_variances:
        return 0.0

    arr = np.array(block_variances)
    mean = np.mean(arr)
    if mean < 1e-6:
        return 0.0

    return float(np.std(arr) / mean)


def extract_noise_feature_vector(patch):
    """
    Full pipeline: patch → noise residual → feature vector.

    Returns a fixed-length numpy array suitable for Isolation Forest.
    Order: [variance, std, kurtosis, skewness, spatial_cv]
    """
    noise = extract_noise_residual(patch)
    stats = noise_statistics(noise)

    return np.array([
        stats['variance'],
        stats['std'],
        stats['kurtosis'],
        stats['skewness'],
        stats['spatial_cv']
    ], dtype=np.float32)
