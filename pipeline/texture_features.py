"""
Texture Feature Extraction Module
Detects paper texture differences between patches using Local Binary Patterns (LBP).

Why LBP?
  LBP captures micro-texture: paper grain, fiber structure, noise pattern.
  AI-edited regions have DIFFERENT micro-texture than surrounding real paper.
  This difference is invisible to the human eye but measurable.

A normal patch has consistent LBP histogram with surrounding patches.
An AI-edited patch will show a different distribution.
"""

import numpy as np
from skimage.feature import local_binary_pattern


def compute_lbp(patch, radius=1, n_points=8, method='uniform'):
    """
    Compute Local Binary Pattern for a single patch.

    Args:
        patch: Grayscale numpy array (single patch)
        radius: Radius of LBP neighborhood
        n_points: Number of circularly symmetric neighbor points
        method: 'uniform' reduces noise, good for texture

    Returns:
        lbp: LBP image (same size as patch)
    """
    lbp = local_binary_pattern(patch, n_points, radius, method=method)
    return lbp


def lbp_histogram(patch, radius=1, n_points=8, n_bins=None):
    """
    Compute normalized LBP histogram for a patch.

    The histogram is the feature vector — it captures the
    statistical distribution of micro-texture patterns.

    Args:
        patch: Grayscale numpy array
        radius: LBP radius
        n_points: LBP neighbor points
        n_bins: Number of histogram bins (default: n_points + 2 for uniform)

    Returns:
        hist: Normalized histogram (1D numpy array)
    """
    if n_bins is None:
        n_bins = n_points + 2  # uniform LBP has n_points+2 patterns

    lbp = compute_lbp(patch, radius=radius, n_points=n_points)

    hist, _ = np.histogram(
        lbp.ravel(),
        bins=n_bins,
        range=(0, n_bins),
        density=True
    )

    # Avoid zeros for chi-squared distance later
    hist = hist + 1e-7

    return hist


def extract_texture_features(patch, scales=None):
    """
    Extract multi-scale LBP texture features from a patch.

    Using multiple scales captures both fine and coarse texture.
    Fine scale: individual paper fibers, ink texture
    Coarse scale: broader noise patterns

    Args:
        patch: Grayscale numpy array
        scales: List of (radius, n_points) tuples

    Returns:
        feature_vector: Concatenated normalized histograms
    """
    if scales is None:
        scales = [
            (1, 8),   # Fine texture
            (2, 16),  # Medium texture
            (3, 24),  # Coarse texture
        ]

    if patch.shape[0] < 10 or patch.shape[1] < 10:
        # Patch too small — return zero vector
        total_bins = sum(n_pts + 2 for _, n_pts in scales)
        return np.zeros(total_bins)

    feature_parts = []
    for radius, n_points in scales:
        hist = lbp_histogram(patch, radius=radius, n_points=n_points)
        feature_parts.append(hist)

    return np.concatenate(feature_parts)


def texture_distance(hist_a, hist_b):
    """
    Compute chi-squared distance between two LBP histograms.

    Chi-squared is the standard distance metric for histogram comparison.
    Small distance = similar texture.
    Large distance = different texture = potential edit boundary.

    Returns: float distance value
    """
    return 0.5 * np.sum((hist_a - hist_b) ** 2 / (hist_a + hist_b + 1e-7))
