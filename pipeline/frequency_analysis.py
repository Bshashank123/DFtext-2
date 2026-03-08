"""
Frequency Analysis Module
Detects frequency spectrum anomalies per patch using FFT.

Why frequency analysis?
  Real handwriting on real paper has a characteristic frequency distribution:
  - Paper grain and ink texture contribute specific high-frequency content
  - Camera sensor adds its own frequency signature

  AI-edited regions often show:
  - Different high-frequency energy distribution
  - Unnatural smoothness in certain frequency bands
  - Missing or altered frequency peaks

Per-patch FFT workflow:
  1. Apply FFT to patch
  2. Shift zero-frequency to center
  3. Compute magnitude spectrum
  4. Extract energy in frequency bands (low, mid, high)
  5. Return band ratios as features
"""

import numpy as np
import cv2


def compute_fft_magnitude(patch):
    """
    Compute the 2D FFT magnitude spectrum of a patch.

    Args:
        patch: Grayscale numpy array

    Returns:
        magnitude: 2D FFT magnitude spectrum (same size as patch)
    """
    # Zero-pad to power of 2 for efficiency
    h, w = patch.shape
    fft = np.fft.fft2(patch.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    return magnitude


def extract_band_energies(magnitude):
    """
    Partition the frequency spectrum into 3 bands and compute energy ratios.

    Bands (by radius from center):
      - Low frequency: center 25% — coarse structure
      - Mid frequency: 25-60%    — texture, edges
      - High frequency: 60-100%  — fine detail, noise, sensor pattern

    Returns:
        dict with low_energy, mid_energy, high_energy, high_to_low_ratio,
        high_to_mid_ratio, spectral_entropy
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cx, cy)

    # Build radius grid
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    radius_map = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

    low_mask  = radius_map <= 0.25 * max_radius
    mid_mask  = (radius_map > 0.25 * max_radius) & (radius_map <= 0.60 * max_radius)
    high_mask = radius_map > 0.60 * max_radius

    total = magnitude.sum() + 1e-6

    low_energy  = float(magnitude[low_mask].sum()  / total)
    mid_energy  = float(magnitude[mid_mask].sum()  / total)
    high_energy = float(magnitude[high_mask].sum() / total)

    high_to_low = float(high_energy / (low_energy + 1e-6))
    high_to_mid = float(high_energy / (mid_energy + 1e-6))

    # Spectral entropy: how spread is energy across the spectrum?
    # Real paper: spread. AI: concentrated or missing bands.
    mag_flat = magnitude.flatten()
    mag_norm = mag_flat / (mag_flat.sum() + 1e-6)
    spectral_entropy = float(-np.sum(mag_norm * np.log(mag_norm + 1e-9)))

    return {
        'low_energy': low_energy,
        'mid_energy': mid_energy,
        'high_energy': high_energy,
        'high_to_low_ratio': high_to_low,
        'high_to_mid_ratio': high_to_mid,
        'spectral_entropy': spectral_entropy
    }


def radial_profile(magnitude, n_bins=16):
    """
    Compute radially-averaged frequency profile.
    Captures how energy falls off from center to edge.
    Provides a richer feature than just 3 band energies.

    Returns:
        profile: 1D numpy array of length n_bins (normalized)
    """
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    max_radius = min(cx, cy)

    y_coords, x_coords = np.mgrid[0:h, 0:w]
    radius_map = np.sqrt((y_coords - cy) ** 2 + (x_coords - cx) ** 2)

    profile = np.zeros(n_bins)
    bin_size = max_radius / n_bins

    for i in range(n_bins):
        r_min = i * bin_size
        r_max = (i + 1) * bin_size
        mask = (radius_map >= r_min) & (radius_map < r_max)
        if mask.sum() > 0:
            profile[i] = magnitude[mask].mean()

    # Normalize
    total = profile.sum()
    if total > 0:
        profile = profile / total

    return profile


def extract_frequency_feature_vector(patch, n_radial_bins=16):
    """
    Full pipeline: patch → FFT → feature vector.

    Feature vector = [low_energy, mid_energy, high_energy,
                      high_to_low_ratio, high_to_mid_ratio,
                      spectral_entropy, *radial_profile]

    Returns numpy array suitable for Isolation Forest.
    """
    if patch.shape[0] < 8 or patch.shape[1] < 8:
        return np.zeros(6 + n_radial_bins, dtype=np.float32)

    magnitude = compute_fft_magnitude(patch)
    bands = extract_band_energies(magnitude)
    profile = radial_profile(magnitude, n_bins=n_radial_bins)

    base_features = np.array([
        bands['low_energy'],
        bands['mid_energy'],
        bands['high_energy'],
        bands['high_to_low_ratio'],
        bands['high_to_mid_ratio'],
        bands['spectral_entropy']
    ], dtype=np.float32)

    return np.concatenate([base_features, profile])
