"""
STROKE ANALYSIS - COMPLETELY REWRITTEN
The old version was returning 1.0 for everything
This version uses different approach: texture analysis instead of edge detection
"""

import cv2
import numpy as np
from scipy.stats import entropy
from scipy.ndimage import generic_filter

def extract_ink_regions(gray_img, threshold=127):
    """
    Extract only the ink (dark) regions
    """
    # Invert: ink becomes white, paper becomes black
    inverted = 255 - gray_img
    
    # Threshold to get binary mask
    _, binary = cv2.threshold(inverted, threshold, 255, cv2.THRESH_BINARY)
    
    return binary

def analyze_ink_texture(gray_img):
    """
    Analyze texture of ink strokes
    Real ink: grainy, variable intensity
    AI ink: smooth, uniform
    """
    # Get ink regions
    ink_mask = extract_ink_regions(gray_img)
    
    # Get intensity values in ink regions
    ink_pixels = gray_img[ink_mask > 0]
    
    if len(ink_pixels) < 100:
        return 0.5  # Not enough ink
    
    # 1. Intensity variance (real ink varies more)
    intensity_var = np.var(ink_pixels)
    
    # 2. Local variance (texture roughness)
    def local_variance(values):
        return np.var(values)
    
    # Calculate local variance in 5x5 windows
    local_var_map = generic_filter(gray_img.astype(float), local_variance, size=5)
    local_var_in_ink = local_var_map[ink_mask > 0]
    
    if len(local_var_in_ink) > 0:
        texture_roughness = np.mean(local_var_in_ink)
    else:
        texture_roughness = 0.0
    
    # 3. Edge sharpness (AI edges are too perfect)
    edges = cv2.Canny(gray_img, 100, 200)
    edge_in_ink = edges[ink_mask > 0]
    edge_density = np.sum(edge_in_ink) / (len(ink_pixels) + 1)
    
    # Real handwriting: intensity_var ~400-800, texture_roughness ~50-150, edge_density ~0.1-0.3
    # AI: intensity_var ~200-400, texture_roughness ~20-50, edge_density ~0.05-0.15
    
    # Scoring
    if intensity_var > 500:
        var_score = 0.0  # Real
    elif intensity_var < 300:
        var_score = 1.0  # AI
    else:
        var_score = (500 - intensity_var) / 200
    
    if texture_roughness > 100:
        texture_score = 0.0  # Real
    elif texture_roughness < 40:
        texture_score = 1.0  # AI
    else:
        texture_score = (100 - texture_roughness) / 60
    
    if edge_density > 0.25:
        edge_score = 0.0  # Real - lots of variation
    elif edge_density < 0.1:
        edge_score = 1.0  # AI - too smooth
    else:
        edge_score = (0.25 - edge_density) / 0.15
    
    # Combine
    score = 0.4 * var_score + 0.35 * texture_score + 0.25 * edge_score
    
    return max(0.0, min(1.0, score))

def analyze_stroke_pressure(gray_img):
    """
    Analyze pressure variation (width changes)
    Real: varies with pressure
    AI: constant width
    """
    # Get ink regions
    ink_mask = extract_ink_regions(gray_img)
    
    # Distance transform shows "thickness" at each point
    dist_transform = cv2.distanceTransform(ink_mask, cv2.DIST_L2, 5)
    
    # Get thickness values where there's ink
    thickness_values = dist_transform[ink_mask > 0]
    
    if len(thickness_values) < 50:
        return 0.5
    
    # Calculate statistics
    thickness_std = np.std(thickness_values)
    thickness_range = np.ptp(thickness_values)  # max - min
    thickness_cv = thickness_std / (np.mean(thickness_values) + 1e-6)
    
    # Real: std ~2-5, range ~8-20, CV ~0.3-0.7
    # AI: std ~0.5-2, range ~2-8, CV ~0.1-0.3
    
    if thickness_std > 3:
        std_score = 0.0
    elif thickness_std < 1.5:
        std_score = 1.0
    else:
        std_score = (3 - thickness_std) / 1.5
    
    if thickness_range > 12:
        range_score = 0.0
    elif thickness_range < 5:
        range_score = 1.0
    else:
        range_score = (12 - thickness_range) / 7
    
    if thickness_cv > 0.4:
        cv_score = 0.0
    elif thickness_cv < 0.2:
        cv_score = 1.0
    else:
        cv_score = (0.4 - thickness_cv) / 0.2
    
    score = 0.4 * std_score + 0.3 * range_score + 0.3 * cv_score
    
    return max(0.0, min(1.0, score))

def analyze_stroke_direction(gray_img):
    """
    Analyze stroke direction changes
    Real: natural curves, direction changes
    AI: too smooth, mechanical
    """
    # Calculate gradients
    sobelx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Get angles
    angles = np.arctan2(sobely, sobelx)
    
    # Get ink regions
    ink_mask = extract_ink_regions(gray_img)
    angles_in_ink = angles[ink_mask > 0]
    
    if len(angles_in_ink) < 100:
        return 0.5
    
    # Calculate directional variance
    # Real handwriting has more angular variation
    angle_std = np.std(angles_in_ink)
    
    # Histogram of angles
    hist, _ = np.histogram(angles_in_ink, bins=36, range=(-np.pi, np.pi))
    angle_entropy = entropy(hist + 1)
    
    # Real: angle_std ~0.8-1.5, entropy ~2.5-3.5
    # AI: angle_std ~0.4-0.8, entropy ~1.5-2.5
    
    if angle_std > 1.0:
        std_score = 0.0
    elif angle_std < 0.6:
        std_score = 1.0
    else:
        std_score = (1.0 - angle_std) / 0.4
    
    if angle_entropy > 3.0:
        ent_score = 0.0
    elif angle_entropy < 2.0:
        ent_score = 1.0
    else:
        ent_score = (3.0 - angle_entropy) / 1.0
    
    score = 0.6 * std_score + 0.4 * ent_score
    
    return max(0.0, min(1.0, score))

def analyze_stroke_naturalness(gray_img):
    """
    MAIN FUNCTION: Combines all stroke analyses
    Returns score [0,1] where higher = more AI-like
    """
    # Run all analyses
    texture_score = analyze_ink_texture(gray_img)
    pressure_score = analyze_stroke_pressure(gray_img)
    direction_score = analyze_stroke_direction(gray_img)
    
    # Weighted combination
    final_score = (
        0.40 * texture_score +
        0.35 * pressure_score +
        0.25 * direction_score
    )
    
    return max(0.0, min(1.0, final_score))