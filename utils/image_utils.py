"""
Image Utilities
Helpers for loading, saving, and converting images.
"""

import cv2
import numpy as np
from PIL import Image
import base64
import io


def load_image(filepath):
    """
    Load an image from disk.
    Returns BGR numpy array (OpenCV default).
    """
    img = cv2.imread(filepath)
    if img is None:
        raise ValueError(f"Could not read image: {filepath}")
    return img


def load_image_from_bytes(file_bytes):
    """
    Load an image from raw bytes (e.g., Flask file upload).
    Returns BGR numpy array.
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image from bytes")
    return img


def image_to_base64(img):
    """
    Convert numpy image to base64 PNG string.
    Useful for embedding result images in JSON responses.
    """
    success, buffer = cv2.imencode('.png', img)
    if not success:
        return None
    b64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def save_image(img, filepath):
    """Save image to disk."""
    cv2.imwrite(filepath, img)
