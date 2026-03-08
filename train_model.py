"""
DFText — Supervised Model Training Script

Trains a Random Forest binary classifier on real vs fake patches.
Requires BOTH real and fake/AI-generated handwriting images.

Usage:
    venv\Scripts\python.exe train_model.py \
        --real-dir datasets/real \
        --fake-dir datasets/edited

The more images in each folder, the better.
Minimum: 10 images per class. Recommended: 20+ per class.
"""

import argparse
import os
import sys
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.preprocess import preprocess_image
from pipeline.text_regions import detect_text_regions
from pipeline.anomaly_detector import train_model
from utils.patch_utils import extract_patches

SUPPORTED = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}


def load_patches_from_dir(directory, label):
    """Load all images from directory and extract patches."""
    all_patches = []
    files = [f for f in sorted(os.listdir(directory))
             if os.path.splitext(f)[1].lower() in SUPPORTED]

    if not files:
        print(f"  No images found in {directory}")
        return []

    for fname in files:
        fpath = os.path.join(directory, fname)
        img = cv2.imread(fpath)
        if img is None:
            print(f"  ⚠ Could not read: {fname}")
            continue

        gray = preprocess_image(img)
        boxes = detect_text_regions(gray)
        patch_data = extract_patches(gray, boxes, padding=20)
        patches = [p for p, _ in patch_data]
        all_patches.extend(patches)
        print(f"  [{label}] {fname}: {len(patches)} patches")

    return all_patches


def main():
    parser = argparse.ArgumentParser(description='Train DFText supervised classifier')
    parser.add_argument('--real-dir',  default='datasets/real',
                        help='Real handwriting images directory')
    parser.add_argument('--fake-dir',  default='datasets/edited',
                        help='AI-generated/edited handwriting images directory')
    args = parser.parse_args()

    # Validate directories
    for d, name in [(args.real_dir, 'real'), (args.fake_dir, 'fake')]:
        if not os.path.isdir(d):
            print(f"Error: {name} directory not found: {d}")
            print(f"Create it and add handwriting images.")
            sys.exit(1)

    print(f"\nLoading REAL images from:  {args.real_dir}")
    real_patches = load_patches_from_dir(args.real_dir, 'REAL')

    print(f"\nLoading FAKE images from:  {args.fake_dir}")
    fake_patches = load_patches_from_dir(args.fake_dir, 'FAKE')

    print(f"\nTotal: {len(real_patches)} real patches, {len(fake_patches)} fake patches")

    if len(real_patches) < 10:
        print("Not enough real patches (need 10+). Add more real images.")
        sys.exit(1)
    if len(fake_patches) < 10:
        print("Not enough fake patches (need 10+).")
        print(f"Add Gemini/AI-generated handwriting images to: {args.fake_dir}")
        sys.exit(1)

    print(f"\nTraining supervised Random Forest classifier...")

    # Delete old models first
    import glob
    for old in glob.glob('models/*.pkl'):
        os.remove(old)
        print(f"  Removed old model: {old}")

    model, scaler, score_stats = train_model(
        real_patches,
        fake_patches,
        save=True
    )

    sep = score_stats['separation']
    print(f"\n{'='*50}")
    print(f"✅ Training complete!")
    if sep > 0.40:
        print(f"   Quality: EXCELLENT — separation {sep:.3f}")
    elif sep > 0.25:
        print(f"   Quality: GOOD — separation {sep:.3f}")
    elif sep > 0.10:
        print(f"   Quality: OK — add more diverse fake images to improve")
    else:
        print(f"   Quality: POOR — add more fake images, different AI tools")
    print(f"{'='*50}")
    print(f"\nRun the app:  venv\\Scripts\\python.exe app.py")


if __name__ == '__main__':
    main()