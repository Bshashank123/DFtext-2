"""
DFText — AI Handwriting Tampering Detector
"""

import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template

from pipeline.preprocess import preprocess_image
from pipeline.text_regions import detect_text_regions
from pipeline.image_forensics import analyze_image_forensics
from pipeline.anomaly_detector import score_patches, load_model
from pipeline.visualizer import draw_region_boxes, build_heatmap
from utils.patch_utils import extract_patches
from utils.image_utils import load_image_from_bytes, image_to_base64

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'}

print("Loading forensic model...")
_MODEL, _SCALER, _SCORE_STATS = load_model()
if _MODEL is not None:
    mode = _SCORE_STATS.get('mode', 'unknown') if _SCORE_STATS else 'no stats'
    print(f"  Model loaded — mode: {mode}")
else:
    print("  No trained model found — using heuristic fallback")
    print("  Run: venv\\Scripts\\python.exe train_model.py --real-dir datasets/real --fake-dir datasets/edited")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def run_analysis(img_bgr):
    gray = _preprocess(img_bgr)

    boxes = detect_text_regions(gray)
    if not boxes:
        boxes = [(0, 0, gray.shape[1], gray.shape[0])]

    patch_data  = extract_patches(gray, boxes, padding=20)
    patches     = [p for p, _ in patch_data]
    patch_boxes = [b for _, b in patch_data]

    try:
        anomaly_scores = score_patches(patches, _MODEL, _SCALER, _SCORE_STATS)
    except Exception as e:
        print(f"Warning: patch scoring failed: {e}")
        anomaly_scores = [0.3] * len(patches)

    if len(anomaly_scores) != len(patch_boxes):
        anomaly_scores = [0.3] * len(patch_boxes)

    try:
        global_scores = analyze_image_forensics(gray)
    except Exception as e:
        print(f"Warning: global forensics failed: {e}")
        global_scores = {'noise': 0.3, 'stroke': 0.3, 'paper': 0.3, 'image_forensics': 0.3}

    if anomaly_scores:
        patch_mean = float(np.mean(anomaly_scores))
        patch_max  = float(np.max(anomaly_scores))
        overall = (
            0.60 * (0.7 * patch_mean + 0.3 * patch_max) +
            0.40 * global_scores['image_forensics']
        )
    else:
        overall = global_scores['image_forensics']

    overall = float(np.clip(overall, 0.0, 1.0))

    if overall < 0.40:
        verdict    = "Appears Authentic"
        confidence = int(70 + (0.40 - overall) / 0.40 * 25)
    elif overall < 0.65:
        verdict    = "Suspicious"
        confidence = int(50 + abs(overall - 0.525) / 0.125 * 20)
    else:
        verdict    = "AI Edit Likely"
        confidence = int(70 + (overall - 0.65) / 0.35 * 25)

    confidence = max(40, min(95, confidence))

    boxes_with_scores = list(zip(patch_boxes, anomaly_scores))

    region_results = []
    for (box, score) in boxes_with_scores:
        x1, y1, x2, y2 = box
        status = "suspicious" if score > 0.65 else "moderate" if score > 0.45 else "normal"
        region_results.append({
            'box':    [x1, y1, x2, y2],
            'score':  round(float(score), 3),
            'status': status
        })

    try:
        annotated     = draw_region_boxes(img_bgr, boxes_with_scores)
        heatmap       = build_heatmap(gray, boxes_with_scores)
        annotated_b64 = image_to_base64(annotated)
        heatmap_b64   = image_to_base64(heatmap)
    except Exception as e:
        print(f"Warning: visualization failed: {e}")
        annotated_b64 = None
        heatmap_b64   = None

    return {
        'verdict':            verdict,
        'confidence':         confidence,
        'overall_score':      round(overall, 3),
        'region_count':       len(region_results),
        'suspicious_regions': sum(1 for r in region_results if r['status'] == 'suspicious'),
        'region_results':     region_results,
        'global_forensics': {
            'noise':           round(float(global_scores['noise']), 3),
            'stroke':          round(float(global_scores['stroke']), 3),
            'paper':           round(float(global_scores['paper']), 3),
            'image_forensics': round(float(global_scores['image_forensics']), 3)
        },
        'annotated_image': annotated_b64,
        'heatmap_image':   heatmap_b64
    }


def _preprocess(img_bgr):
    if len(img_bgr.shape) == 3:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_bgr.copy()

    h, w = gray.shape
    if w > 1200:
        new_h = int(h * 1200 / w)
        gray  = cv2.resize(gray, (1200, new_h), interpolation=cv2.INTER_AREA)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray  = clahe.apply(gray)
    gray  = cv2.bilateralFilter(gray, d=3, sigmaColor=10, sigmaSpace=10)
    return gray


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400

    try:
        img_bgr = load_image_from_bytes(file.read())
        if img_bgr is None:
            return jsonify({'error': 'Could not decode image'}), 400
        result = run_analysis(img_bgr)
        return jsonify(result)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"Analysis error:\n{tb}")
        return jsonify({'error': str(e), 'traceback': tb}), 500


@app.route('/compare', methods=['POST'])
def compare():
    if 'real' not in request.files or 'suspect' not in request.files:
        return jsonify({'error': 'Need both real and suspect image fields'}), 400

    try:
        real_r    = run_analysis(load_image_from_bytes(request.files['real'].read()))
        suspect_r = run_analysis(load_image_from_bytes(request.files['suspect'].read()))

        diffs = {
            k: round(abs(suspect_r['global_forensics'][k] - real_r['global_forensics'][k]), 3)
            for k in ['noise', 'stroke', 'paper', 'image_forensics']
        }
        diffs['overall'] = round(abs(suspect_r['overall_score'] - real_r['overall_score']), 3)

        return jsonify({
            'real':              {k: v for k, v in real_r.items()
                                  if k not in ('annotated_image', 'heatmap_image', 'region_results')},
            'suspect':           {k: v for k, v in suspect_r.items()
                                  if k not in ('annotated_image', 'heatmap_image', 'region_results')},
            'differences':       diffs,
            'strongest_signal':  max(diffs.items(), key=lambda x: x[1])[0],
            'suspect_annotated': suspect_r['annotated_image'],
            'real_annotated':    real_r['annotated_image']
        })
    except Exception as e:
        import traceback
        return jsonify({'error': str(e), 'traceback': traceback.format_exc()}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)