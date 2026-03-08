# DFText — AI Handwriting Forgery Detector

A forensic tool that detects AI-generated or AI-edited handwriting in scanned documents.
It uses **micro-texture analysis** at the word level — examining pixel-level signals that
AI editing tools cannot perfectly replicate.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Training the Model](#training-the-model)
6. [Running the Application](#running-the-application)
7. [Using the Web Interface](#using-the-web-interface)
8. [Detection Method — Technical Details](#detection-method)
9. [Accuracy & Results](#accuracy--results)
10. [Troubleshooting](#troubleshooting)

---

## How It Works

DFText does **not** use OCR or read the text content. Instead it analyses the raw pixel
texture of handwriting at the word level. The core insight is:

> Real handwriting photographed on paper has **physical imperfections** — ink pressure
> variation, paper grain, camera sensor noise, and uneven contrast. AI-generated or
> AI-edited handwriting has **suspicious uniformity** — too clean, too consistent.

The system compares every word patch in the document against a trained model that has
learned the difference between real and AI-generated textures.

---

## Project Structure

```
DFText/
├── app.py                    ← Flask web server + analysis pipeline
├── train_model.py            ← Standalone training script
├── requirements.txt
│
├── pipeline/
│   ├── preprocess.py         ← Resize + CLAHE + bilateral filter
│   ├── text_regions.py       ← Word-level bounding box detection (OpenCV)
│   ├── texture_features.py   ← LBP multi-scale texture histograms
│   ├── noise_analysis.py     ← Noise residual statistics per patch
│   ├── frequency_analysis.py ← FFT band energies + radial profile
│   ├── anomaly_detector.py   ← Random Forest classifier + scoring
│   ├── image_forensics.py    ← Global image-level forensic signals
│   ├── stroke_analysis.py    ← Pen stroke consistency analysis
│   └── visualizer.py         ← Heatmap + bounding box overlay
│
├── utils/
│   ├── patch_utils.py        ← Patch extraction with padding
│   └── image_utils.py        ← Image load/save helpers
│
├── models/
│   ├── classifier.pkl        ← Trained Random Forest (created by train_model.py)
│   ├── scaler.pkl            ← StandardScaler (created by train_model.py)
│   └── score_stats.pkl       ← Score distribution stats (created by train_model.py)
│
├── datasets/
│   ├── real/                 ← Real handwriting images for training
│   └── edited/               ← AI-generated/edited images for training
│
├── templates/
│   └── index.html            ← Web UI
│
└── outputs/                  ← Analysis output images (auto-created)
```

---

## Requirements

- **Python 3.10** (recommended — tested and working)
- Windows 10/11
- ~2GB disk space for dependencies

---

## Installation

### Step 1 — Clone or extract the project

```
cd C:\path\to\DFText
```

### Step 2 — Create a Python 3.10 virtual environment

```powershell
py -3.10 -m venv venv
```

### Step 3 — Activate the virtual environment

```powershell
venv\Scripts\activate
```

You should see `(venv)` at the start of your prompt.

### Step 4 — Upgrade pip

```powershell
venv\Scripts\python.exe -m pip install --upgrade pip
```

### Step 5 — Install PyTorch (CPU only)

```powershell
venv\Scripts\pip.exe install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
```

### Step 6 — Install all other dependencies

```powershell
venv\Scripts\pip.exe install Flask==2.3.3 Werkzeug==2.3.7 opencv-python==4.8.0.76 Pillow==10.0.0 numpy==1.24.3 scipy==1.11.2 scikit-image==0.21.0 scikit-learn==1.3.0 easyocr==1.7.0 matplotlib==3.7.2
```

---

## Training the Model

The model is a **supervised Random Forest** trained on real vs AI-generated handwriting.
You must train before first use — the `models/` folder is empty on a fresh install.

### Step 1 — Prepare training data

```
datasets/
├── real/     ← Put real handwritten document images here (JPG/PNG)
│               Aim for 30+ images, different writers and paper types
└── edited/   ← Put AI-generated/edited images here (JPG/PNG)
                Gemini, DALL-E, Stable Diffusion, or Photoshop edits
```

**Tips for good training data:**
- Real images: photos of actual handwritten notes, assignments, exam papers
- Edited images: AI-generated handwriting, or real notes with AI-inserted words
- More variety = better generalisation
- Current trained model: 33 real + 16 Gemini-generated images

### Step 2 — Run training

```powershell
venv\Scripts\python.exe train_model.py --real-dir datasets/real --fake-dir datasets/edited
```

**Expected output:**
```
Loading images from datasets/real...   33 images loaded
Loading images from datasets/edited... 16 images loaded
Extracting features: 660 real + 320 fake patches...
Balanced dataset: 320 real, 320 fake

Training performance:
              precision    recall  f1-score
        Real       0.97      0.98      0.97
        Fake       0.98      0.97      0.97

ROC AUC: 0.998

Real avg score: 0.089  (want < 0.35)
Fake avg score: 0.885  (want > 0.65)
Separation:     0.796  (EXCELLENT)

Model saved to models/classifier.pkl
```

**What the numbers mean:**
- **ROC AUC 0.998** — near-perfect class separation
- **Separation 0.796** — real and fake scores are far apart
- Separation > 0.30 = good, > 0.50 = excellent, > 0.70 = outstanding

### Retraining

Retrain any time you add new images to `datasets/real/` or `datasets/edited/`:

```powershell
venv\Scripts\python.exe train_model.py --real-dir datasets/real --fake-dir datasets/edited
```

The old model files are automatically overwritten.

---

## Running the Application

### Start the server

```powershell
venv\Scripts\python.exe app.py
```

**Expected startup output:**
```
Loading forensic model...
  Model loaded — mode: supervised
 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

### Open the web interface

```
http://localhost:5000
```

### Stop the server

Press `Ctrl + C` in the terminal.

---

## Using the Web Interface

### Single Document Analysis

1. Drag and drop (or click to upload) a handwritten document image
2. Supported formats: PNG, JPG, JPEG, BMP, TIFF, WEBP (max 16MB)
3. Click **Analyse Document**
4. Results appear showing:
   - **Verdict**: Appears Authentic / Suspicious / AI Edit Likely
   - **Confidence %**
   - **Overall score** (0.0 = definitely real, 1.0 = definitely AI)
   - **Forensic Annotation** — image with coloured boxes per word
   - **Anomaly Heatmap** — heat overlay showing suspicious regions
   - **Global Forensics Scores** — noise, stroke, paper, overall
   - **Region Details** — per-word scores table

### Box colours in Forensic Annotation

| Colour | Meaning |
|--------|---------|
| 🔴 Red box | SUSPICIOUS — this word is anomalous vs the page |
| 🟡 Yellow box | MODERATE — slightly anomalous |
| 🟢 Green box | OK — consistent with the rest of the document |

### Compare Two Documents (API)

POST to `/compare` with `real` and `suspect` file fields to compare a known-authentic
document against a suspect one. Returns diff scores for each forensic signal.

---

## Detection Method

### Why not OCR / text analysis?

OCR destroys the pixel-level signals we care about. We do **not** read the text.
We analyse the raw image texture of each word patch.

### Pipeline (in order)

#### 1. Preprocessing (`pipeline/preprocess.py`)
- Convert to grayscale
- Resize to **1200px width** (must match training resolution — do not change)
- CLAHE contrast normalisation — removes lighting bias while preserving noise pattern
- Light bilateral filtering — smooths outliers, preserves edges

#### 2. Word Detection (`pipeline/text_regions.py`)
- Adaptive threshold (`blockSize=25, C=10`) to find ink pixels
- Morphological open (`3×3`) to remove noise dots
- Horizontal dilation (`18×2`) to merge letters into word-width blobs
- Vertical dilation (`1×6`) to capture ascenders/descenders
- Contour detection — each contour becomes one word box
- Filters: min area, max area, aspect ratio, ink fill density
- Sorted into reading order (top→bottom, left→right)

#### 3. Patch Extraction (`utils/patch_utils.py`)
- Each word box is extracted as a padded image patch (20px padding)
- Patches smaller than 30×30px are skipped
- Patches with < 1% ink are skipped (blank margins)

#### 4. Feature Extraction — 3 modules, ~71 features total

**Texture features** (`pipeline/texture_features.py`) — ~30 features
- Multi-scale Local Binary Pattern (LBP) histograms
- Captures micro-texture of ink strokes and paper surface
- Real handwriting: rough, variable texture
- AI handwriting: unnaturally smooth and uniform

**Noise features** (`pipeline/noise_analysis.py`) — 5 features
- Noise residual after Gaussian smoothing
- Mean, std, kurtosis, skewness of noise pattern
- Real photos: DSLR/phone camera sensor noise present
- AI images: missing natural camera noise, or artificial noise added uniformly

**Frequency features** (`pipeline/frequency_analysis.py`) — ~22 features
- 2D FFT of each patch
- Band energies (low/mid/high frequency) + radial power profile
- Real handwriting: natural frequency distribution from ink + paper
- AI handwriting: different spectral signature from generation process

**Discriminative extras** (`pipeline/anomaly_detector.py`) — 14 features
- Local std in 8×8 blocks (texture roughness)
- Gradient magnitude statistics (Sobel)
- Background uniformity (paper texture variance)
- Ink pixel statistics (density, variance, skewness)
- Edge density (Canny) + Laplacian variance (sharpness)

#### 5. Classification (`pipeline/anomaly_detector.py`)
- **Model**: `RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5)`
- Wrapped in `CalibratedClassifierCV(cv=3, method='isotonic')` for reliable probabilities
- Output: probability that each patch is AI-generated (0.0–1.0)

#### 6. Relative Scoring (`app.py → _relative_scores`)
- Raw scores tell us "how AI-like is this globally"
- Relative scoring measures each word vs **the document's own baseline**
- Uses median as baseline (robust to outliers)
- Z-score normalisation within the document
- Blended: 30% absolute + 70% relative
- Result: only words genuinely anomalous vs the rest of the page get flagged

#### 7. Global Forensics (`pipeline/image_forensics.py`)
- Image-level signals: noise consistency, stroke regularity, paper texture
- Combined with patch scores: `60% patches + 40% global forensics`

#### 8. Verdict
| Score | Verdict |
|-------|---------|
| < 0.40 | Appears Authentic |
| 0.40 – 0.65 | Suspicious |
| > 0.65 | AI Edit Likely |

---

## Accuracy & Results

Trained on 33 real + 16 Gemini-generated images:

| Metric | Value |
|--------|-------|
| Training Accuracy | 97% |
| ROC AUC | 0.998 |
| Real avg score | 0.089 |
| Fake avg score | 0.885 |
| Separation | 0.796 (EXCELLENT) |

**Known limitations:**
- Model trained primarily on Gemini-generated images
- May miss forgeries from other tools (ChatGPT/DALL-E, Stable Diffusion, Photoshop)
- Add diverse fake samples to `datasets/edited/` and retrain to improve generalisation
- Printed/typed documents score higher than real handwriting (model not calibrated for them)

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'pipeline.doc_type'`**
→ Old `app.py` still on disk. Replace with the latest version.

**`No trained model found — using heuristic fallback`**
→ Run `train_model.py` first. Check that `models/classifier.pkl` exists after training.

**All words showing as SUSPICIOUS**
→ This was the old behaviour before relative scoring was added. Make sure you have the
latest `app.py` with the `_relative_scores` function.

**EasyOCR download on first run**
→ EasyOCR downloads its models (~100MB) on first use. This is normal. It only happens once.
The system falls back to OpenCV ink detection if EasyOCR fails.

**Very slow analysis**
→ Normal for first run (EasyOCR model download). Subsequent runs are faster.
Model is loaded once at startup — do not restart the server between analyses.

**Port 5000 already in use**
→ Change the port in the last line of `app.py`:
```python
app.run(debug=True, host='0.0.0.0', port=5001)
```

**Training fails with "Not enough valid fake patches"**
→ Add more images to `datasets/edited/`. Minimum ~5 images needed, 15+ recommended.
