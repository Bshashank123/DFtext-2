"""
Anomaly Detection Module — Supervised Binary Classifier
Replaces Isolation Forest (one-class, blind to fakes) with
Random Forest trained on BOTH real and fake patch examples.
"""

import numpy as np
import os
import pickle
import cv2
from scipy.stats import kurtosis, skew

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score

from pipeline.texture_features import extract_texture_features
from pipeline.noise_analysis import extract_noise_feature_vector
from pipeline.frequency_analysis import extract_frequency_feature_vector
from utils.patch_utils import normalize_patch_size


MODEL_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'classifier.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'scaler.pkl')
STATS_PATH  = os.path.join(os.path.dirname(__file__), '..', 'models', 'score_stats.pkl')


def _extra_discriminative_features(patch):
    p = patch.astype(np.float32)

    local_std = []
    for y in range(0, 64 - 8, 8):
        for x in range(0, 64 - 8, 8):
            local_std.append(float(np.std(p[y:y+8, x:x+8])))
    local_std = np.array(local_std)
    feat_local_std_mean = float(np.mean(local_std))
    feat_local_std_std  = float(np.std(local_std))
    feat_local_std_cv   = feat_local_std_std / (feat_local_std_mean + 1e-6)

    gx = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    feat_grad_mean = float(np.mean(grad_mag))
    feat_grad_std  = float(np.std(grad_mag))
    try:
        feat_grad_kurt = float(kurtosis(grad_mag.flatten()))
    except Exception:
        feat_grad_kurt = 0.0

    bg_mask = patch > 180
    if bg_mask.sum() > 20:
        bg = patch[bg_mask].astype(np.float32)
        feat_bg_var   = float(np.var(bg))
        feat_bg_range = float(bg.max() - bg.min())
        try:
            feat_bg_kurt = float(kurtosis(bg))
        except Exception:
            feat_bg_kurt = 0.0
    else:
        feat_bg_var = feat_bg_range = feat_bg_kurt = 0.0

    ink_mask = patch < 128
    if ink_mask.sum() > 20:
        ink = patch[ink_mask].astype(np.float32)
        feat_ink_var     = float(np.var(ink))
        feat_ink_density = float(ink_mask.sum()) / (64 * 64)
        try:
            feat_ink_skew = float(skew(ink))
        except Exception:
            feat_ink_skew = 0.0
    else:
        feat_ink_var = feat_ink_density = feat_ink_skew = 0.0

    edges = cv2.Canny(patch, 50, 150)
    feat_edge_density = float(edges.sum()) / (64 * 64 * 255)

    laplacian = cv2.Laplacian(patch, cv2.CV_64F)
    feat_laplacian_var = float(np.var(laplacian))

    return np.array([
        feat_local_std_mean, feat_local_std_std, feat_local_std_cv,
        feat_grad_mean, feat_grad_std, feat_grad_kurt,
        feat_bg_var, feat_bg_range, feat_bg_kurt,
        feat_ink_var, feat_ink_density, feat_ink_skew,
        feat_edge_density, feat_laplacian_var,
    ], dtype=np.float32)


def build_feature_vector(patch):
    normalized = normalize_patch_size(patch, target_size=(64, 64))
    texture = extract_texture_features(normalized)
    noise   = extract_noise_feature_vector(normalized)
    freq    = extract_frequency_feature_vector(normalized)
    extra   = _extra_discriminative_features(normalized)
    return np.concatenate([texture, noise, freq, extra]).astype(np.float32)


def build_feature_matrix(patches):
    vectors = []
    for patch in patches:
        try:
            vec = build_feature_vector(patch)
            if not np.any(np.isnan(vec)) and not np.any(np.isinf(vec)):
                vectors.append(vec)
        except Exception:
            continue
    if not vectors:
        return None
    return np.array(vectors, dtype=np.float32)


def train_model(real_patches, fake_patches, save=True):
    print(f"Extracting features: {len(real_patches)} real + {len(fake_patches)} fake patches...")

    X_real = build_feature_matrix(real_patches)
    X_fake = build_feature_matrix(fake_patches)

    if X_real is None or len(X_real) < 5:
        raise ValueError("Not enough valid real patches")
    if X_fake is None or len(X_fake) < 5:
        raise ValueError("Not enough valid fake patches — add images to datasets/edited/")

    n = min(len(X_real), len(X_fake))
    rng = np.random.RandomState(42)
    if len(X_real) > n:
        X_real = X_real[rng.choice(len(X_real), n, replace=False)]
    if len(X_fake) > n:
        X_fake = X_fake[rng.choice(len(X_fake), n, replace=False)]

    print(f"Balanced dataset: {len(X_real)} real, {len(X_fake)} fake")

    X = np.vstack([X_real, X_fake])
    y = np.array([0] * len(X_real) + [1] * len(X_fake))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model = CalibratedClassifierCV(rf, cv=3, method='isotonic')
    model.fit(X_scaled, y)

    probs = model.predict_proba(X_scaled)[:, 1]
    preds = (probs > 0.5).astype(int)
    print(f"\nTraining performance:")
    print(classification_report(y, preds, target_names=['Real', 'Fake']))
    try:
        print(f"ROC AUC: {roc_auc_score(y, probs):.3f}")
    except Exception:
        pass

    real_probs = probs[y == 0]
    fake_probs = probs[y == 1]
    score_stats = {
        'real_mean':  float(np.mean(real_probs)),
        'fake_mean':  float(np.mean(fake_probs)),
        'separation': float(np.mean(fake_probs) - np.mean(real_probs)),
        'threshold':  0.5,
        'mode':       'supervised'
    }
    print(f"\nReal avg score: {score_stats['real_mean']:.3f}  (want < 0.35)")
    print(f"Fake avg score: {score_stats['fake_mean']:.3f}  (want > 0.65)")
    print(f"Separation:     {score_stats['separation']:.3f}  (want > 0.30)")

    if save:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH,  'wb') as f: pickle.dump(model,  f)
        with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
        with open(STATS_PATH,  'wb') as f: pickle.dump(score_stats, f)
        print(f"Model saved.")

    return model, scaler, score_stats


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        return None, None, None
    try:
        with open(MODEL_PATH,  'rb') as f: model  = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
        stats = None
        if os.path.exists(STATS_PATH):
            with open(STATS_PATH, 'rb') as f: stats = pickle.load(f)
        return model, scaler, stats
    except Exception as e:
        print(f"Could not load model: {e}")
        return None, None, None


def score_patches(patch_list, model=None, scaler=None, score_stats=None):
    """
    Score each patch. Returns probability of being AI-generated [0,1].
    Fully defensive — never raises, always returns a list of floats.
    """
    if patch_list is None or len(patch_list) == 0:
        return []

    if model is None:
        model, scaler, score_stats = load_model()

    vectors, valid_indices = [], []
    for i, patch in enumerate(patch_list):
        try:
            if patch is None or patch.size == 0:
                continue
            vec = build_feature_vector(patch)
            if vec is None or vec.ndim != 1:
                continue
            if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                continue
            vectors.append(vec)
            valid_indices.append(i)
        except Exception as e:
            print(f"  patch {i} feature error: {e}")
            continue

    scores_out = [0.3] * len(patch_list)
    if not vectors:
        return scores_out

    try:
        X = np.array(vectors, dtype=np.float32)
        if X.ndim != 2 or X.shape[0] == 0:
            return scores_out
    except Exception:
        return scores_out

    if model is not None and scaler is not None:
        try:
            X_scaled = scaler.transform(X)
            proba = model.predict_proba(X_scaled)
            if proba is None or proba.ndim != 2 or proba.shape[1] < 2:
                raise ValueError("predict_proba returned unexpected shape")
            probs = proba[:, 1]
            for idx, prob in zip(valid_indices, probs):
                scores_out[idx] = float(np.clip(prob, 0.0, 1.0))
        except Exception as e:
            print(f"  model inference error: {e} — using heuristic")
            heuristic = _heuristic_anomaly_scores(X)
            for idx, s in zip(valid_indices, heuristic):
                scores_out[idx] = float(s)
    else:
        heuristic = _heuristic_anomaly_scores(X)
        for idx, s in zip(valid_indices, heuristic):
            scores_out[idx] = float(s)

    return scores_out


def _heuristic_anomaly_scores(X):
    if len(X) == 1:
        return np.array([0.3])
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-6
    dist = np.sqrt(np.sum(((X - mean) / std) ** 2, axis=1))
    d_norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)
    sig = 1.0 / (1.0 + np.exp(-4 * (d_norm - 0.5)))
    return np.clip(sig * 0.7 + 0.1, 0.1, 0.85)


