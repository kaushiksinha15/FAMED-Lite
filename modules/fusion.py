from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
import numpy as np
import pandas as pd


def compute_reliability(df):
    """
    Compute a [0, 1] reliability score for a data modality DataFrame.
    Based on two quality signals:
      1. Missing Rate  — proportion of NaN values in the batch
      2. Noise (CoV)   — mean coefficient of variation across columns
                         (high CoV = noisy / unstable sensor readings)

    Formula:
        reliability = 1.0 - 0.5 * missing_rate - 0.5 * noise_penalty
        noise_penalty = tanh(mean_CoV)  → bounded [0, ~1]

    Returns a float in [0.01, 1.00]. Clipped to avoid zero-weight collapse.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return 0.5  # safe neutral default

    # Signal 1: missing rate (0 = no missing, 1 = all missing)
    missing_rate = float(df.isna().mean().mean())

    # Signal 2: mean coefficient of variation (noise proxy)
    col_means = df.mean(numeric_only=True).abs() + 1e-6
    col_stds  = df.std(numeric_only=True)
    cov       = (col_stds / col_means).mean()
    noise_penalty = float(np.tanh(cov)) * 0.5  # scaled to max 0.5

    reliability = 1.0 - (missing_rate * 0.5) - noise_penalty
    return float(np.clip(reliability, 0.01, 1.0))


def reliability_weights(reliabilities):
    """
    Convert a list of raw reliability scores into normalised fusion weights.
    w_i = rel_i / sum(rel)
    """
    r = np.array(reliabilities, dtype=float)
    total = r.sum()
    if total == 0:
        return np.ones(len(r)) / len(r)
    return r / total

def train_modality_encoder(X, y, name=""):
    """
    Train a single XGBoost model on one data modality.
    """
    model = XGBClassifier(
        n_estimators=100, 
        max_depth=4,
        eval_metric='logloss'
    )
    model.fit(X, y)
    return model

def get_meta_features(encoders, X_list, mask, alpha=0.5):
    """
    Generate the risk scores from individual encoders.
    mask is a list of booleans: True = modality present.

    For PRESENT modalities: use the encoder's predicted probability.
    For ABSENT modalities: use Confidence-Weighted Neutral Imputation (Patent Claim 1):
        neutral = 0.5 + alpha * (mean_confidence - 0.5)
    where mean_confidence is the average score across all PRESENT modalities.
    This means a missing modality is no longer a hard 0.5 — it is softly pulled
    toward the evidence already seen by the available modalities.
    alpha controls the strength of that pull (0 = flat 0.5, 1 = full pull).
    """
    # Safely get the number of samples from the first available matrix
    n_samples = 0
    for x in X_list:
        if x is not None:
            n_samples = len(x)
            break

    # --- Pass 1: Compute scores for all PRESENT modalities ---
    present_scores = []
    for i, (enc, X) in enumerate(zip(encoders, X_list)):
        if mask[i] and X is not None:
            present_scores.append(enc.predict_proba(X)[:, 1])

    # Compute mean confidence across all present modalities (per sample)
    if present_scores:
        mean_confidence = np.mean(np.column_stack(present_scores), axis=1)
    else:
        mean_confidence = np.full(n_samples, 0.5)

    # Confidence-weighted neutral value for missing modalities
    # neutral = 0.5 + alpha * (confidence - 0.5)
    weighted_neutral = 0.5 + alpha * (mean_confidence - 0.5)

    # --- Pass 2: Assemble final score matrix ---
    scores = []
    present_iter = iter(present_scores)
    for i, (enc, X) in enumerate(zip(encoders, X_list)):
        if mask[i] and X is not None:
            scores.append(next(present_iter))
        else:
            # Use confidence-weighted neutral instead of flat 0.5
            scores.append(weighted_neutral)

    return np.column_stack(scores)

class FamedMetaLearner(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, meta_X, y):
        # Fit logic bypassed. Retained for pipeline structural backwards compatibility.
        pass

    def predict_proba(self, meta_X, weights=None):
        """
        Compute final risk score as a weighted average of modality scores.

        weights: array-like of length == meta_X.shape[1]
            Reliability-derived weights. If None, falls back to equal weighting.
            Formula: final = w1*score1 + w2*score2 + ...
            where w_i = reliability_i / sum(reliabilities)  [self-adjusting]
        """
        if weights is None:
            # Equal weighting fallback
            return np.mean(meta_X, axis=1)
        w = np.array(weights, dtype=float)
        w = w / w.sum()  # normalise to ensure weights sum to 1
        return np.dot(meta_X, w)

    def predict(self, meta_X, weights=None):
        return np.where(self.predict_proba(meta_X, weights=weights) >= 0.5, 1, 0)
