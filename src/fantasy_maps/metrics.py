"""Disentanglement metrics (plan.md Stage 3): MIG and DCI.

The whole project rests on these numbers being trustworthy, so the metrics are
calibrated against synthetic cases with known answers (tests/test_metrics.py,
gate_stage3): a code that is a permuted copy of the factors must score MIG/DCI
near 1; a random rotation of the factors (fully entangled) must score near 0;
codes independent of the factors must score ~0. No measurement on a real
checkpoint is believed until that calibration passes.

- MIG (Chen et al. 2018): per factor, the gap between its top-1 and top-2 latent
  by mutual information, normalized by the factor's entropy; averaged over factors.
- DCI (Eastwood & Williams 2018): from per-(latent,factor) importances of random
  forests predicting each factor — Disentanglement (each latent codes one factor),
  Completeness (each factor is coded by one latent), Informativeness (predictability).

Everything is estimated on discretized variables via histogram mutual information.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mutual_info_score

_EPS = 1e-12


def _discretize(col: np.ndarray, n_bins: int) -> np.ndarray:
    """Continuous vector -> equal-width bin indices in [0, n_bins)."""
    lo, hi = float(col.min()), float(col.max())
    if hi <= lo:
        return np.zeros_like(col, dtype=np.int64)
    edges = np.linspace(lo, hi, n_bins + 1)[1:-1]
    return np.digitize(col, edges)


def _discretize_matrix(X: np.ndarray, n_bins: int) -> np.ndarray:
    return np.stack([_discretize(X[:, j], n_bins) for j in range(X.shape[1])], axis=1)


def _entropy(labels: np.ndarray) -> float:
    _, counts = np.unique(labels, return_counts=True)
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p + _EPS)))


def mutual_info_matrix(codes: np.ndarray, factors: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """(N,D) codes, (N,K) factors -> (D,K) mutual information (nats), histogram estimator."""
    zc = _discretize_matrix(np.asarray(codes, float), n_bins)
    fc = _discretize_matrix(np.asarray(factors, float), n_bins)
    D, K = zc.shape[1], fc.shape[1]
    mi = np.zeros((D, K))
    for j in range(D):
        for k in range(K):
            mi[j, k] = mutual_info_score(zc[:, j], fc[:, k])
    return mi


def mig(codes: np.ndarray, factors: np.ndarray, n_bins: int = 20) -> dict:
    """Mutual Information Gap. Returns {'mig': mean, 'per_factor': (K,), 'mi': (D,K)}."""
    factors = np.asarray(factors, float)
    fc = _discretize_matrix(factors, n_bins)
    mi = mutual_info_matrix(codes, factors, n_bins)
    K = fc.shape[1]
    per_factor = np.zeros(K)
    for k in range(K):
        hk = _entropy(fc[:, k])
        order = np.sort(mi[:, k])[::-1]  # descending MI over latents
        gap = order[0] - order[1] if order.size >= 2 else order[0]
        per_factor[k] = gap / (hk + _EPS)
    return {"mig": float(np.mean(per_factor)), "per_factor": per_factor, "mi": mi}


def dci(codes: np.ndarray, factors: np.ndarray, test_frac: float = 0.2,
        n_estimators: int = 50, seed: int = 0) -> dict:
    """DCI via random-forest importances. Returns disentanglement/completeness/informativeness."""
    codes = np.asarray(codes, float)
    factors = np.asarray(factors, float)
    N, D = codes.shape
    K = factors.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)
    cut = int(N * (1 - test_frac))
    tr, te = idx[:cut], idx[cut:]

    R = np.zeros((D, K))          # importance of latent j for factor k
    informativeness = np.zeros(K)
    for k in range(K):
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1)
        rf.fit(codes[tr], factors[tr, k])
        R[:, k] = rf.feature_importances_
        pred = rf.predict(codes[te])
        var = factors[te, k].var()
        informativeness[k] = 1.0 - np.mean((pred - factors[te, k]) ** 2) / (var + _EPS)  # R^2

    # Disentanglement: 1 - normalized entropy of each latent's importance over factors,
    # weighted by that latent's total importance.
    Pj = R / (R.sum(axis=1, keepdims=True) + _EPS)
    Hj = -np.sum(Pj * np.log(Pj + _EPS), axis=1) / (np.log(K) + _EPS)
    Dj = 1.0 - Hj
    rho = R.sum(axis=1) / (R.sum() + _EPS)
    disentanglement = float(np.sum(rho * Dj))

    # Completeness: 1 - normalized entropy of each factor's importance over latents.
    Pk = R / (R.sum(axis=0, keepdims=True) + _EPS)
    Hk = -np.sum(Pk * np.log(Pk + _EPS), axis=0) / (np.log(D) + _EPS)
    completeness = float(np.mean(1.0 - Hk))

    return {
        "disentanglement": disentanglement,
        "completeness": completeness,
        "informativeness": float(np.mean(informativeness)),
        "informativeness_per_factor": informativeness,
        "importance": R,
    }
