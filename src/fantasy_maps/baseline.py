"""PCA-on-pixels baseline (plan.md Stage 2, hypothesis H3).

The honest "does deep buy anything" control: a linear encoder at the same latent
dimensionality as the β-VAE. Stage 3 scores its codes on MIG exactly like a VAE
cell; if the best β-VAE cannot beat this, the deep model earns nothing on
disentanglement. Fit on the train split, saved for reuse on the held-out eval set.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.decomposition import PCA

from fantasy_maps import audit, dataset

ROOT = audit.ROOT
CKPT_DIR = ROOT / "results" / "checkpoints"


def load_flat_images(seeds, img_size: int) -> np.ndarray:
    maps_dir = ROOT / "data" / "raw" / "maps"
    rows = []
    for s in seeds:
        with Image.open(maps_dir / f"{s}.png") as im:
            im = im.convert("RGB")
            if im.size != (img_size, img_size):
                im = im.resize((img_size, img_size), Image.LANCZOS)
            rows.append(np.asarray(im, dtype=np.float32).ravel() / 255.0)
    return np.stack(rows)


def fit_pca(img_size: int = 64, n_components: int = 16, limit: int | None = None, seed: int = 0):
    """Fit PCA on the train split; returns (pca, fingerprint). Saves to results/checkpoints."""
    gen_cfg = audit.load_config()
    splits = dataset.make_splits(gen_cfg)
    fp = dataset.split_fingerprint(splits)
    train = splits["train"]
    if limit:
        rng = np.random.default_rng(seed)
        train = sorted(rng.choice(train, size=min(limit, len(train)), replace=False).tolist())
    X = load_flat_images(train, img_size)
    pca = PCA(n_components=n_components, random_state=seed).fit(X)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(CKPT_DIR / f"pca_{n_components}d_{img_size}px.npz",
             components=pca.components_, mean=pca.mean_,
             explained_variance_ratio=pca.explained_variance_ratio_,
             img_size=img_size, n_components=n_components, split_fingerprint=fp)
    return pca, fp


def transform(pca_npz: Path, seeds, img_size: int) -> np.ndarray:
    d = np.load(pca_npz)
    X = load_flat_images(seeds, img_size)
    return (X - d["mean"]) @ d["components"].T
