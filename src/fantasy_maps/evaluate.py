"""Score every trained checkpoint on the held-out eval set (plan.md Stage 3).

Encodes the 3,000 eval maps through each β-VAE (latent = posterior mean, so it's
deterministic), then computes MIG + DCI against the ground-truth factors and the
eval reconstruction error. The PCA baselines are scored the same way on their
linear codes. Results are written to results/stage3_metrics.json — the input to
the H1-H3 adjudication (scripts/adjudicate.py).

    uv run python -m fantasy_maps.evaluate
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from fantasy_maps import audit, baseline, dataset, factors, metrics, models

ROOT = audit.ROOT
CKPT_DIR = ROOT / "results" / "checkpoints"


def load_eval(img_size: int):
    """(images (N,3,H,W) float, factors (N,K), seeds) for the eval split at img_size."""
    cfg = audit.load_config()
    splits = dataset.make_splits(cfg)
    table = dataset.load_factor_table(cfg)
    seeds = splits["eval"]
    ds = dataset.MapDataset(seeds, img_size=img_size, config=cfg, factor_table=table)
    imgs = torch.stack([ds[i][0] for i in range(len(ds))])
    facs = np.array([table[s] for s in seeds], dtype=np.float64)
    return imgs, facs, seeds


@torch.no_grad()
def encode(model, imgs, batch=256, device="cpu"):
    model.eval().to(device)
    mus, recon_se, n_pix = [], 0.0, 0
    for i in range(0, len(imgs), batch):
        x = imgs[i:i + batch].to(device)
        recon, mu, _lv, _z = model(x)
        mus.append(mu.cpu().numpy())
        recon_se += torch.sum((recon - x) ** 2).item()
        n_pix += x.numel()
    return np.concatenate(mus), recon_se / n_pix  # codes, mean per-pixel MSE


def evaluate_checkpoint(pt: Path, imgs, facs) -> dict:
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    sc = ck["sidecar"]
    model = models.ConvVAE(sc["img_size"], sc["latent_dim"], sc["channels"])
    model.load_state_dict(ck["model_state"])
    codes, recon_mse = encode(model, imgs)
    m = metrics.mig(codes, facs)
    d = metrics.dci(codes, facs)
    kind = sc.get("loss", "vae")
    return {
        "tag": sc["tag"], "model": "bvae" if kind == "vae" else kind, "beta": sc["beta"], "seed": sc["seed"],
        "img_size": sc["img_size"], "latent_dim": sc["latent_dim"],
        "split_fingerprint": sc["split_fingerprint"],
        "mig": m["mig"], "mig_per_factor": m["per_factor"].tolist(),
        "dci_disentanglement": d["disentanglement"], "dci_completeness": d["completeness"],
        "dci_informativeness": d["informativeness"], "recon_mse": recon_mse,
    }


def evaluate_pca(npz: Path, seeds, facs, img_size: int) -> dict:
    codes = baseline.transform(npz, seeds, img_size)
    m = metrics.mig(codes, facs)
    d = metrics.dci(codes, facs)
    return {
        "tag": npz.stem, "model": "pca", "beta": None, "seed": 0, "img_size": img_size,
        "latent_dim": int(codes.shape[1]),
        "mig": m["mig"], "mig_per_factor": m["per_factor"].tolist(),
        "dci_disentanglement": d["disentanglement"], "dci_completeness": d["completeness"],
        "dci_informativeness": d["informativeness"], "recon_mse": None,
    }


def main() -> None:
    order = factors.GROUND_TRUTH
    results = []
    for img_size in (64, 128):
        print(f"=== loading eval set at {img_size}px ===")
        imgs, facs, seeds = load_eval(img_size)
        print(f"  {len(seeds)} eval maps, {facs.shape[1]} factors")
        for pt in sorted(CKPT_DIR.glob(f"*_{img_size}px.pt")):
            r = evaluate_checkpoint(pt, imgs, facs)
            results.append(r)
            print(f"  {r['tag']}: MIG={r['mig']:.3f} DCI-D={r['dci_disentanglement']:.3f} "
                  f"recon_mse={r['recon_mse']:.4f}")
        pca = CKPT_DIR / f"pca_16d_{img_size}px.npz"
        if pca.exists():
            r = evaluate_pca(pca, seeds, facs, img_size)
            results.append(r)
            print(f"  {r['tag']}: MIG={r['mig']:.3f} DCI-D={r['dci_disentanglement']:.3f} (baseline)")

    out = ROOT / "results" / "stage3_metrics.json"
    out.write_text(json.dumps({"factors": order, "results": results}, indent=2))
    print(f"\nwrote {out} ({len(results)} rows)")


if __name__ == "__main__":
    main()
