# Checkpoint provenance — Stage 2 β-VAE sweep

Trained 2026-07-18. These are the committed, CPU-loadable model artifacts; each
`.pt` has a matching `.json` sidecar with its full config, seed, torch version,
per-epoch loss history, and split fingerprint. `MANIFEST.sha256` hashes every file.

## What's here

| Set | Files | Config | Grid |
| --- | --- | --- | --- |
| 64px (headline) | `beta{1,2,4,8}.0_seed{0,1,2}_64px.pt` | [configs/train_sweep.yaml](../../configs/train_sweep.yaml) | 4 β × 3 seeds = 12 |
| 128px (sharpness stretch) | `beta{1,2,4,8}.0_seed{0,1,2}_128px.pt` | [configs/train_sweep_128.yaml](../../configs/train_sweep_128.yaml) | 4 β × 3 seeds = 12 |
| 64px β-TCVAE (C7 follow-up) | `tcvae_beta{1,2,4,8}.0_seed{0,1,2}_64px.pt` | [configs/train_tcvae_64.yaml](../../configs/train_tcvae_64.yaml) | 4 β × 3 seeds = 12, 60 epochs |
| PCA baselines (H3) | `pca_16d_{64,128}px.npz` | — | linear, matched 16-dim |

All cells: 30 epochs, 16-dim latent, Adam lr 1e-3, batch 128. Same architecture
across each resolution — only β (the KL weight) varies — so disentanglement
differences are attributable to β, not capacity.

## Environment

- **Hardware:** 1× NVIDIA RTX 4090 (24 GB), rented on Vast.ai (NGC `nvcr.io/nvidia/pytorch:26.01-py3`).
- **torch (training):** `2.10.0a0+a36e1d39eb.nv26.01` (CUDA). Checkpoints verified to load and
  run forward on CPU with the dev-box torch (2.13.0+cpu).
- **Data split fingerprint:** `7b9f7caefb7ee4e4` (identical across all 24 cells). Metrics in
  Stage 3 assert this fingerprint so a checkpoint is never scored against a mismatched split.

## Reproducing

The dataset is regenerated from the committed data engine (see [REPRODUCING.md](../../REPRODUCING.md));
retraining uses `scripts/run_sweep.sh` (64px) and `CFG=configs/train_sweep_128.yaml scripts/run_sweep.sh`
(128px). Exact-retrain bit-equality is not expected across GPU/torch versions; the reproducibility
contract is that these committed checkpoints load on CPU and yield the Stage 3 metrics recorded in CLAIMS.md.
