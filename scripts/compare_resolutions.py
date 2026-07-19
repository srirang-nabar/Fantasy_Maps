"""Reconstruction sharpness comparison across VAE resolutions (Stage 4 / sharpness story).

For a handful of held-out eval maps, shows the ground-truth map next to each VAE's
reconstruction (64px / 128px / 256px, β=1), all displayed at the same size so the
detail each model actually captures is directly comparable. Answers: how much does
scaling the plain VAE to 256px sharpen things — enough, or is a VAE-GAN needed?

    uv run python scripts/compare_resolutions.py
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from fantasy_maps import audit, dataset, models

ROOT = audit.ROOT
CKPT = ROOT / "results" / "checkpoints"
TAGS = [("64px (demo)", "beta1.0_seed0_64px"), ("256px (new)", "beta1.0_seed0_256px")]
N = 5
DISP = 256  # display size for every tile (upsampled so detail is comparable)


@torch.no_grad()
def _recon(tag, seeds, table, cfg):
    ck = torch.load(CKPT / f"{tag}.pt", map_location="cpu", weights_only=False)
    sc = ck["sidecar"]
    m = models.ConvVAE(sc["img_size"], sc["latent_dim"], sc["channels"])
    m.load_state_dict(ck["model_state"]); m.eval()
    ds = dataset.MapDataset(seeds, img_size=sc["img_size"], config=cfg, factor_table=table)
    x = torch.stack([ds[i][0] for i in range(len(seeds))])
    r, _mu, _lv, _z = m(x)
    return r.permute(0, 2, 3, 1).numpy(), sc["img_size"]


def _up(img):  # HxWx3 float -> DISP x DISP uint8, bilinear so blur is honest
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((DISP, DISP), Image.BILINEAR))


def main():
    cfg = audit.load_config()
    table = dataset.load_factor_table(cfg)
    splits = dataset.make_splits(cfg)
    seeds = splits["eval"][:N]

    # originals at native 256
    orig_ds = dataset.MapDataset(seeds, img_size=256, config=cfg, factor_table=table)
    originals = [orig_ds[i][0].permute(1, 2, 0).numpy() for i in range(N)]
    recons = {lbl: _recon(tag, seeds, table, cfg)[0] for lbl, tag in TAGS}

    rows = [("ground truth", originals)] + [(lbl, recons[lbl]) for lbl, _ in TAGS]
    fig, axes = plt.subplots(len(rows), N, figsize=(N * 2.0, len(rows) * 2.0))
    for r, (lbl, imgs) in enumerate(rows):
        for c in range(N):
            axes[r, c].imshow(_up(imgs[c])); axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if c == 0:
                axes[r, c].set_ylabel(lbl, fontsize=11, rotation=90, va="center")
            if r == 0:
                axes[r, c].set_title(f"seed {seeds[c]}", fontsize=8)
    fig.suptitle("Plain-VAE reconstruction sharpness vs resolution (β=1)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = ROOT / "results" / "resolution_comparison.png"
    fig.savefig(out, dpi=130); print(f"wrote {out}")


if __name__ == "__main__":
    main()
