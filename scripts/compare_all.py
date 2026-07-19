"""Full sharpness verdict: VAE (64/256px) vs VAE-GAN (Stage 4 / sharpness story).

Produces three figures in results/:
  reconstruction_all.png — same eval maps: ground truth vs each model's reconstruction
  gan_samples.png        — novel maps sampled from the VAE-GAN prior (z~N(0,I))
  gan_traversal.png      — walking VAE-GAN latents (does the latent still do anything?)
Plus an objective sharpness score (variance of the Laplacian; higher = sharper edges).

    uv run python scripts/compare_all.py
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

from fantasy_maps import audit, dataset, models

ROOT = audit.ROOT
CKPT = ROOT / "results" / "checkpoints"
RECON_MODELS = [("64px VAE", "beta1.0_seed0_64px"), ("256px VAE", "beta1.0_seed0_256px"),
                ("VAE-GAN 128px", "vaegan_seed0_128px")]
N, DISP = 5, 256


def _load(tag):
    ck = torch.load(CKPT / f"{tag}.pt", map_location="cpu", weights_only=False)
    sc = ck["sidecar"]
    m = models.ConvVAE(sc["img_size"], sc["latent_dim"], sc["channels"])
    m.load_state_dict(ck["model_state"]); m.eval()
    return m, sc


def _disp(img):
    im = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((DISP, DISP), Image.BILINEAR))


def _sharpness(img):  # variance of Laplacian on grayscale
    g = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).convert("L")
    lap = np.asarray(g.filter(ImageFilter.FIND_EDGES), dtype=np.float64)
    return float(lap.var())


@torch.no_grad()
def main():
    cfg = audit.load_config()
    table = dataset.load_factor_table(cfg)
    seeds = dataset.make_splits(cfg)["eval"][:N]
    orig = [dataset.MapDataset(seeds, img_size=256, config=cfg, factor_table=table)[i][0].permute(1, 2, 0).numpy()
            for i in range(N)]

    # ---- reconstruction comparison ----
    rows = [("ground truth", orig)]
    sharp = {"ground truth": np.mean([_sharpness(o) for o in orig])}
    gan_model = None
    for lbl, tag in RECON_MODELS:
        m, sc = _load(tag)
        if sc.get("loss") == "vaegan":
            gan_model = m
        ds = dataset.MapDataset(seeds, img_size=sc["img_size"], config=cfg, factor_table=table)
        x = torch.stack([ds[i][0] for i in range(N)])
        recon = m(x)[0].permute(0, 2, 3, 1).numpy()
        rows.append((lbl, list(recon)))
        sharp[lbl] = np.mean([_sharpness(r) for r in recon])

    fig, ax = plt.subplots(len(rows), N, figsize=(N * 2, len(rows) * 2))
    for r, (lbl, imgs) in enumerate(rows):
        for c in range(N):
            ax[r, c].imshow(_disp(imgs[c])); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if c == 0:
                ax[r, c].set_ylabel(f"{lbl}\n(sharp {sharp[lbl]:.0f})", fontsize=9, rotation=90, va="center")
            if r == 0:
                ax[r, c].set_title(f"seed {seeds[c]}", fontsize=8)
    fig.suptitle("Reconstruction: VAE vs VAE-GAN (sharpness = Laplacian variance, higher=sharper)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(ROOT / "results" / "reconstruction_all.png", dpi=130)
    print("wrote results/reconstruction_all.png")
    print("sharpness:", {k: round(v) for k, v in sharp.items()})

    # ---- GAN prior samples ----
    torch.manual_seed(0)
    samp = gan_model.decode(torch.randn(N, gan_model.latent_dim)).permute(0, 2, 3, 1).numpy()
    fig, ax = plt.subplots(1, N, figsize=(N * 2, 2.2))
    for c in range(N):
        ax[c].imshow(_disp(samp[c])); ax[c].axis("off")
    fig.suptitle("VAE-GAN novel samples (z ~ N(0,I) → decode)", fontsize=11)
    fig.tight_layout(); fig.savefig(ROOT / "results" / "gan_samples.png", dpi=130)
    print("wrote results/gan_samples.png")

    # ---- GAN latent traversal (is the latent alive?) ----
    base = torch.zeros(1, gan_model.latent_dim)
    dims = list(range(6))
    steps = np.linspace(-2.5, 2.5, 7)
    fig, ax = plt.subplots(len(dims), len(steps), figsize=(len(steps) * 1.4, len(dims) * 1.4))
    for r, d in enumerate(dims):
        for c, t in enumerate(steps):
            z = base.clone(); z[0, d] = float(t)
            img = gan_model.decode(z)[0].permute(1, 2, 0).numpy()
            ax[r, c].imshow(_disp(img)); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if c == 0:
                ax[r, c].set_ylabel(f"z{d}", fontsize=8, rotation=0, ha="right", va="center")
    fig.suptitle("VAE-GAN latent traversal (flat rows = latent collapsed)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(ROOT / "results" / "gan_traversal.png", dpi=120)
    print("wrote results/gan_traversal.png")


if __name__ == "__main__":
    main()
