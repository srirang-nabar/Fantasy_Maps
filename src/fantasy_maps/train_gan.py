"""Train a VAE-GAN for sharp map generation (Stage 4 / sharpness story).

Same encoder→latent→decoder as the β-VAE (so the disentanglement metrics and the
slider demo still apply), but the decoder is trained with an L1 + adversarial
(PatchGAN) objective instead of pixel-MSE — the adversarial term is what produces
sharp edges. Alternating hinge updates: the discriminator learns real-vs-fake
(reconstructions and prior samples), the generator learns to reconstruct (L1),
keep a usable latent (KL), and fool the discriminator.

Saves a checkpoint whose sidecar (loss="vaegan") is drop-in for fantasy_maps.evaluate.

    uv run python -m fantasy_maps.train_gan --config configs/train_vaegan_128.yaml
    uv run python -m fantasy_maps.train_gan --config configs/train_vaegan_128.yaml --smoke
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from fantasy_maps import audit, dataset, factors, models

ROOT = audit.ROOT
CKPT_DIR = ROOT / "results" / "checkpoints"


def seed_everything(seed):
    np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def _val_l1(vae, loader, device):
    vae.eval()
    tot, n = 0.0, 0
    for img, _f, _s in loader:
        img = img.to(device)
        recon, _mu, _lv, _z = vae(img)
        tot += F.l1_loss(recon, img, reduction="sum").item(); n += img.numel()
    return tot / max(n, 1)


def train_gan(cfg, seed, smoke, device="cpu"):
    seed_everything(seed)
    gen_cfg = audit.load_config()
    splits = dataset.make_splits(gen_cfg)
    fp = dataset.split_fingerprint(splits)
    table = dataset.load_factor_table(gen_cfg)

    train_seeds, val_seeds = splits["train"], splits["val"]
    tl = cfg.get("train_limit", 400 if smoke else None)
    vl = cfg.get("val_limit", 100 if smoke else None)
    if tl:
        train_seeds = train_seeds[:tl]
    if vl:
        val_seeds = val_seeds[:vl]

    common = dict(img_size=cfg["img_size"], config=gen_cfg, factor_table=table, cache=cfg.get("cache", False))
    workers = cfg.get("num_workers", 0)
    persist = workers > 0
    train_ld = DataLoader(dataset.MapDataset(train_seeds, **common), batch_size=cfg["batch_size"],
                          shuffle=True, num_workers=workers, persistent_workers=persist, drop_last=True)
    val_ld = DataLoader(dataset.MapDataset(val_seeds, **common), batch_size=cfg["batch_size"],
                        shuffle=False, num_workers=workers, persistent_workers=persist)

    vae = models.ConvVAE(cfg["img_size"], cfg["latent_dim"], cfg["channels"]).to(device)
    disc = models.PatchDiscriminator(cfg.get("disc_channels", 64), cfg.get("disc_layers", 3)).to(device)
    opt_g = torch.optim.Adam(vae.parameters(), lr=cfg["lr"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=cfg.get("disc_lr", cfg["lr"]), betas=(0.5, 0.999))
    l1w, advw, klw = cfg["lambda_l1"], cfg["lambda_adv"], cfg["beta_kl"]
    free_bits = cfg.get("free_bits", 0.0)       # per-dim KL floor: latent can't collapse below this
    kl_warm = max(cfg.get("kl_warmup", 1), 1)
    adv_start = cfg.get("adv_start", 0)         # pure autoencoder before this epoch
    adv_warm = max(cfg.get("adv_warmup", 1), 1)

    history = []
    for epoch in range(cfg["epochs"]):
        vae.train(); disc.train()
        # anti-collapse schedule: learn a real latent (L1 + free-bits KL) first, then ramp
        # in the KL regularizer and the adversarial sharpener so it can't steamroll the encoder.
        kl_w = klw * min(1.0, (epoch + 1) / kl_warm)
        adv_w = 0.0 if epoch < adv_start else advw * min(1.0, (epoch - adv_start + 1) / adv_warm)
        t0 = time.time()
        agg = {"d": 0.0, "g_adv": 0.0, "l1": 0.0, "kl": 0.0, "n": 0}
        for img, _f, _s in train_ld:
            img = img.to(device)
            # ---- discriminator (only once the adversarial phase is on) ----
            d_val = 0.0
            if adv_w > 0:
                with torch.no_grad():
                    recon, mu, _lv, _z = vae(img)
                    samp = vae.decode(torch.randn_like(mu))
                loss_d = models.d_hinge_loss(disc(img), disc(recon), disc(samp))
                opt_d.zero_grad(); loss_d.backward(); opt_d.step()
                d_val = loss_d.item()
            # ---- generator (encoder + decoder) ----
            recon, mu, logvar, _z = vae(img)
            l1 = F.l1_loss(recon, img)
            kl_dims = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).mean(0)   # (D,) per-dim KL
            kl_free = torch.clamp(kl_dims, min=free_bits).sum()                  # free bits
            loss_g = l1w * l1 + kl_w * kl_free
            g_adv_val = 0.0
            if adv_w > 0:
                samp = vae.decode(torch.randn_like(mu))
                g_adv = models.g_hinge_loss(disc(recon), disc(samp))
                loss_g = loss_g + adv_w * g_adv
                g_adv_val = g_adv.item()
            opt_g.zero_grad(); loss_g.backward(); opt_g.step()

            bs = img.size(0)
            agg["d"] += d_val * bs; agg["g_adv"] += g_adv_val * bs
            agg["l1"] += l1.item() * bs; agg["kl"] += float(kl_dims.sum().item()) * bs; agg["n"] += bs
        n = max(agg["n"], 1)
        vl1 = _val_l1(vae, val_ld, device)
        rec = {"epoch": epoch, "d_loss": agg["d"] / n, "g_adv": agg["g_adv"] / n,
               "l1": agg["l1"] / n, "kl": agg["kl"] / n, "val_l1": vl1,
               "kl_w": round(kl_w, 4), "adv_w": round(adv_w, 3), "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"[vaegan seed={seed}] epoch {epoch}: d={rec['d_loss']:.3f} g_adv={rec['g_adv']:.3f} "
              f"l1={rec['l1']:.4f} kl={rec['kl']:.2f} adv_w={adv_w:.2f} val_l1={vl1:.4f}  {rec['seconds']}s")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{'smoke_' if smoke else ''}vaegan_seed{seed}_{cfg['img_size']}px"
    sidecar = {
        "tag": tag, "loss": "vaegan", "beta": cfg["beta_kl"], "seed": seed, "smoke": smoke,
        "img_size": cfg["img_size"], "latent_dim": cfg["latent_dim"], "channels": cfg["channels"],
        "lambda_l1": l1w, "lambda_adv": advw, "free_bits": free_bits, "kl_warmup": kl_warm,
        "adv_start": adv_start, "adv_warmup": adv_warm, "epochs": cfg["epochs"], "n_train": len(train_seeds),
        "split_fingerprint": fp, "factors": factors.GROUND_TRUTH,
        "torch_version": torch.__version__, "history": history,
    }
    torch.save({"model_state": vae.state_dict(), "disc_state": disc.state_dict(), "sidecar": sidecar},
               CKPT_DIR / f"{tag}.pt")
    (CKPT_DIR / f"{tag}.json").write_text(json.dumps(sidecar, indent=2))
    print(f"saved {CKPT_DIR / (tag + '.pt')}")
    return CKPT_DIR / f"{tag}.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = args.seed if args.seed is not None else cfg.get("seeds", [0])[0]
    train_gan(cfg, seed, args.smoke, args.device)


if __name__ == "__main__":
    main()
