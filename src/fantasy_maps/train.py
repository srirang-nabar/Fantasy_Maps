"""Train one β-VAE cell (one beta, one seed) — plan.md Stage 2.

A single run trains one (beta, seed) combination and writes a checkpoint plus a
JSON sidecar (config, seed, torch version, loss history, split fingerprint) so the
headline cells are reproducible and CPU-loadable for the Tier-1 gate. The β-sweep
is the outer product of betas x seeds, driven by scripts/run_sweep.sh.

    uv run python -m fantasy_maps.train --config configs/train_smoke.yaml --smoke
    uv run python -m fantasy_maps.train --config configs/train_sweep.yaml --beta 4 --seed 0
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from fantasy_maps import audit, dataset, models

ROOT = audit.ROOT
CKPT_DIR = ROOT / "results" / "checkpoints"


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _run_epoch(model, loader, beta, device, opt=None, loss_type="vae", dataset_size=0):
    train = opt is not None
    model.train(train)
    tot = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
    for img, _factor, _seed in loader:
        img = img.to(device)
        with torch.set_grad_enabled(train):
            recon, mu, logvar, z = model(img)
            if loss_type == "tcvae":
                loss, recon_t, parts = models.tc_vae_loss(recon, img, mu, logvar, z, beta, dataset_size)
                kl_t = parts["mi"] + parts["tc"] + parts["dwkl"]
            else:
                loss, recon_t, kl_t = models.vae_loss(recon, img, mu, logvar, beta)
        if train:
            opt.zero_grad()
            loss.backward()
            opt.step()
        bs = img.size(0)
        tot["loss"] += loss.item() * bs
        tot["recon"] += recon_t.item() * bs
        tot["kl"] += kl_t.item() * bs
        tot["n"] += bs
    n = max(tot["n"], 1)
    return {k: tot[k] / n for k in ("loss", "recon", "kl")}


def train_cell(cfg: dict, beta: float, seed: int, smoke: bool, device: str = "cpu") -> Path:
    seed_everything(seed)
    gen_cfg = audit.load_config()
    splits = dataset.make_splits(gen_cfg)
    fp = dataset.split_fingerprint(splits)
    factor_table = dataset.load_factor_table(gen_cfg)

    train_seeds, val_seeds = splits["train"], splits["val"]
    if smoke:
        train_seeds = train_seeds[: cfg.get("train_limit", 400)]
        val_seeds = val_seeds[: cfg.get("val_limit", 100)]

    common = dict(img_size=cfg["img_size"], config=gen_cfg,
                  factor_table=factor_table, cache=cfg.get("cache", False))
    train_ds = dataset.MapDataset(train_seeds, **common)
    val_ds = dataset.MapDataset(val_seeds, **common)
    # num_workers=0 keeps the dataset's decoded-image cache in the main process so
    # it persists across epochs (worker processes would each hold a partial copy that
    # dies each epoch). With cache=true this makes epochs 2..N pure GPU compute.
    workers = cfg.get("num_workers", 0)
    persist = workers > 0
    train_ld = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                          num_workers=workers, persistent_workers=persist)
    val_ld = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=workers, persistent_workers=persist)

    loss_type = cfg.get("loss", "vae")
    n_train = len(train_seeds)
    model = models.ConvVAE(cfg["img_size"], cfg["latent_dim"], cfg["channels"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    history = []
    for epoch in range(cfg["epochs"]):
        t0 = time.time()
        tr = _run_epoch(model, train_ld, beta, device, opt, loss_type, n_train)
        va = _run_epoch(model, val_ld, beta, device, None, loss_type, n_train)
        history.append({"epoch": epoch, "train": tr, "val": va, "seconds": round(time.time() - t0, 1)})
        print(f"[{loss_type} beta={beta} seed={seed}] epoch {epoch}: "
              f"train_loss={tr['loss']:.1f} (recon {tr['recon']:.1f}, kl {tr['kl']:.2f}) "
              f"val_loss={va['loss']:.1f}  {history[-1]['seconds']}s")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    kind = "" if loss_type == "vae" else f"{loss_type}_"
    tag = f"{'smoke_' if smoke else ''}{kind}beta{beta}_seed{seed}_{cfg['img_size']}px"
    sidecar = {
        "tag": tag, "loss": loss_type, "beta": beta, "seed": seed, "smoke": smoke,
        "img_size": cfg["img_size"], "latent_dim": cfg["latent_dim"], "channels": cfg["channels"],
        "lr": cfg["lr"], "batch_size": cfg["batch_size"], "epochs": cfg["epochs"],
        "n_train": n_train, "n_val": len(val_seeds),
        "split_fingerprint": fp, "factors": models_factor_order(),
        "torch_version": torch.__version__, "history": history,
    }
    torch.save({"model_state": model.state_dict(), "sidecar": sidecar}, CKPT_DIR / f"{tag}.pt")
    (CKPT_DIR / f"{tag}.json").write_text(json.dumps(sidecar, indent=2))
    print(f"saved {CKPT_DIR / (tag + '.pt')}")
    return CKPT_DIR / f"{tag}.pt"


def models_factor_order():
    from fantasy_maps import factors
    return factors.GROUND_TRUTH


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--beta", type=float, default=None, help="override; else uses config betas[0]")
    ap.add_argument("--seed", type=int, default=None, help="override; else uses config seeds[0]")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    beta = args.beta if args.beta is not None else cfg["betas"][0]
    seed = args.seed if args.seed is not None else cfg["seeds"][0]
    train_cell(cfg, beta, seed, args.smoke, args.device)


if __name__ == "__main__":
    main()
