"""Dataset + fixed, fingerprinted splits for VAE training (plan.md Stage 2).

The β-VAE is trained unsupervised on the map images; the ground-truth factors
(fantasy_maps.factors) ride along only for logging and for the Stage 3 metrics on
the held-out eval set. Splits are derived deterministically from the seed id so
they are stable across machines and code-guarded by a fingerprint — a training
run records the fingerprint in its checkpoint sidecar, and metrics refuse to mix
checkpoints trained on different splits.

  eval : seeds >= eval_seed_start        (held out for MIG/DCI; never trained on)
  val  : ~val_fraction of the rest, chosen by a stable hash of the seed
  train: everything else
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from fantasy_maps import audit, factors

ROOT = audit.ROOT


def _hash_bucket(seed: int, mod: int) -> int:
    return int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:8], "big") % mod


def make_splits(config: dict | None = None, val_fraction: float = 0.1) -> dict[str, list[int]]:
    """Deterministic train/val/eval seed lists. Pure function of the config."""
    config = config or audit.load_config()
    start, end, eval_start = config["seed_start"], config["seed_end"], config["eval_seed_start"]
    val_mod = max(2, round(1 / val_fraction))
    train, val, evl = [], [], []
    for s in range(start, end + 1):
        if s >= eval_start:
            evl.append(s)
        elif _hash_bucket(s, val_mod) == 0:
            val.append(s)
        else:
            train.append(s)
    return {"train": train, "val": val, "eval": evl}


def split_fingerprint(splits: dict[str, list[int]]) -> str:
    """Stable hash of the split assignment — the code guard against silent drift."""
    payload = json.dumps({k: splits[k] for k in ("train", "val", "eval")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_factor_table(config: dict | None = None) -> dict[int, list[float]]:
    """seed -> ground-truth factor vector (fantasy_maps.factors.GROUND_TRUTH order)."""
    config = config or audit.load_config()
    records, _ = audit.load_records(ROOT / "data" / "raw", config["n_shards"])
    order = factors.GROUND_TRUTH
    table = {}
    for r in records:
        d = factors.derive(r["factors"])
        table[r["seed"]] = [d[f] for f in order]
    return table


class MapDataset(Dataset):
    """Maps → (image CxHxW in [0,1], factor vector, seed).

    Images are stored at 256px; they are downsampled to `img_size` at load with
    LANCZOS (consistent with generation-time preprocessing). Small enough that
    caching decoded tensors in RAM is optional; off by default.
    """

    def __init__(self, seeds, img_size: int, config: dict | None = None,
                 factor_table: dict[int, list[float]] | None = None, cache: bool = False):
        self.config = config or audit.load_config()
        self.seeds = list(seeds)
        self.img_size = img_size
        self.maps_dir = ROOT / "data" / "raw" / "maps"
        self.factor_table = factor_table if factor_table is not None else load_factor_table(self.config)
        self.cache: dict[int, torch.Tensor] | None = {} if cache else None

    def __len__(self) -> int:
        return len(self.seeds)

    def _load_image(self, seed: int) -> torch.Tensor:
        if self.cache is not None and seed in self.cache:
            return self.cache[seed]
        with Image.open(self.maps_dir / f"{seed}.png") as im:
            im = im.convert("RGB")
            if im.size != (self.img_size, self.img_size):
                im = im.resize((self.img_size, self.img_size), Image.LANCZOS)
            arr = np.asarray(im, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CxHxW in [0,1]
        if self.cache is not None:
            self.cache[seed] = t
        return t

    def __getitem__(self, idx: int):
        seed = self.seeds[idx]
        img = self._load_image(seed)
        factor = torch.tensor(self.factor_table[seed], dtype=torch.float32)
        return img, factor, seed
