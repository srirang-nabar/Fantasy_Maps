"""Headline figure (plan.md Stage 3): disentanglement and reconstruction vs β.

Left panel is the headline — MIG vs β with across-seed bands, 64px and 128px, and
the PCA-on-pixels baseline as a reference line (H1, H3). Right panel is the
rate-distortion story — reconstruction error vs β (H2). One generating script,
one figure: results/headline.png.

    uv run python scripts/plot_headline.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# CVD-safe categorical pair (validated dataviz palette): blue / orange, + neutral ink.
C64, C128 = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e6e3"


def _agg(rows, img_size, key):
    by_beta = defaultdict(list)
    for r in rows:
        if r["model"] == "bvae" and r["img_size"] == img_size:
            by_beta[r["beta"]].append(r[key])
    betas = sorted(by_beta)
    mean = np.array([np.mean(by_beta[b]) for b in betas])
    std = np.array([np.std(by_beta[b]) for b in betas])
    return np.array(betas), mean, std


def _pca(rows, img_size, key):
    r = next((r for r in rows if r["model"] == "pca" and r["img_size"] == img_size), None)
    return r[key] if r else None


def _panel(ax, rows, key, title, ylabel, show_pca=True):
    for img_size, c, lbl in ((64, C64, "64px"), (128, C128, "128px")):
        b, m, s = _agg(rows, img_size, key)
        x = np.log2(b)
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15, linewidth=0)
        ax.plot(x, m, "-o", color=c, lw=2, ms=7, label=lbl, zorder=3)
        if show_pca:
            p = _pca(rows, img_size, key)
            if p is not None:
                ax.axhline(p, color=c, ls=":", lw=1.5, alpha=0.7, zorder=1)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel("β", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"): ax.spines[sp].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def main():
    data = json.loads((ROOT / "results" / "stage3_metrics.json").read_text())
    rows = data["results"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), facecolor="white")

    _panel(a1, rows, "mig", "Disentanglement (MIG) vs β", "MIG  (higher = more disentangled)")
    # annotate the PCA reference once
    p64 = _pca(rows, 64, "mig")
    a1.annotate("PCA baseline", xy=(0, p64), xytext=(0.05, p64 - 0.012),
                color=MUTED, fontsize=8, va="top")
    a1.legend(frameon=False, loc="upper left", labelcolor=INK)

    _panel(a2, rows, "recon_mse", "Reconstruction error vs β", "per-pixel MSE  (lower = sharper)",
           show_pca=False)
    a2.legend(frameon=False, loc="upper left", labelcolor=INK)

    fig.suptitle("Does β actually disentangle? Fantasy-map factor oracle",
                 x=0.02, ha="left", color=INK, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / "results" / "headline.png"
    fig.savefig(out, dpi=150, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
