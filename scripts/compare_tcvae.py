"""Compare β-TCVAE vs β-VAE on disentanglement (plan.md Stage 4 stretch).

Reads results/stage3_metrics.json (which after re-evaluation holds both model kinds)
and answers: does penalizing total correlation raise MIG where cranking β on the
plain KL did not? Per β and overall, with a t-test of the best-β cells (across seeds)
and a comparison plot. 64px, the resolution both sweeps share.

    uv run python scripts/compare_tcvae.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
CB, CT, MUTED, INK, GRID = "#2a78d6", "#eb6834", "#52514e", "#0b0b0b", "#e6e6e3"


def _by_beta(rows, model, img=64, key="mig"):
    d = defaultdict(list)
    for r in rows:
        if r["model"] == model and r["img_size"] == img:
            d[r["beta"]].append(r[key])
    return {b: np.array(v) for b, v in sorted(d.items())}


def main():
    data = json.loads((ROOT / "results" / "stage3_metrics.json").read_text())
    rows = data["results"]
    vae = _by_beta(rows, "bvae"); tc = _by_beta(rows, "tcvae")
    pca = next((r["mig"] for r in rows if r["model"] == "pca" and r["img_size"] == 64), None)
    if not tc:
        print("no tcvae rows in stage3_metrics.json — run evaluate first"); return

    betas = sorted(set(vae) & set(tc))
    print(f"{'beta':>6} {'bVAE MIG':>12} {'TCVAE MIG':>12}   winner")
    for b in betas:
        v, t = vae[b].mean(), tc[b].mean()
        print(f"{b:>6g} {v:>12.3f} {t:>12.3f}   {'TCVAE' if t > v else 'bVAE'}")

    best_v_beta = max(vae, key=lambda b: vae[b].mean())
    best_t_beta = max(tc, key=lambda b: tc[b].mean())
    bv, bt = vae[best_v_beta], tc[best_t_beta]
    tstat, p = stats.ttest_ind(bt, bv, alternative="greater", equal_var=False)
    verdict = ("TCVAE > β-VAE" if p < 0.05 else "no significant TCVAE gain")
    print(f"\nbest β-VAE  MIG={bv.mean():.3f} (β={best_v_beta:g})")
    print(f"best TCVAE  MIG={bt.mean():.3f} (β={best_t_beta:g})")
    print(f"PCA baseline MIG={pca:.3f}")
    print(f"t-test best TCVAE > best β-VAE: p={p:.4f}  ->  {verdict}")

    out = {"betas": betas,
           "bvae_mig": {b: float(vae[b].mean()) for b in betas},
           "tcvae_mig": {b: float(tc[b].mean()) for b in betas},
           "best_bvae": {"beta": float(best_v_beta), "mig": float(bv.mean())},
           "best_tcvae": {"beta": float(best_t_beta), "mig": float(bt.mean())},
           "pca_mig": float(pca), "t_p": float(p), "verdict": verdict}
    (ROOT / "results" / "tcvae_comparison.json").write_text(json.dumps(out, indent=2))

    # comparison plot
    fig, ax = plt.subplots(figsize=(6, 4.2), facecolor="white")
    for d, c, lbl in ((vae, CB, "β-VAE"), (tc, CT, "β-TCVAE")):
        x = np.log2(betas)
        m = np.array([d[b].mean() for b in betas]); s = np.array([d[b].std() for b in betas])
        ax.fill_between(x, m - s, m + s, color=c, alpha=0.15, lw=0)
        ax.plot(x, m, "-o", color=c, lw=2, ms=7, label=lbl)
    ax.axhline(pca, color=MUTED, ls=":", lw=1.5); ax.annotate("PCA", (0, pca), (0.05, pca), color=MUTED, fontsize=8)
    ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(["1", "2", "4", "8"])
    ax.set_xlabel("β", color=INK); ax.set_ylabel("MIG (higher = more disentangled)", color=INK)
    ax.set_title("β-TCVAE vs β-VAE — does penalizing total correlation help? (64px)",
                 color=INK, fontsize=10, loc="left")
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "tcvae_comparison.png", dpi=150, facecolor="white")
    print("wrote results/tcvae_comparison.{json,png}")


if __name__ == "__main__":
    main()
