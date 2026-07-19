"""Adjudicate H1-H3 from the Stage 3 metrics (plan.md Stage 3, HYPOTHESES.md).

Reads results/stage3_metrics.json and, per resolution:
  H1  MIG rises with beta         -> Spearman trend test (beta vs MIG), one-sided +
  H2  recon error rises with beta -> Spearman trend test (beta vs recon_mse), one-sided +
  H3  best beta-VAE beats PCA MIG -> one-sample t-test of the best-beta seeds vs the PCA scalar
The three primary p-values are Holm-Bonferroni corrected at alpha=0.05.
64px is the pre-registered headline; 128px is reported as a comparison.

    uv run python scripts/adjudicate.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
ALPHA = 0.05


def _holm(pvals: dict) -> dict:
    """Holm-Bonferroni. pvals: {name: p} -> {name: (p_adj, reject)}."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, prev = {}, 0.0
    for i, (name, p) in enumerate(items):
        p_adj = max(prev, min(1.0, (m - i) * p))
        out[name] = (p_adj, p_adj < ALPHA)
        prev = p_adj
    return out


def adjudicate(rows, img_size):
    bvae = [r for r in rows if r["model"] == "bvae" and r["img_size"] == img_size]
    pca = next((r for r in rows if r["model"] == "pca" and r["img_size"] == img_size), None)
    by_beta = defaultdict(list)
    for r in bvae:
        by_beta[r["beta"]].append(r)
    betas = sorted(by_beta)

    beta_arr = np.array([r["beta"] for r in bvae])
    mig_arr = np.array([r["mig"] for r in bvae])
    recon_arr = np.array([r["recon_mse"] for r in bvae])

    # H1: Spearman beta vs MIG, one-sided positive
    r1, p1_two = stats.spearmanr(beta_arr, mig_arr)
    p1 = p1_two / 2 if r1 > 0 else 1 - p1_two / 2
    # H2: Spearman beta vs recon, one-sided positive
    r2, p2_two = stats.spearmanr(beta_arr, recon_arr)
    p2 = p2_two / 2 if r2 > 0 else 1 - p2_two / 2
    # H3: best-beta MIG seeds vs PCA scalar, one-sided greater
    best_beta = max(betas, key=lambda b: np.mean([r["mig"] for r in by_beta[b]]))
    best_migs = np.array([r["mig"] for r in by_beta[best_beta]])
    if pca and len(best_migs) >= 2:
        t, p3 = stats.ttest_1samp(best_migs, pca["mig"], alternative="greater")
    else:
        p3 = float("nan")

    holm = _holm({"H1": p1, "H2": p2, "H3": p3})
    return {
        "img_size": img_size, "betas": betas,
        "mig_by_beta": {b: (float(np.mean([r["mig"] for r in by_beta[b]])),
                            float(np.std([r["mig"] for r in by_beta[b]]))) for b in betas},
        "recon_by_beta": {b: (float(np.mean([r["recon_mse"] for r in by_beta[b]])),
                              float(np.std([r["recon_mse"] for r in by_beta[b]]))) for b in betas},
        "dci_by_beta": {b: float(np.mean([r["dci_disentanglement"] for r in by_beta[b]])) for b in betas},
        "H1": {"spearman": float(r1), "p": float(p1), "p_holm": float(holm["H1"][0]), "reject": bool(holm["H1"][1])},
        "H2": {"spearman": float(r2), "p": float(p2), "p_holm": float(holm["H2"][0]), "reject": bool(holm["H2"][1])},
        "H3": {"best_beta": float(best_beta), "best_mig": float(np.mean(best_migs)),
               "pca_mig": float(pca["mig"]) if pca else None,
               "p": float(p3), "p_holm": float(holm["H3"][0]), "reject": bool(holm["H3"][1])},
    }


def _fmt(v):
    return f"{v[0]:.3f}±{v[1]:.3f}"


def main():
    data = json.loads((ROOT / "results" / "stage3_metrics.json").read_text())
    verdicts = {}
    for img_size in (64, 128):
        v = adjudicate(data["results"], img_size)
        verdicts[img_size] = v
        tag = "HEADLINE" if img_size == 64 else "comparison"
        print(f"\n===== {img_size}px ({tag}) =====")
        print("beta:      " + "  ".join(f"{b:>5g}" for b in v["betas"]))
        print("MIG:       " + "  ".join(f"{v['mig_by_beta'][b][0]:5.3f}" for b in v["betas"]))
        print("recon_mse: " + "  ".join(f"{v['recon_by_beta'][b][0]:5.3f}" for b in v["betas"]))
        print("DCI-D:     " + "  ".join(f"{v['dci_by_beta'][b]:5.3f}" for b in v["betas"]))
        for h in ("H1", "H2", "H3"):
            d = v[h]
            verdict = "SUPPORTED" if d["reject"] else "not supported"
            extra = (f"spearman={d['spearman']:+.3f}" if "spearman" in d
                     else f"best(beta={d['best_beta']:g}) MIG={d['best_mig']:.3f} vs PCA {d['pca_mig']:.3f}")
            print(f"  {h}: {verdict:14s} p={d['p']:.4f} p_holm={d['p_holm']:.4f}  ({extra})")

    (ROOT / "results" / "stage3_verdicts.json").write_text(json.dumps(verdicts, indent=2))
    print("\nwrote results/stage3_verdicts.json")


if __name__ == "__main__":
    main()
