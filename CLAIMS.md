# Claims register

Every headline number, with its source. Metrics are MIG (Mutual Information Gap)
and DCI, computed on the 3,000-map held-out eval set (split fingerprint
`7b9f7caefb7ee4e4`), against the 5 ground-truth factors. The metric estimators
are calibration-gated (`tests/test_metrics.py`, `gate_stage3`) — they score ~1 on
synthetic perfectly-disentangled codes and ~0 on entangled/random codes, so the
numbers below are believed. β-sweep = {1,2,4,8} × 3 seeds. 64px is the
pre-registered headline; 128px is a comparison. Verdicts Holm–Bonferroni corrected,
α=0.05.

| Claim ID | Statement | Value | Notebook cell | Test | Source artifact |
| --- | --- | --- | --- | --- | --- |
| C1 (H1) | MIG does **not** rise significantly with β (64px). Weak positive trend only. | Spearman ρ(β,MIG)=+0.45, p_holm=0.069 → **not supported** | 03 cell "H1" | scripts/adjudicate.py | results/stage3_verdicts.json |
| C2 (H2) | Rate–distortion tradeoff is real: reconstruction error rises monotonically with β (64px). | Spearman ρ(β,recon)=+0.97, p_holm<0.001 → **supported**; recon_mse 0.0043→0.0066 over β 1→8 | 03 cell "H2" | scripts/adjudicate.py | results/stage3_verdicts.json |
| C3 (H3) | Best β-VAE beats the PCA-on-pixels baseline on MIG at matched 16-dim (64px). | best (β=8) MIG=0.095 vs PCA 0.043; t-test p_holm=0.029 → **supported** | 03 cell "H3" | scripts/adjudicate.py | results/stage3_metrics.json |
| C4 | Absolute disentanglement is low across the whole sweep (MIG ≈ 0.06–0.13). β-VAE only weakly recovers the factor oracle. | max MIG 0.13; mean ≈ 0.09 | 03 headline | fantasy_maps.evaluate | results/stage3_metrics.json |
| C5 | On DCI-disentanglement the linear PCA baseline scores *higher* than every β-VAE cell — the two metrics disagree. | PCA DCI-D≈0.16 vs β-VAE DCI-D≈0.09 | 03 cell "DCI" | fantasy_maps.evaluate | results/stage3_metrics.json |
| C6 | At 128px the β-effect vanishes: MIG is flat in β (H1 ρ=−0.07) and the rate–distortion trend weakens (H2 p_holm=0.12, not supported); H3 still holds (MIG 0.103 vs 0.043). | see results | 03 comparison | scripts/adjudicate.py | results/stage3_verdicts.json |
| C7 | β-TCVAE (penalizing total correlation directly) gives **no significant MIG gain** over plain β-VAE at 64px — the method built to disentangle does not beat the one that doesn't. | best TCVAE MIG=0.089 vs best β-VAE 0.095; t-test p=0.72 | tcvae compare | scripts/compare_tcvae.py | results/tcvae_comparison.json |

## Reading (Locatello et al. 2019)

The headline is a **replication-skeptical** result, consistent with the impossibility
finding: turning the β knob up does not reliably buy disentanglement (H1 fails at both
resolutions), even though β mechanically does its job on the rate–distortion frontier
(H2, at 64px). The β-VAE extracts *more* factor-aligned structure than a linear baseline
(H3) but at a low absolute level, and MIG vs DCI disagree on whether it beats PCA at all —
exactly the metric- and seed-dependence Locatello warns makes unsupervised disentanglement
claims fragile.

The β-TCVAE follow-up (C7) sharpens this. β-TCVAE penalizes *total correlation* directly —
the term most theoretically tied to disentanglement — yet it gives no significant MIG gain
over plain β-VAE (0.089 vs 0.095, p=0.72), tracking it almost exactly across the β-sweep.
So the low disentanglement is not an artifact of choosing the wrong VAE variant: a method
purpose-built to disentangle also cannot recover the factor oracle from this data
unsupervised. Two different objectives, same ceiling — the strongest form of the
Locatello-consistent conclusion this benchmark can give.
