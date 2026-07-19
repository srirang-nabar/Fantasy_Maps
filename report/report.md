# This Fantasy Map Does Not Exist — does β actually disentangle?

A factor-labeled disentanglement benchmark built from a procedural fantasy-map
generator, used to test — by measurement, not inspection — whether a β-VAE's
disentanglement claim replicates. A data point in the Locatello et al. (2019)
impossibility debate.

## Setup

- **Data engine.** 33,000 maps generated headlessly from a *pinned offline snapshot*
  of Azgaar's Fantasy Map Generator (deterministic per seed; clean heightmap render,
  no labels/political layers), each logged with its true generative properties.
- **Factor oracle.** Five scale-free ground-truth factors — land fraction, mountain
  fraction, coastline raggedness, river density, lake count — derived from the
  in-page data model. The set was **redesigned before training** after the original
  failed its pre-registered pairwise-correlation ceiling (|ρ|≤0.8); the final set's
  worst pair is −0.74 (HYPOTHESES.md amendment 1).
- **Models.** Convolutional β-VAE, 16-dim latent, β ∈ {1,2,4,8} × 3 seeds, at 64px
  (pre-registered headline) and 128px (comparison). Same architecture across each
  sweep — only the KL weight changes. PCA-on-pixels at matched 16-dim is the baseline.
- **Metrics.** MIG and DCI on the 3,000-map held-out eval set. The estimators are
  **calibration-gated**: they score ~1 on synthetic perfectly-disentangled codes and
  ~0 on entangled/random codes (`gate_stage3`), so the reported numbers are trusted.

## Results (64px headline, Holm–Bonferroni, α=0.05)

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| **H1** MIG rises with β | **Not supported** | Spearman ρ(β,MIG)=+0.45, p_holm=0.069 |
| **H2** reconstruction error rises with β | **Supported** | ρ(β,recon)=+0.97, p<0.001 (0.0043→0.0066) |
| **H3** β-VAE beats PCA on MIG | **Supported** | best 0.095 vs PCA 0.043, p_holm=0.029 |

- **Absolute disentanglement is low everywhere** (MIG ≈ 0.06–0.13). Even the best cell
  only weakly recovers the factor oracle.
- **MIG and DCI disagree:** on DCI-disentanglement the linear PCA baseline (≈0.16) beats
  *every* β-VAE cell (≈0.09–0.13), while the β-VAE wins on MIG.
- **At 128px the β-effect vanishes:** MIG flat in β (ρ=−0.07), the rate–distortion trend
  weakens (H2 not supported), H3 still holds.

## Reading

A **replication-skeptical** result in the spirit of Locatello et al. Turning β up does
not reliably buy disentanglement — the rate–distortion mechanism works (H2) but does not
translate into a factor-aligned latent code, the effect is swamped by across-seed variance,
and two standard metrics disagree on whether the deep model beats a linear one. That
seed- and metric-dependence is precisely why the impossibility result argues unsupervised
disentanglement cannot be guaranteed without inductive biases or supervision. The
`report/demo.html` slider makes this tangible: one latent reads as a rough "mountain"
control, but several latents redundantly track land fraction and every slider nudges
multiple properties at once.

## Follow-up: β-TCVAE

To test whether the low disentanglement is just the wrong VAE variant, we ran a β-TCVAE
sweep (same grid, 64px, 60 epochs) — it penalizes the *total-correlation* term of the KL
directly, the component most tied to disentanglement. It gives **no significant gain**:
best-β MIG 0.089 vs the β-VAE's 0.095 (t-test p=0.72), tracking the β-VAE almost exactly
across the sweep. A method purpose-built to disentangle hits the same ~0.09 ceiling. Two
different objectives, same result — the strongest Locatello-consistent statement this
benchmark supports: the ceiling is a property of the unsupervised setting and the data, not
of the particular model. (`results/tcvae_comparison.png`, C7.)

## Limitations

- Factors are generator-specific derived properties, not canonical independent generative
  factors; the correlation ceiling bounds but does not eliminate their coupling.
- Modest resolution (64/128px) and a small conv VAE; β-VAE reconstructions are blurry by
  design — this project measures disentanglement, not image quality.
- Three seeds per cell bounds the power of the trend tests (H1 at 64px lands just above α).
- MIG/DCI are the two metrics reported; other estimators (SAP, FactorVAE score) may rank
  cells differently — itself part of the finding.

## Reproducing

Data engine → gates → training → metrics are scripted and gated end-to-end; committed
CPU-loadable checkpoints reproduce the metrics in CLAIMS.md. See REPRODUCING.md.
