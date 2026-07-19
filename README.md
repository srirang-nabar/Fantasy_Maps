# Fantasy Map Generation & Disentanglement (β-VAE, Representation Learning)

*a.k.a. This Fantasy Map Does Not Exist*

Do β-VAEs actually disentangle? A factor-labeled benchmark built from a
procedural fantasy-map generator: every training image comes with its true
generative factors, so disentanglement (MIG/DCI) is *measured*, not eyeballed
— a data point in the Locatello et al. (2019) impossibility debate, wearing a
wizard hat.

## Headline numbers

> Filled as stage gates pass; every number will have a CLAIMS.md row and an
> asserting notebook.

| # | Question | Result (64px headline) |
| - | -------- | ------ |
| 1 | Does MIG rise with β? | **No** — weak trend only (Spearman ρ=+0.45, p_holm=0.069) |
| 2 | Rate–distortion tradeoff measurable? | **Yes** — recon error rises with β (ρ=+0.97, p_holm<0.001) |
| 3 | Does the β-VAE beat PCA on MIG? | **Yes** — 0.095 vs 0.043 (p_holm=0.029) |

**Headline:** a replication-skeptical result in the Locatello (2019) spirit —
cranking β up does *not* reliably buy disentanglement (absolute MIG stays ≈0.1),
even though β does its mechanical job on the rate–distortion frontier, and the
β-VAE only modestly outperforms a linear PCA baseline. A **β-TCVAE** follow-up
(which penalizes total correlation directly) gives *no* significant gain over
β-VAE (0.089 vs 0.095, p=0.72) — two different unsupervised objectives hit the
same ceiling. Full numbers + the Locatello reading in [CLAIMS.md](CLAIMS.md).

Status: complete through Stage 3. Data engine verified; 33k maps generated +
gated; β-sweep trained at 64px & 128px (RTX 4090); MIG/DCI measured with a
calibration-gated estimator. See HYPOTHESES.md for the pre-registered protocol.
