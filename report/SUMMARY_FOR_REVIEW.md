# Reviewer summary — Fantasy Map Generation & Disentanglement (β-VAE, Representation Learning)

**One paragraph.** β-VAEs famously claim that turning up β makes latent dimensions "disentangle" into
human-meaningful factors — but measuring that requires knowing the *true* factors, which real datasets
never provide. This project manufactures the ground truth: 33,000 fantasy maps generated headlessly from
a pinned procedural generator, each logged with its true generative factors (land fraction, mountain
fraction, coastline raggedness, river density, lake count) — a factor-labeled benchmark in the dSprites
tradition, with a pre-registered protocol (dated HYPOTHESES.md with an amendments log), calibration-gated
metric estimators, and Holm-corrected verdicts. The dataset itself was gated on completeness, coverage,
and a factor-correlation ceiling — and when the original factor set failed that gate, it was redesigned
*before* training, on the record.

**Findings (64px pre-registered headline; Holm-corrected, α = 0.05):**

| # | Question | Result |
| - | -------- | ------ |
| H1 ✗ | Does disentanglement (MIG) rise with β? | **No** — weak trend only (Spearman ρ = 0.45, p_holm = 0.07); absolute MIG ≈ 0.06–0.13 across the whole sweep. A replication-skeptical result consistent with Locatello et al.'s (2019) impossibility argument |
| H2 ✓ | Does the rate–distortion tradeoff hold? | **Yes** — reconstruction error rises monotonically with β (ρ = 0.97, p_holm < 0.001): β does its mechanical job, it just doesn't buy disentanglement |
| H3 ✓ | Does the β-VAE beat a linear baseline? | **Yes, modestly** — best β-VAE MIG 0.095 vs PCA-on-pixels 0.043 (p_holm = 0.029) |
| — | β-TCVAE follow-up | The objective *designed* to disentangle (penalizing total correlation directly) gives **no significant gain** (0.089 vs 0.095, p = 0.72) — two unsupervised objectives, one ~0.09 ceiling |

**How to review quickly (~5 min):**

**Fastest path: `notebooks/00_review_walkthrough.ipynb`** — a single commented, pre-executed notebook
backing every resume point from committed result files, with the asserts inline.

1. Deeper dives: `notebooks/03_disentanglement.ipynb` (full metric analysis), `notebooks/01_data_engine.ipynb`
   (dataset gates), `report/report.md`, and the `report/demo*.html` latent-slider demo — whose soft,
   overlapping sliders make the *measured* low disentanglement tangible.
2. Optional: `uv sync --frozen && uv run pytest -q` (incl. the `gate_stage3` metric-calibration tests);
   checkpoints are CPU-loadable with a SHA-256 manifest and provenance.

**Scope honesty:** factors are generator-specific and the resolution modest (64/128px). Two disclosed
wrinkles strengthen rather than weaken the story: **the two standard metrics disagree** (on
DCI-disentanglement the PCA baseline *outscores* every β-VAE cell, opposite of MIG — metric choice
matters and is reported, not hidden), and **at 128px the β-effect vanishes entirely** (H1 ρ ≈ −0.07;
H2 no longer significant; H3 still holds) — the headline claims are stated at the pre-registered 64px
and the 128px comparison is reported alongside.
