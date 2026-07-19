# Pre-registered hypotheses — Fantasy Map Generation & Disentanglement

Dated: 2026-07-17 (registered before any VAE training; data generation in progress).

## Primary (Holm–Bonferroni family, α = 0.05)

- **H1 (disentanglement rises with β):** MIG against the ground-truth factor set increases monotonically over β ∈ {1, 2, 4, 8} (trend test across ≥3 training seeds per β, held-out eval set).
- **H2 (rate–distortion tradeoff):** reconstruction error (per-pixel MSE, eval set) increases with β under the same protocol.
- **H3 (deep beats linear):** the best β-VAE exceeds a PCA-on-pixels baseline on MIG at matched latent dimensionality (16).

## Exploratory (descriptive only)

Factor-wise learnability; DCI decomposition; latent-dimension sweep; latent–factor assignment stability across seeds (the Locatello angle).

## Ground-truth factor set (design decision, Stage 0)

Derived measurable properties, not raw generator knobs (knobs interact): land_fraction, mountain_fraction_of_land, coast_fraction_of_land, mean_land_height, river_count, total_river_length, lake_count — computed in-page from the generator's data model at generation time. Pre-registered ceiling: pairwise |ρ| between chosen factors ≤ 0.8 on the generated dataset, else the factor set is redesigned *before* training (recorded here if so).

## Fixed generation config

Pinned snapshot (hash recorded per record), heightmap preset, UI stripped, 512×512 viewport (map size affects the cell grid — it is part of the config), downsampled to 256×256; seeds 1–33000 (last 3000 reserved as eval).

## Amendments

**Amendment 1 (2026-07-18, before any training) — ground-truth factor set redesigned.**
The originally registered set failed its own pre-registered correlation ceiling
(|ρ| ≤ 0.8) on the generated dataset (measured on 32,101 maps; three pairs over
ceiling): land_fraction ~ coast_fraction_of_land = −0.97, mountain_fraction_of_land
~ mean_land_height = +0.94, river_count ~ total_river_length = +0.92. Per the
pre-registered rule ("else the factor set is redesigned *before* training"), the
set is revised to scale-free / non-duplicated properties. New set (definitions in
`src/fantasy_maps/factors.py`; each a pure function of quantities already logged
per map, so no regeneration needed):

- `land_fraction` (kept)
- `mountain_fraction_of_land` (kept)
- `coastline_raggedness` = coast_cells / √land_cells — replaces `coast_fraction_of_land` (a size-coupled ratio)
- `river_density` = river_count / land_cells — replaces `river_count` + `total_river_length` (collapsed; length was a duplicate view)
- `lake_count` (kept); `mean_land_height` dropped (duplicated mountain fraction)

Worst surviving pair on the current data: land_fraction ~ coastline_raggedness =
−0.74 (under ceiling). The correlation gate is re-adjudicated on the full 33k set
once generation completes; final matrix recorded in results/dataset_summary.json.
