"""Ground-truth factor set for the disentanglement benchmark (HYPOTHESES.md).

Redesigned after the Stage 1 correlation gate rejected the original set: three
pairs sat over the pre-registered 0.8 |rho| ceiling on the generated data —
land_fraction ~ coast_fraction_of_land (-0.97), mountain_fraction_of_land ~
mean_land_height (+0.94), river_count ~ total_river_length (+0.92). The size-
coupled ratio and the duplicate river/height measures are replaced with scale-
free properties; worst surviving pair is land_fraction ~ coastline_raggedness
(-0.74), under the ceiling.

Each factor is a pure function of the raw quantities that generate.FACTOR_JS
already logs per map, so the existing dataset is relabeled, not regenerated.
This module is the single source of truth for what the ground-truth factors are;
configs/generation.yaml's `factors:` list must match GROUND_TRUTH (asserted in
tests/test_factors.py).
"""
from __future__ import annotations

import math

GROUND_TRUTH = [
    "land_fraction",
    "mountain_fraction_of_land",
    "coastline_raggedness",
    "river_density",
    "lake_count",
]


def derive(raw: dict) -> dict[str, float]:
    """Map a record's logged `factors` dict to the ground-truth factor set.

    `raw` is the in-page measurement block (generate.FACTOR_JS): land_fraction,
    mountain_fraction_of_land, coast_fraction_of_land, river_count,
    total_river_length, lake_count, n_cells. Raw cell counts are reconstructed
    from the logged fractions; land-free maps (never expected — land_fraction is
    floored well above 0) degrade to 0 rather than dividing by zero.
    """
    n = raw["n_cells"]
    land_cells = raw["land_fraction"] * n
    coast_cells = raw["coast_fraction_of_land"] * land_cells
    return {
        "land_fraction": float(raw["land_fraction"]),
        "mountain_fraction_of_land": float(raw["mountain_fraction_of_land"]),
        "coastline_raggedness": coast_cells / math.sqrt(land_cells) if land_cells > 0 else 0.0,
        "river_density": raw["river_count"] / land_cells if land_cells > 0 else 0.0,
        "lake_count": float(raw["lake_count"]),
    }
