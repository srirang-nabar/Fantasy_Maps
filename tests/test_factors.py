"""Fast unit tests for the ground-truth factor definitions (fantasy_maps.factors).

Browser-free and data-free — always run. They pin the redesigned factor set so
the config, the derivation, and the audit can't silently drift apart.
"""
import math

from fantasy_maps import audit, factors


def test_config_matches_ground_truth():
    """configs/generation.yaml must list exactly factors.GROUND_TRUTH, in order."""
    assert audit.load_config()["factors"] == factors.GROUND_TRUTH


def test_derive_returns_exactly_the_ground_truth_keys():
    raw = {
        "land_fraction": 0.5, "mountain_fraction_of_land": 0.2,
        "coast_fraction_of_land": 0.1, "mean_land_height": 30.0,
        "river_count": 40, "total_river_length": 2000.0, "lake_count": 7, "n_cells": 10000,
    }
    assert set(factors.derive(raw)) == set(factors.GROUND_TRUTH)


def test_derive_known_values():
    raw = {
        "land_fraction": 0.5, "mountain_fraction_of_land": 0.2,
        "coast_fraction_of_land": 0.1, "mean_land_height": 30.0,
        "river_count": 40, "total_river_length": 2000.0, "lake_count": 7, "n_cells": 10000,
    }
    d = factors.derive(raw)
    land_cells = 0.5 * 10000            # 5000
    coast_cells = 0.1 * land_cells      # 500
    assert d["land_fraction"] == 0.5
    assert d["mountain_fraction_of_land"] == 0.2
    assert math.isclose(d["coastline_raggedness"], coast_cells / math.sqrt(land_cells))
    assert math.isclose(d["river_density"], 40 / land_cells)
    assert d["lake_count"] == 7.0


def test_derive_handles_zero_land():
    raw = {
        "land_fraction": 0.0, "mountain_fraction_of_land": 0.0,
        "coast_fraction_of_land": 0.0, "mean_land_height": 0.0,
        "river_count": 0, "total_river_length": 0.0, "lake_count": 0, "n_cells": 10000,
    }
    d = factors.derive(raw)
    assert d["coastline_raggedness"] == 0.0
    assert d["river_density"] == 0.0
