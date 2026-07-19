"""Dataset audit for the Stage 1 hard gate (plan.md Stage 1, HYPOTHESES.md).

Loads the generated shard JSONLs + rendered PNGs and computes the three
dataset-level checks that decide whether the data engine is done and usable:

  1. completeness  — one record per expected seed, one PNG per record, correct
     image size, no gaps, no duplicates, no orphan images;
  2. coverage      — every ground-truth factor spans a real range (not degenerate);
  3. correlation   — pairwise |rho| between factors is under the pre-registered
     ceiling, else the factor set must be redesigned before any training.

Pure/stateless and browser-free: reads only files under data/raw and the pinned
config. Shared by tests/test_dataset.py, scripts/build_manifest.py, and the
Stage 1 notebook so all three agree on one definition of "done".
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "generation.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text())


@dataclass
class DatasetAudit:
    """Everything the Stage 1 gate needs, computed once from the raw dataset."""

    config: dict
    n_records: int
    seeds: set[int]
    duplicate_seeds: list[int]
    missing_seeds: list[int]
    unexpected_seeds: list[int]
    missing_images: list[int]        # records with no PNG
    orphan_images: list[int]         # PNGs with no record
    wrong_size_images: list[int]     # PNGs not out_size x out_size
    factor_matrix: np.ndarray        # (n_records, n_factors), column order = config["factors"]
    coverage: dict[str, dict]        # per-factor stats + pass flag
    corr: np.ndarray                 # (n_factors, n_factors) Pearson matrix
    max_abs_offdiag: float
    worst_pair: tuple[str, str, float]

    @property
    def factors(self) -> list[str]:
        return list(self.config["factors"])

    @property
    def complete(self) -> bool:
        return not (
            self.duplicate_seeds or self.missing_seeds or self.unexpected_seeds
            or self.missing_images or self.orphan_images or self.wrong_size_images
        )

    @property
    def coverage_ok(self) -> bool:
        return all(c["ok"] for c in self.coverage.values())

    @property
    def correlation_ok(self) -> bool:
        return self.max_abs_offdiag <= self.config["gates"]["max_abs_correlation"]

    @property
    def gate_ok(self) -> bool:
        return self.complete and self.coverage_ok and self.correlation_ok

    def summary(self) -> dict:
        return {
            "n_records": self.n_records,
            "expected": self.config["seed_end"] - self.config["seed_start"] + 1,
            "complete": self.complete,
            "n_duplicate_seeds": len(self.duplicate_seeds),
            "n_missing_seeds": len(self.missing_seeds),
            "n_unexpected_seeds": len(self.unexpected_seeds),
            "n_missing_images": len(self.missing_images),
            "n_orphan_images": len(self.orphan_images),
            "n_wrong_size_images": len(self.wrong_size_images),
            "coverage_ok": self.coverage_ok,
            "coverage": self.coverage,
            "correlation_ok": self.correlation_ok,
            "max_abs_offdiag_correlation": self.max_abs_offdiag,
            "worst_correlated_pair": {
                "a": self.worst_pair[0], "b": self.worst_pair[1], "rho": self.worst_pair[2],
            },
            "gate_ok": self.gate_ok,
        }


def load_records(raw: Path, n_shards: int) -> tuple[list[dict], list[int]]:
    """Read every shard JSONL. Returns (records, duplicate_seeds).

    Skips torn tail lines (an interrupted run may leave a half-written record);
    those seeds simply read as not-done, matching the generator's resume logic.
    """
    records: list[dict] = []
    seen: dict[int, int] = {}
    duplicates: list[int] = []
    for shard in range(n_shards):
        jsonl = raw / f"factors_shard{shard}.jsonl"
        if not jsonl.exists():
            continue
        for line in jsonl.read_text().splitlines():
            try:
                rec = json.loads(line)
                seed = rec["seed"]
            except (json.JSONDecodeError, KeyError):
                continue
            seen[seed] = seen.get(seed, 0) + 1
            if seen[seed] == 2:
                duplicates.append(seed)
            records.append(rec)
    return records, sorted(duplicates)


def _coverage(matrix: np.ndarray, factors: list[str], gates: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for j, name in enumerate(factors):
        col = matrix[:, j]
        if col.size == 0:
            out[name] = {"ok": False, "reason": "no data", "distinct": 0}
            continue
        vals, counts = np.unique(col, return_counts=True)
        distinct = int(vals.size)
        mode_fraction = float(counts.max() / col.size)
        std = float(col.std())
        ok = (
            std > 0.0
            and distinct >= gates["min_distinct"]
            and mode_fraction < gates["max_mode_fraction"]
        )
        out[name] = {
            "ok": bool(ok), "distinct": distinct, "std": std,
            "min": float(col.min()), "max": float(col.max()),
            "mode_fraction": mode_fraction,
        }
    return out


def audit_dataset(raw: Path | None = None, config: dict | None = None) -> DatasetAudit:
    config = config or load_config()
    raw = raw or (ROOT / "data" / "raw")
    factors = list(config["factors"])
    n_shards = config["n_shards"]
    start, end = config["seed_start"], config["seed_end"]
    out_size = config["out_size"]

    records, duplicate_seeds = load_records(raw, n_shards)
    seeds = {r["seed"] for r in records}
    expected = set(range(start, end + 1))
    missing_seeds = sorted(expected - seeds)
    unexpected_seeds = sorted(seeds - expected)

    maps_dir = raw / "maps"
    png_seeds = set()
    if maps_dir.exists():
        for p in maps_dir.glob("*.png"):
            try:
                png_seeds.add(int(p.stem))
            except ValueError:
                continue
    missing_images = sorted(seeds - png_seeds)
    orphan_images = sorted(png_seeds - seeds)

    # Image size check on the records' PNGs (Pillow imported lazily so the audit
    # runs even where Pillow is absent, as long as sizes aren't requested).
    wrong_size_images: list[int] = []
    if maps_dir.exists() and (seeds & png_seeds):
        from PIL import Image
        for seed in sorted(seeds & png_seeds):
            with Image.open(maps_dir / f"{seed}.png") as im:
                if im.size != (out_size, out_size):
                    wrong_size_images.append(seed)

    # Factor matrix in fixed column order; rows sorted by seed for determinism.
    # Ground-truth factors are derived from each record's raw logged measurements
    # (fantasy_maps.factors), so a factor-set redesign needs no regeneration.
    from fantasy_maps import factors as factor_defs

    by_seed = {r["seed"]: r for r in records}
    rows = [
        [factor_defs.derive(by_seed[s]["factors"])[f] for f in factors]
        for s in sorted(by_seed)
    ]
    matrix = np.asarray(rows, dtype=np.float64).reshape(-1, len(factors))

    coverage = _coverage(matrix, factors, config["gates"])

    if matrix.shape[0] >= 2:
        corr = np.corrcoef(matrix, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0)  # a constant factor => nan; treated as 0 (coverage catches it)
    else:
        corr = np.full((len(factors), len(factors)), np.nan)

    max_abs, worst = 0.0, (factors[0], factors[0], 0.0)
    for i, k in itertools.combinations(range(len(factors)), 2):
        rho = float(corr[i, k])
        if abs(rho) > max_abs:
            max_abs, worst = abs(rho), (factors[i], factors[k], rho)

    return DatasetAudit(
        config=config, n_records=len(records), seeds=seeds,
        duplicate_seeds=duplicate_seeds, missing_seeds=missing_seeds,
        unexpected_seeds=unexpected_seeds, missing_images=missing_images,
        orphan_images=orphan_images, wrong_size_images=wrong_size_images,
        factor_matrix=matrix, coverage=coverage, corr=corr,
        max_abs_offdiag=max_abs, worst_pair=worst,
    )
