"""Manifest the generated dataset — the Stage 1 hard gate's fingerprint step.

Runs the dataset audit, then writes:
  - results/MANIFEST.sha256      : `<sha256>  <relpath>` for every PNG + shard
                                   JSONL + the pinned config, sorted by path;
                                   the reproducibility fingerprint (data/raw is
                                   gitignored, so this is how a Tier-3 regen is
                                   checked against the published dataset).
  - results/dataset_summary.json : audit summary (counts, coverage, correlation,
                                   gate verdict) — the human-readable companion.

Exit code is non-zero if the Stage 1 gate does not pass, so this doubles as a
CI-style guard: a green manifest run means the dataset is complete, covered, and
under the correlation ceiling.

    uv run python scripts/build_manifest.py
"""
import hashlib
import json
import sys
from pathlib import Path

from fantasy_maps import audit

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_paths(config: dict) -> list[Path]:
    paths: list[Path] = [audit.CONFIG_PATH]
    for shard in range(config["n_shards"]):
        j = RAW / f"factors_shard{shard}.jsonl"
        if j.exists():
            paths.append(j)
    maps = RAW / "maps"
    if maps.exists():
        paths.extend(sorted(maps.glob("*.png"), key=lambda p: int(p.stem)))
    return paths


def main() -> int:
    a = audit.audit_dataset(RAW)
    summary = a.summary()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "dataset_summary.json").write_text(json.dumps(summary, indent=2))

    paths = manifest_paths(a.config)
    lines = []
    for i, p in enumerate(paths, 1):
        lines.append(f"{sha256(p)}  {p.relative_to(ROOT)}")
        if i % 2000 == 0:
            print(f"  hashed {i}/{len(paths)} files")
    (RESULTS / "MANIFEST.sha256").write_text("\n".join(lines) + "\n")

    print(f"\nmanifested {len(paths)} files -> results/MANIFEST.sha256")
    print(f"records={summary['n_records']}/{summary['expected']}  "
          f"complete={summary['complete']}  coverage_ok={summary['coverage_ok']}  "
          f"correlation_ok={summary['correlation_ok']}")
    wp = summary["worst_correlated_pair"]
    print(f"worst factor pair: {wp['a']} ~ {wp['b']}  rho={wp['rho']:+.3f} "
          f"(ceiling {a.config['gates']['max_abs_correlation']})")
    if not a.coverage_ok:
        bad = [n for n, c in a.coverage.items() if not c["ok"]]
        print(f"DEGENERATE FACTORS: {bad}", file=sys.stderr)

    if not a.gate_ok:
        print("\nStage 1 gate: FAIL (see results/dataset_summary.json)", file=sys.stderr)
        return 1
    print("\nStage 1 gate: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
