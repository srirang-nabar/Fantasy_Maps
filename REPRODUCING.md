# Reproducing — three tiers (filled in as stages complete)

- **Tier 1 (≤10 min, CPU, no downloads):** `uv sync --frozen`, run numbered notebooks — they load committed checkpoints + committed eval subset and assert vs CLAIMS.md. *Checkpoints present:* `results/checkpoints/` holds the 24-cell β-sweep (64px + 128px) + 2 PCA baselines, all CPU-loadable, hashed in `results/checkpoints/MANIFEST.sha256` with `PROVENANCE.md`. *(metric-asserting notebooks land in Stage 3)*
- **Tier 2 (≤1 hr, CPU):** recompute metrics from committed artifacts. *(after Stage 3)*
- **Tier 3 (GPU day):** regenerate the dataset from the data engine and retrain into published tolerance bands.
  Data engine (working now): `uv run python scripts/build_snapshot.py` then `scripts/run_bulk.sh`
  (resumable; ~11 h at 4 shards). Generator gate tests (browser): `uv run pytest -m gate_stage1`.
  When generation completes, the dataset gate (fast, no browser) checks completeness,
  factor coverage, and the pre-registered correlation ceiling:
  `uv run pytest -m gate_stage1_data`, then `uv run python scripts/build_manifest.py`
  (writes `results/MANIFEST.sha256` + `results/dataset_summary.json`; non-zero exit if the gate fails).
  Config for all three lives in `configs/generation.yaml`.

## Stage 2 — training (GPU)

- Dependencies: `uv sync` (installs CPU torch on this box; a GPU box installs its CUDA build).
- Smoke the code path on CPU (~7 s, junk numbers): `uv run python -m fantasy_maps.train --config configs/train_smoke.yaml --smoke`.
- Path sanity tests (CPU, no GPU): `uv run pytest -m gate_stage2 tests/test_splits.py`.
- Full β-sweep (GPU): `scripts/run_sweep.sh` — trains every (β, seed) cell in `configs/train_sweep.yaml`
  and fits the PCA baseline. Checkpoints + JSON sidecars land in `results/checkpoints/`.
- Splits are deterministic and fingerprinted (`fantasy_maps.dataset.make_splits`); every checkpoint records
  the split fingerprint so metrics never mix incompatible runs.

### Running the sweep on a rented GPU (Vast.ai)

The dev box has no GPU and pyproject pins CPU torch, so do NOT `uv sync` on the GPU box.

1. Local: `scripts/pack_dataset.sh` → `maps.tgz`; also `tar czf project.tgz --exclude=.venv --exclude=data/raw .`
2. Rent a GPU (RTX 3090/4090 is ample) with a PyTorch/CUDA image; `scp -P <port>` both tarballs to `/workspace` (or the repo dir).
3. On the box: extract `project.tgz`, then `PY=python3 scripts/setup_vast.sh` (installs non-torch deps, keeps the image's CUDA torch, unpacks `maps.tgz`, runs a GPU sanity step).
4. `PY=python3 scripts/run_sweep.sh` — trains all 12 cells + PCA baseline into `results/checkpoints/`.
5. `scp` `results/checkpoints/` back; destroy the instance. Est. ~30–50 min, well under $1 on a 3090.
