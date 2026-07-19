#!/bin/bash
# β-VAE sweep (plan.md Stage 2): trains every (beta, seed) cell from
# configs/train_sweep.yaml, then fits the PCA baseline.
#
# PY selects the interpreter: defaults to `uv run python` (local dev); on a GPU
# box set PY=python3 so it uses the image's CUDA torch (see scripts/setup_vast.sh).
# Resumable: each cell writes its own checkpoint; delete a cell's .pt to redo it.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-uv run python}"
# Make fantasy_maps importable without an editable install (needed on NGC/GPU
# boxes where we run the image's python, not the uv venv). Harmless locally.
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
# CFG selects the sweep config; default 64px. For the 128px stretch:
#   CFG=configs/train_sweep_128.yaml PY=python3 scripts/run_sweep.sh
CFG="${CFG:-configs/train_sweep.yaml}"

BETAS=$($PY -c "import yaml;print(' '.join(map(str,yaml.safe_load(open('$CFG'))['betas'])))")
SEEDS=$($PY -c "import yaml;print(' '.join(map(str,yaml.safe_load(open('$CFG'))['seeds'])))")
IMG=$($PY -c "import yaml;print(yaml.safe_load(open('$CFG'))['img_size'])")
# loss kind -> checkpoint filename prefix ("" for plain vae, "tcvae_" etc.)
KIND=$($PY -c "import yaml;k=yaml.safe_load(open('$CFG')).get('loss','vae');print('' if k=='vae' else k+'_')")

for beta in $BETAS; do
  for seed in $SEEDS; do
    ckpt="results/checkpoints/${KIND}beta${beta}_seed${seed}_${IMG}px.pt"
    if [ -f "$ckpt" ]; then
      echo "skip existing $ckpt"; continue
    fi
    echo "=== training ${KIND}beta=$beta seed=$seed ==="
    $PY -m fantasy_maps.train --config "$CFG" --beta "$beta" --seed "$seed"
  done
done

# PCA baseline is loss-independent — fit once per resolution, reuse across sweeps.
if [ ! -f "results/checkpoints/pca_16d_${IMG}px.npz" ]; then
  echo "=== fitting PCA baseline ==="
  $PY -c "from fantasy_maps import baseline; baseline.fit_pca(img_size=$IMG, n_components=16)"
else
  echo "=== PCA baseline pca_16d_${IMG}px.npz exists, reusing ==="
fi
echo "SWEEP COMPLETE $(date)"
