#!/bin/bash
# Turnkey setup for a Vast.ai (or any CUDA) GPU box. Run from the project root
# AFTER uploading the repo and the dataset tarball (see scripts/pack_dataset.sh).
#
# Why not `uv sync`: the committed pyproject pins CPU torch (via [tool.uv.sources])
# for the no-GPU dev box. On a GPU instance that is exactly wrong — it would train
# on CPU. So we keep the image's CUDA torch and install only what's missing.
#
# Works on NGC PyTorch containers (Python 3.10/3.12): we use PYTHONPATH rather than
# `pip install -e .` (which pyproject's requires-python>=3.13 would reject), and we
# only install deps the image lacks — never force-upgrading its numpy/torch pairing.
#
#   PY=python3 scripts/setup_vast.sh      # PY defaults to python3 (the CUDA image's)
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python3}"
export PYTHONPATH="$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"

echo "=== python: $($PY --version) ==="
echo "=== installing only missing deps (keeping the image's CUDA torch + numpy) ==="
$PY - <<'EOF'
import importlib, subprocess, sys
need = []
for mod, pkg in [("yaml", "pyyaml"), ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib"),
                 ("numpy", "numpy"), ("PIL", "pillow")]:
    try:
        importlib.import_module(mod)
    except ImportError:
        need.append(pkg)
if need:
    print("installing:", need)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *need])
else:
    print("all non-torch deps already present")
EOF

echo "=== torch / CUDA check ==="
$PY - <<'EOF'
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "no CUDA GPU visible — rent a GPU instance and use a CUDA/PyTorch image"
print("device:", torch.cuda.get_device_name(0))
EOF

echo "=== dataset ==="
if [ ! -d data/raw/maps ] || [ -z "$(ls -A data/raw/maps 2>/dev/null)" ]; then
  TB="$(ls maps.tgz dataset.tgz data_raw.tgz /workspace/maps.tgz 2>/dev/null | head -1 || true)"
  if [ -n "$TB" ]; then
    echo "extracting $TB ..."
    tar xzf "$TB"
  else
    echo "ERROR: data/raw/maps is empty and no tarball found (maps.tgz)."
    echo "  Locally: scripts/pack_dataset.sh ; scp maps.tgz to this box; re-run."
    exit 1
  fi
fi
echo "maps present: $(ls data/raw/maps | wc -l)"

echo "=== GPU sanity: one training step on device ==="
$PY - <<'EOF'
import torch
from fantasy_maps import models
m = models.ConvVAE(64, 16, 32).cuda()
opt = torch.optim.Adam(m.parameters(), 1e-3)
x = torch.rand(128, 3, 64, 64, device="cuda")
r, mu, lv, z = m(x)
loss, _, _ = models.vae_loss(r, x, mu, lv, 4.0)
opt.zero_grad(); loss.backward(); opt.step()
print("one CUDA training step OK, loss=%.1f" % loss.item())
EOF

echo
echo "Setup complete. Launch the full sweep with:"
echo "    PY=\"$PY\" scripts/run_sweep.sh"
echo "Checkpoints + sidecars will land in results/checkpoints/ — scp those back when done."
