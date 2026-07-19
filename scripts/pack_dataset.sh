#!/bin/bash
# Pack the dataset for upload to a GPU box (Vast.ai etc.). Run locally.
# Produces maps.tgz (images + factor shards); scp it to the instance and let
# scripts/setup_vast.sh unpack it. The generated data is not in git, so this
# tarball (or a re-run of the data engine) is how the dataset travels.
set -euo pipefail
cd "$(dirname "$0")/.."
tar czf maps.tgz data/raw/maps data/raw/factors_shard*.jsonl
echo "wrote maps.tgz ($(du -h maps.tgz | cut -f1))"
echo "next: scp maps.tgz  root@<vast-host>:<port>:/workspace/  (or into the repo root), then run scripts/setup_vast.sh"
