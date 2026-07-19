#!/bin/bash
# Bulk generation: 33k maps, 4 shards, one snapshot server. Resumable — rerun safely.
cd "$(dirname "$0")/.."
python3 -m http.server 8901 -d vendor/azgaar_snapshot >/dev/null 2>&1 &
SERVER=$!
sleep 1
for i in 0 1 2 3; do
  .venv/bin/python -m fantasy_maps.generate --start 1 --end 33000 --shard-index $i --shard-count 4 --port 8901 \
    > data/raw/logs/shard$i.log 2>&1 &
done
wait %2 %3 %4 %5 2>/dev/null
kill $SERVER 2>/dev/null
echo "BULK GENERATION COMPLETE $(date)" >> data/raw/logs/bulk.done
