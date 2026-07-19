"""Data engine: batch-generate factor-labeled fantasy maps from the pinned Azgaar snapshot.

Design (see plan.md Stage 1):
- Serves the vendored snapshot locally; every map is generated at a fixed
  512x512 viewport (map size is part of the generation config — factors are
  size-dependent), rendered with the clean heightmap preset (no labels/UI),
  screenshotted, and downsampled to 256x256.
- Per-map record appended to a shard JSONL: seed, generator knobs (DOM),
  derived ground-truth factors (from the in-page `pack` model), snapshot
  hash, render time.
- Resumable: seeds already present in the shard's JSONL are skipped.
- Parallelism via independent shard processes (`--shard-index/--shard-count`),
  each with its own browser; shards never share files.
"""
import argparse
import hashlib
import io
import json
import subprocess
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "vendor" / "azgaar_snapshot"
RAW = ROOT / "data" / "raw"
VIEW = 512
OUT_SIZE = 256

CLEAN_JS = """() => {
  for (const el of document.body.children) if (el.id !== 'map') el.style.display = 'none';
  for (const id of ['scaleBar','vignette']) { const g = document.getElementById(id); if (g) g.remove(); }
}"""

# knobs from the DOM + derived ground-truth factors from the data model
FACTOR_JS = """() => {
  const val = id => { const el = document.getElementById(id); return el ? el.value : null; };
  const h = pack.cells.h; const n = h.length;
  let land = 0, mountain = 0, coast = 0, hsum = 0;
  for (let i = 0; i < n; i++) {
    if (h[i] >= 20) {
      land++; hsum += h[i];
      if (h[i] >= 60) mountain++;
      // coastal land cell: any neighbor is water
      const nb = pack.cells.c[i];
      for (const j of nb) { if (pack.cells.h[j] < 20) { coast++; break; } }
    }
  }
  let riverLen = 0;
  for (const r of pack.rivers) riverLen += (r.length || 0);
  return {
    template: val('templateInput'),
    knobs: {
      pointsInput: val('pointsInput'), statesNumber: val('statesNumber'),
      culturesNumber: val('culturesInput'), religionsNumber: val('religionsNumber'),
      precipitation: val('precInput'), temperatureEquator: val('temperatureEquatorInput'),
    },
    factors: {
      land_fraction: land / n,
      mountain_fraction_of_land: land ? mountain / land : 0,
      coast_fraction_of_land: land ? coast / land : 0,
      mean_land_height: land ? hsum / land : 0,
      river_count: pack.rivers.length,
      total_river_length: riverLen,
      lake_count: pack.features.filter(f => f && f.type === 'lake').length,
      n_cells: n,
    },
  };
}"""


def snapshot_hash() -> str:
    manifest = SNAPSHOT / "SNAPSHOT_MANIFEST.json"
    return hashlib.sha256(manifest.read_bytes()).hexdigest()[:16]


def done_seeds(jsonl: Path) -> set[int]:
    if not jsonl.exists():
        return set()
    seeds = set()
    for line in jsonl.read_text().splitlines():
        try:
            seeds.add(json.loads(line)["seed"])
        except (json.JSONDecodeError, KeyError):
            continue  # torn tail line from an interrupted run; will be regenerated
    return seeds


def generate_shard(seeds: list[int], shard: int, port: int, headed_retry: int = 2) -> None:
    maps_dir = RAW / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)
    jsonl = RAW / f"factors_shard{shard}.jsonl"
    done = done_seeds(jsonl)
    todo = [s for s in seeds if s not in done]
    snap = snapshot_hash()
    print(f"[shard {shard}] {len(todo)} to generate ({len(done)} already done)")
    if not todo:
        return

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/usr/bin/google-chrome", headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        ctx = browser.new_context(viewport={"width": VIEW, "height": VIEW})
        ctx.add_init_script("localStorage.setItem('preset','heightmap');")
        page = ctx.new_page()
        with jsonl.open("a") as out:
            for i, seed in enumerate(todo):
                t0 = time.time()
                for attempt in range(headed_retry + 1):
                    try:
                        page.goto(f"http://localhost:{port}/index.html?seed={seed}&width={VIEW}&height={VIEW}",
                                  timeout=60000)
                        page.wait_for_function(
                            "() => typeof pack !== 'undefined' && pack.cells && pack.cells.h && pack.cells.h.length > 0",
                            timeout=60000)
                        page.wait_for_timeout(700)
                        page.evaluate(CLEAN_JS)
                        rec = page.evaluate(FACTOR_JS)
                        png = page.locator("#map").screenshot()
                        break
                    except Exception as e:
                        if attempt == headed_retry:
                            print(f"[shard {shard}] seed {seed} FAILED after retries: {e}")
                            rec = None
                        else:  # recycle the page and retry
                            page.close()
                            page = ctx.new_page()
                if rec is None:
                    continue
                img = Image.open(io.BytesIO(png)).convert("RGB").resize((OUT_SIZE, OUT_SIZE), Image.LANCZOS)
                img.save(maps_dir / f"{seed}.png", optimize=True)
                rec.update(seed=seed, view=VIEW, out_size=OUT_SIZE,
                           snapshot=snap, gen_seconds=round(time.time() - t0, 2))
                out.write(json.dumps(rec) + "\n")
                out.flush()
                if (i + 1) % 100 == 0:
                    print(f"[shard {shard}] {i+1}/{len(todo)} ({(time.time()-t0):.1f}s last)")
        browser.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=33000, help="inclusive")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--port", type=int, default=8901)
    ap.add_argument("--serve", action="store_true", help="also start the snapshot server (shard 0 convenience)")
    args = ap.parse_args()

    server = None
    if args.serve:
        server = subprocess.Popen(
            ["python3", "-m", "http.server", str(args.port), "-d", str(SNAPSHOT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
    try:
        seeds = [s for s in range(args.start, args.end + 1) if s % args.shard_count == args.shard_index]
        generate_shard(seeds, args.shard_index, args.port)
    finally:
        if server:
            server.terminate()


if __name__ == "__main__":
    main()
