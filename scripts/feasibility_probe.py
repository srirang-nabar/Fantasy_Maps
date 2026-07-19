"""Feasibility test: batch-generate Azgaar fantasy maps headlessly with logged factors."""
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).parent / "fmg_out"
OUT.mkdir(exist_ok=True)
URL = "https://azgaar.github.io/Fantasy-Map-Generator/"
SEEDS = [11111, 22222, 33333]

FACTOR_JS = """() => {
  const val = id => { const el = document.getElementById(id); return el ? el.value : null; };
  return {
    seed: typeof seed !== 'undefined' ? seed : null,
    template: val('templateInput'),
    pointsInput: val('pointsInput'),
    statesNumber: val('statesNumber'),
    provincesRatio: val('provincesRatio'),
    religionsNumber: val('religionsNumber'),
    sizeVariety: val('sizeVariety'),
    growthRate: val('growthRate'),
    culturesNumber: val('culturesInput'),
    temperatureEquator: val('temperatureEquatorInput'),
    precipitation: val('precInput'),
    // derived map properties (candidate ground-truth factors)
    landCells: (typeof pack !== 'undefined' && pack.cells) ? pack.cells.h.filter(h => h >= 20).length : null,
    totalCells: (typeof pack !== 'undefined' && pack.cells) ? pack.cells.h.length : null,
    burgsCount: (typeof pack !== 'undefined' && pack.burgs) ? pack.burgs.length - 1 : null,
    riversCount: (typeof pack !== 'undefined' && pack.rivers) ? pack.rivers.length : null,
    statesCount: (typeof pack !== 'undefined' && pack.states) ? pack.states.length - 1 : null,
  };
}"""

results = []
with sync_playwright() as p:
    browser = p.chromium.launch(
        executable_path="/usr/bin/google-chrome",
        headless=True,
        args=["--no-sandbox", "--disable-dev-shm-usage"],
    )
    page = browser.new_page(viewport={"width": 1000, "height": 800})
    for s in SEEDS:
        t0 = time.time()
        page.goto(f"{URL}?seed={s}&width=1000&height=800", timeout=90000)
        # wait until the map data structure is populated
        page.wait_for_function(
            "() => typeof pack !== 'undefined' && pack.cells && pack.cells.h && pack.cells.h.length > 0",
            timeout=90000,
        )
        page.wait_for_timeout(1500)  # let rendering settle
        factors = page.evaluate(FACTOR_JS)
        gen_time = round(time.time() - t0, 1)
        png = OUT / f"map_{s}.png"
        page.locator("#map").screenshot(path=str(png))
        factors["gen_seconds"] = gen_time
        factors["png_bytes"] = png.stat().st_size
        results.append(factors)
        print(f"seed={s}: {gen_time}s, factors={json.dumps(factors)[:200]}")
    browser.close()

(OUT / "factors.json").write_text(json.dumps(results, indent=2))
print("DONE:", len(results), "maps")
