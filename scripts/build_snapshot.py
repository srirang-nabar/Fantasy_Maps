"""Vendor a pinned snapshot of the deployed Azgaar Fantasy Map Generator.

Captures every network response during a full generation session and writes
the bodies to vendor/azgaar_snapshot/ preserving URL paths, so the app can be
served offline (python -m http.server) and generation is pinned to one version.
Re-runnable; writes a manifest with SHA256 per asset.
"""
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

BASE = "https://azgaar.github.io/Fantasy-Map-Generator/"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "vendor" / "azgaar_snapshot"

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved: dict[str, str] = {}
    skipped: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/usr/bin/google-chrome",
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 512, "height": 512})

        def on_response(resp):
            url = resp.url
            if not url.startswith(BASE):
                skipped.append(url)
                return
            rel = urlparse(url).path.removeprefix(urlparse(BASE).path)
            if rel in ("", "/"):
                rel = "index.html"
            target = OUT / rel
            if target.exists():
                return
            try:
                body = resp.body()
            except Exception as e:  # redirects/opaque responses
                skipped.append(f"{url} ({e})")
                return
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            saved[rel] = hashlib.sha256(body).hexdigest()

        page.on("response", on_response)
        # two seeds with different templates to pull template-dependent assets
        for seed in (11111, 22222):
            page.goto(f"{BASE}?seed={seed}&width=512&height=512", timeout=120000)
            page.wait_for_function(
                "() => typeof pack !== 'undefined' && pack.cells && pack.cells.h && pack.cells.h.length > 0",
                timeout=120000,
            )
            page.wait_for_timeout(2000)
        browser.close()

    manifest = {
        "base_url": BASE,
        "assets": dict(sorted(saved.items())),
        "n_assets": len(saved),
        "external_skipped": sorted(set(skipped)),
    }
    (OUT / "SNAPSHOT_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"saved {len(saved)} assets to {OUT}")
    print(f"skipped {len(set(skipped))} external/opaque urls")
    if not (OUT / "index.html").exists():
        print("ERROR: no index.html captured", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
