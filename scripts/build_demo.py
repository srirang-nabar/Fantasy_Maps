"""Build the "This Fantasy Map Does Not Exist" latent-slider demo (plan.md Stage 4).

Picks the highest-MIG checkpoint, encodes the eval set to find a prototype latent
and each dimension's realistic traversal range, then decodes a strip of maps per
latent dimension. Emits a self-contained static HTML (all images inlined as
base64 — no server, no model at runtime) where each slider walks one latent, and
also a static traversal-grid PNG for the report.

Each latent is labeled with the ground-truth factor it most tracks (mutual
information on the eval codes), so the demo shows the disentanglement honestly:
with low MIG, moving one slider changes several map properties at once.

    uv run python scripts/build_demo.py
"""
import base64
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from fantasy_maps import audit, dataset, evaluate, factors, metrics, models

ROOT = audit.ROOT
CKPT = ROOT / "results" / "checkpoints"
STEPS = 9


def _best_checkpoint():
    data = json.loads((ROOT / "results" / "stage3_metrics.json").read_text())
    best = max((r for r in data["results"] if r["model"] == "bvae"), key=lambda r: r["mig"])
    return CKPT / f"{best['tag']}.pt", best


def _png_data_uri(arr: np.ndarray, scale: int = 2) -> str:
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@torch.no_grad()
def main():
    pt, meta = _best_checkpoint()
    print(f"best checkpoint: {meta['tag']} (MIG={meta['mig']:.3f})")
    ck = torch.load(pt, map_location="cpu", weights_only=False)
    sc = ck["sidecar"]
    model = models.ConvVAE(sc["img_size"], sc["latent_dim"], sc["channels"])
    model.load_state_dict(ck["model_state"]); model.eval()

    imgs, facs, _seeds = evaluate.load_eval(sc["img_size"])
    codes, _ = evaluate.encode(model, imgs)
    z0 = np.median(codes, axis=0)                       # prototype latent
    lo = np.percentile(codes, 5, axis=0)
    hi = np.percentile(codes, 95, axis=0)

    # label each latent by the factor it most tracks (MI on eval)
    mi = metrics.mutual_info_matrix(codes, facs)        # (D, K)
    order_names = factors.GROUND_TRUTH
    labels = []
    for j in range(sc["latent_dim"]):
        k = int(np.argmax(mi[j])); labels.append({"factor": order_names[k], "mi": float(mi[j, k])})
    dim_order = sorted(range(sc["latent_dim"]), key=lambda j: -labels[j]["mi"])  # most meaningful first

    # decode a traversal strip per latent
    strips = {}
    for j in dim_order:
        frames = []
        for t in np.linspace(lo[j], hi[j], STEPS):
            z = torch.tensor(z0, dtype=torch.float32).clone().unsqueeze(0)
            z[0, j] = float(t)
            img = model.decode(z)[0].permute(1, 2, 0).numpy()
            frames.append(_png_data_uri(img))
        strips[j] = frames

    # static traversal grid (top 8 latents) for the report
    top = dim_order[:8]
    fig, axes = plt.subplots(len(top), STEPS, figsize=(STEPS, len(top)))
    for row, j in enumerate(top):
        for col, uri in enumerate(strips[j]):
            b = base64.b64decode(uri.split(",", 1)[1])
            axes[row, col].imshow(Image.open(io.BytesIO(b))); axes[row, col].axis("off")
        axes[row, 0].set_ylabel(f"z{j}\n{labels[j]['factor'][:10]}", rotation=0, ha="right",
                                va="center", fontsize=7)
        axes[row, 0].axis("on"); axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
    fig.suptitle(f"Latent traversals — {meta['tag']} (MIG={meta['mig']:.3f})", fontsize=10)
    fig.tight_layout()
    grid_path = ROOT / "results" / "latent_traversals.png"
    fig.savefig(grid_path, dpi=130); print(f"wrote {grid_path}")

    _write_html(meta, sc, labels, dim_order, strips)


def _write_html(meta, sc, labels, dim_order, strips):
    payload = {"strips": {str(j): strips[j] for j in dim_order},
               "order": dim_order,
               "labels": {str(j): labels[j] for j in range(sc["latent_dim"])},
               "steps": STEPS, "tag": meta["tag"], "mig": round(meta["mig"], 3),
               "img_size": sc["img_size"]}
    html = _HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload))
    out = ROOT / "report" / "demo.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, self-contained)")


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>This Fantasy Map Does Not Exist</title>
<style>
:root{--bg:#faf9f6;--card:#fff;--ink:#141414;--muted:#6b6a66;--line:#e6e5e1;--accent:#2a78d6}
@media(prefers-color-scheme:dark){:root{--bg:#171716;--card:#212120;--ink:#f3f2ee;--muted:#a3a29a;--line:#33322f;--accent:#3987e5}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;padding:24px}
.wrap{max-width:720px;margin:0 auto}h1{font-size:24px;margin:0 0 2px}
.sub{color:var(--muted);margin:0 0 20px;font-size:13px}
.card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:20px;margin-bottom:16px}
.map{display:block;width:100%;max-width:320px;margin:0 auto;image-rendering:pixelated;border-radius:10px;border:1px solid var(--line)}
.row{display:flex;align-items:center;gap:12px;margin-top:16px}
input[type=range]{flex:1;accent-color:var(--accent)}
select{background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px;font-size:14px;width:100%}
.tag{color:var(--muted);font-size:12px}.big{color:var(--accent);font-weight:600}
.note{font-size:12.5px;color:var(--muted);line-height:1.55}
</style></head><body><div class="wrap">
<h1>This Fantasy Map Does Not Exist</h1>
<p class="sub">Every map below is decoded from a &beta;-VAE latent vector — no map, no seed. Drag a slider to walk one latent dimension.</p>
<div class="card">
<img id="map" class="map" alt="generated map">
<div class="row"><label class="tag" style="min-width:90px">Latent</label>
<select id="dim"></select></div>
<div class="row"><label class="tag" style="min-width:90px">Value</label>
<input id="slider" type="range" min="0" max="8" value="4"></div>
<p class="tag" id="caption"></p>
</div>
<div class="card"><p class="note" id="sci"></p></div>
</div>
<script>
const D=__PAYLOAD__;
const dimSel=document.getElementById('dim'),sl=document.getElementById('slider'),
 map=document.getElementById('map'),cap=document.getElementById('caption');
sl.max=D.steps-1;sl.value=(D.steps-1>>1);
D.order.forEach(j=>{const o=document.createElement('option');const L=D.labels[j];
 o.value=j;o.textContent=`z${j}  — tracks ${L.factor} (MI=${L.mi.toFixed(2)})`;dimSel.appendChild(o);});
function render(){const j=dimSel.value;map.src=D.strips[j][sl.value];
 const L=D.labels[j];cap.innerHTML=`Walking latent <b>z${j}</b>, which most tracks <span class="big">${L.factor}</span>. `+
 `Because disentanglement is weak (headline MIG ≈ ${D.mig}), this slider also nudges other map properties — that entanglement is the result.`;}
dimSel.onchange=render;sl.oninput=render;
document.getElementById('sci').innerHTML=`Model: <b>${D.tag}</b> at ${D.img_size}px, the highest-MIG cell of the &beta;-sweep. `+
 `Latents are ordered by how strongly they track any ground-truth factor. A clean disentangled model would give one crisp slider per factor; `+
 `here the sliders are soft and overlapping — the visible face of the low measured MIG (see CLAIMS.md).`;
render();
</script></body></html>"""


if __name__ == "__main__":
    main()
