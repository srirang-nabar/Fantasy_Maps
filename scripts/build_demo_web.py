"""Build the hosted, mobile-first web demo with a β selector (plan.md Stage 4 / hosting).

Decodes latent traversals for all four β models (best-MIG seed per β, 128px) so the
page lets a viewer switch β and see whether higher β gives cleaner sliders — H1 made
interactive. Each latent is labeled by the factor it most tracks; weak/entangled ones
get their *top two* factors (e.g. "land + coastline") so redundant "land" latents read
as the honest entanglement result, not a copy-paste bug.

Emits report/demo_web.html (standalone) and report/_artifact_demo.html (body-only for
the Artifact tool). Images are 128px PNGs (no 2x upsize) to keep the mobile payload light.

    uv run python scripts/build_demo_web.py
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from fantasy_maps import audit, evaluate, factors, metrics, models

ROOT = audit.ROOT
CKPT = ROOT / "results" / "checkpoints"
RES, TOP, STEPS = 128, 6, 9
BETAS = [1.0, 2.0, 4.0, 8.0]
CLEAN_MI = 0.30
HUMAN = {
    "mountain_fraction_of_land": "mountains", "land_fraction": "land amount",
    "coastline_raggedness": "coastline", "river_density": "rivers", "lake_count": "lakes",
}


def _uri(arr):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _best_tag_per_beta():
    data = json.loads((ROOT / "results" / "stage3_metrics.json").read_text())
    out = {}
    for r in data["results"]:
        if r["model"] == "bvae" and r["img_size"] == RES:
            b = r["beta"]
            if b not in out or r["mig"] > out[b]["mig"]:
                out[b] = r
    return {b: out[b] for b in BETAS}


@torch.no_grad()
def _model_payload(tag_row, imgs, facs, names):
    ck = torch.load(CKPT / f"{tag_row['tag']}.pt", map_location="cpu", weights_only=False)
    sc = ck["sidecar"]
    m = models.ConvVAE(sc["img_size"], sc["latent_dim"], sc["channels"])
    m.load_state_dict(ck["model_state"]); m.eval()
    codes, _ = evaluate.encode(m, imgs)
    z0 = np.median(codes, 0); lo = np.percentile(codes, 5, 0); hi = np.percentile(codes, 95, 0)
    mi = metrics.mutual_info_matrix(codes, facs)                    # (D, K)
    order = list(np.argsort(-mi.max(1))[:TOP])                       # most-informative latents first

    strips, labels = {}, {}
    for j in order:
        j = int(j)
        top2 = np.argsort(-mi[j])[:2]
        primary, second = names[top2[0]], names[top2[1]]
        mip = float(mi[j, top2[0]])
        clean = mip >= CLEAN_MI
        labels[str(j)] = {
            "label": HUMAN[primary] if clean else f"{HUMAN[primary]} + {HUMAN[second]}",
            "primary": HUMAN[primary], "mi": round(mip, 2), "clean": clean,
        }
        frames = []
        for t in np.linspace(lo[j], hi[j], STEPS):
            z = torch.tensor(z0, dtype=torch.float32).clone().unsqueeze(0)
            z[0, j] = float(t)
            frames.append(_uri(m.decode(z)[0].permute(1, 2, 0).numpy()))
        strips[str(j)] = frames
    return {"order": [int(j) for j in order], "strips": strips, "labels": labels,
            "mig": round(tag_row["mig"], 3)}


def _build_payload():
    names = factors.GROUND_TRUTH
    print(f"loading eval set at {RES}px ...")
    imgs, facs, _ = evaluate.load_eval(RES)
    best = _best_tag_per_beta()
    models_payload = {}
    for b in BETAS:
        print(f"  decoding traversals for β={b:g} ({best[b]['tag']}, MIG {best[b]['mig']:.3f})")
        # integer string keys so JS String(betas[i]) ('2') matches the payload key ('2'), not '2.0'
        models_payload[str(int(b))] = _model_payload(best[b], imgs, facs, names)
    return {"betas": [int(b) for b in BETAS], "models": models_payload, "steps": STEPS, "res": RES}


def main():
    # cache the decoded traversals so cosmetic HTML/CSS edits rebuild instantly (pass --fresh to redecode)
    import sys
    cache = ROOT / "results" / "_demo_payload.json"
    if cache.exists() and "--fresh" not in sys.argv:
        print(f"reusing cached traversals ({cache}); pass --fresh to re-decode")
        payload = json.loads(cache.read_text())
    else:
        payload = _build_payload()
        cache.write_text(json.dumps(payload))

    body = _CONTENT.replace("__PAYLOAD__", json.dumps(payload))
    (ROOT / "report" / "_artifact_demo.html").write_text(body)
    doc = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
           '<meta name="viewport" content="width=device-width,initial-scale=1">'
           '<title>This Fantasy Map Does Not Exist</title></head><body>' + body + "</body></html>")
    (ROOT / "report" / "demo_web.html").write_text(doc)
    print(f"wrote report/demo_web.html + _artifact_demo.html (~{len(body)//1024} KB, {len(BETAS)} β models)")


_CSS = """
:root{--paper:#e9edee;--card:#f4f7f7;--ink:#0d1f2d;--ink-soft:#4a5b68;--sea:#16688f;--peak:#cf7d33;
  --line:rgba(13,31,45,.12);--line-strong:rgba(13,31,45,.22);--chip:#dfe6e7}
@media (prefers-color-scheme:dark){:root{--paper:#081521;--card:#0e2334;--ink:#e8f0f5;--ink-soft:#93a7b4;
  --sea:#57b2db;--peak:#e2a25c;--line:rgba(232,240,245,.14);--line-strong:rgba(232,240,245,.26);--chip:#122c40}}
:root[data-theme="light"]{--paper:#e9edee;--card:#f4f7f7;--ink:#0d1f2d;--ink-soft:#4a5b68;--sea:#16688f;--peak:#cf7d33;
  --line:rgba(13,31,45,.12);--line-strong:rgba(13,31,45,.22);--chip:#dfe6e7}
:root[data-theme="dark"]{--paper:#081521;--card:#0e2334;--ink:#e8f0f5;--ink-soft:#93a7b4;--sea:#57b2db;--peak:#e2a25c;
  --line:rgba(232,240,245,.14);--line-strong:rgba(232,240,245,.26);--chip:#122c40}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
  font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased;padding:28px 20px 56px}
.wrap{max-width:600px;margin:0 auto}
.eyebrow{font:600 12px/1 ui-monospace,"SF Mono",Menlo,monospace;letter-spacing:.18em;color:var(--sea)}
h1{font-family:Georgia,"Iowan Old Style",serif;font-weight:600;font-size:clamp(30px,8vw,42px);line-height:1.08;
  text-wrap:balance;margin:14px 0 12px;letter-spacing:-.01em}
.lede{color:var(--ink-soft);font-size:17px;margin:0 0 24px;max-width:52ch}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}
.betabar{display:flex;flex-direction:column;gap:8px;margin-bottom:16px}
.betabar .flabel{display:flex;justify-content:space-between;align-items:baseline}
.seg{display:grid;grid-template-columns:repeat(4,1fr);gap:6px;background:var(--chip);border-radius:11px;padding:5px}
.seg button{font:600 15px/1 ui-monospace,Menlo,monospace;border:0;border-radius:8px;padding:11px 0;color:var(--ink-soft);
  background:transparent;cursor:pointer;transition:background .15s,color .15s}
.seg button[aria-pressed="true"]{background:var(--sea);color:#fff}
.viewer{position:relative;width:100%;max-width:320px;aspect-ratio:1;margin:2px auto 4px;border-radius:12px;overflow:hidden;
  border:1px solid var(--line-strong);box-shadow:inset 0 0 0 6px var(--card),inset 0 0 0 7px var(--line)}
.frame{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;opacity:0;transition:opacity .18s ease}
.frame.on{opacity:1}
@media (prefers-reduced-motion:reduce){.frame{transition:none}}
.controls{margin-top:16px;display:flex;flex-direction:column;gap:14px}
.field{display:flex;flex-direction:column;gap:7px}
.flabel,.flabel span{font:600 11px/1.4 ui-monospace,Menlo,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-soft)}
.flabel .mig{color:var(--sea)}
select{appearance:none;-webkit-appearance:none;background:var(--paper);color:var(--ink);border:1px solid var(--line-strong);
  border-radius:10px;padding:12px 14px;font-size:14px;width:100%;font-family:ui-monospace,Menlo,monospace}
input[type=range]{width:100%;accent-color:var(--peak);height:26px}
.caption{margin:14px 2px 0;font-size:14.5px;color:var(--ink-soft);line-height:1.55}
.caption b{color:var(--ink)}.peak{color:var(--peak);font-weight:600}
.finding{margin-top:30px}
.finding h2{font-family:Georgia,serif;font-weight:600;font-size:22px;margin:0 0 6px;text-wrap:balance}
.finding p{color:var(--ink-soft);margin:0 0 18px;font-size:15.5px}
.stats{display:grid;gap:10px}
.stat{display:flex;align-items:baseline;justify-content:space-between;gap:14px;padding:13px 15px;background:var(--card);
  border:1px solid var(--line);border-radius:11px}
.stat .q{font-size:14.5px}.stat .a{font-family:ui-monospace,Menlo,monospace;font-size:13px;white-space:nowrap;font-variant-numeric:tabular-nums}
.a.no{color:var(--peak)}.a.yes{color:var(--sea)}
.note{margin-top:24px;font-size:13.5px;color:var(--ink-soft);line-height:1.6;border-top:1px solid var(--line);padding-top:18px}
.note a{color:var(--sea)}
footer{margin-top:26px;font:500 11px/1.5 ui-monospace,Menlo,monospace;letter-spacing:.04em;color:var(--ink-soft);text-transform:uppercase}
"""

_CONTENT = """<style>__CSS__</style>
<div class="wrap">
  <div class="eyebrow">&beta;-VAE &middot; UNSUPERVISED DISENTANGLEMENT</div>
  <h1>This Fantasy Map Does Not Exist</h1>
  <p class="lede">Every map here is dreamed up by a neural network from a 16-number code &mdash; no seed, no
  source image. Pick a model, then drag the slider to walk one dimension of that code.</p>

  <div class="card">
    <div class="betabar">
      <div class="flabel"><span>Model &middot; regularization strength</span><span class="mig" id="migTag"></span></div>
      <div class="seg" id="seg" role="group" aria-label="select beta"></div>
    </div>
    <div class="viewer"><img id="fA" class="frame on" alt="generated fantasy map"><img id="fB" class="frame" alt=""></div>
    <div class="controls">
      <div class="field"><span class="flabel">Latent dimension</span><select id="dim" aria-label="latent dimension"></select></div>
      <div class="field"><span class="flabel">Position along it</span><input id="slider" type="range" min="0" max="8" value="4" aria-label="latent value"></div>
    </div>
    <p class="caption" id="cap"></p>
  </div>

  <div class="finding">
    <h2>The point isn&rsquo;t the picture &mdash; it&rsquo;s the sliders.</h2>
    <p>A truly disentangled model gives one clean slider per real property. Turning &beta; up is supposed to
    sharpen that. Switch between the models above: it barely does &mdash; and that&rsquo;s the measured result.</p>
    <div class="stats">
      <div class="stat"><span class="q">Does raising &beta; buy disentanglement?</span><span class="a no">No &middot; p=0.07</span></div>
      <div class="stat"><span class="q">Rate&ndash;distortion tradeoff real?</span><span class="a yes">Yes &middot; p&lt;0.001</span></div>
      <div class="stat"><span class="q">Beats a plain PCA baseline?</span><span class="a yes">Yes, barely &middot; 0.095 vs 0.043</span></div>
      <div class="stat"><span class="q">Does &beta;-TCVAE (built to disentangle) win?</span><span class="a no">No gain &middot; p=0.72</span></div>
    </div>
  </div>

  <p class="note">Several latents share a label like &ldquo;land amount&rdquo; on purpose: the model is entangled, so
  many dimensions redundantly encode the same property and none cleanly isolate the rest &mdash; weak ones are tagged
  by their top two factors. Maps are soft by design (a &beta;-VAE trades sharpness for a structured latent). This is a
  <b>representation-learning measurement</b>, not an image-quality project &mdash; a data point in the
  <b>Locatello et&nbsp;al. (2019)</b> impossibility debate, wearing a wizard hat.</p>

  <footer id="foot"></footer>
</div>
<script>
const D=__PAYLOAD__;
const seg=document.getElementById('seg'),sel=document.getElementById('dim'),sl=document.getElementById('slider'),
  fA=document.getElementById('fA'),fB=document.getElementById('fB'),cap=document.getElementById('cap'),
  migTag=document.getElementById('migTag'),foot=document.getElementById('foot');
let front=fA,back=fB,beta=String(D.betas[1]);
sl.max=D.steps-1;
D.betas.forEach(b=>{const btn=document.createElement('button');btn.textContent='\\u03b2 '+(''+b).replace('.0','');
  btn.setAttribute('aria-pressed',String(b)===beta);btn.onclick=()=>{beta=String(b);syncBeta();};seg.appendChild(btn);});
function syncBeta(){
  [...seg.children].forEach((btn,i)=>btn.setAttribute('aria-pressed',String(D.betas[i])===beta));
  const M=D.models[beta];migTag.textContent='MIG '+M.mig;
  sel.innerHTML='';
  M.order.forEach(j=>{const L=M.labels[j];const o=document.createElement('option');o.value=j;
    o.textContent='z'+j+'  \\u00b7  '+L.label+'  (MI '+L.mi+')';sel.appendChild(o);});
  sl.value=(D.steps-1)>>1;render();
}
function render(){
  const M=D.models[beta],j=sel.value,L=M.labels[j];
  back.src=M.strips[j][sl.value];back.classList.add('on');front.classList.remove('on');[front,back]=[back,front];
  cap.innerHTML='Walking latent <b>z'+j+'</b>, which most tracks <span class="peak">'+L.primary+'</span>. '+
    (L.clean?'This one is fairly clean &mdash; the model found a real knob.'
            :'But it&rsquo;s weak and entangled (MI '+L.mi+'): moving it nudges several properties at once &mdash; exactly what low disentanglement looks like.');
  foot.textContent='\\u03b2='+(''+beta).replace('.0','')+' model \\u00b7 '+D.res+'px \\u00b7 MIG '+M.mig+' \\u00b7 measured, not eyeballed';
}
sel.onchange=render;sl.oninput=render;syncBeta();
</script>"""

_CONTENT = _CONTENT.replace("__CSS__", _CSS)

if __name__ == "__main__":
    main()
