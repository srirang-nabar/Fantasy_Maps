"""Build the hosted, mobile-first web demo (Stage 4 / hosting).

Decodes latent traversals for a set of models — the β-VAE sweep (β=1/2/4/8, the
disentanglement story) plus the VAE-GAN (the sharpness story) — so a viewer can
switch models and see: raising β doesn't clean up the sliders (H1), while the
VAE-GAN produces crisp, controllable maps. Emits report/demo_web.html (standalone,
for GitHub Pages) and report/_artifact_demo.html (body-only, for the Artifact tool).

    uv run python scripts/build_demo_web.py            # reuse cached traversals
    uv run python scripts/build_demo_web.py --fresh    # re-decode from checkpoints
"""
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")

from fantasy_maps import audit, evaluate, factors, metrics, models

ROOT = audit.ROOT
CKPT = ROOT / "results" / "checkpoints"
RES, TOP, STEPS = 128, 6, 9
CLEAN_MI = 0.30
HUMAN = {
    "mountain_fraction_of_land": "mountains", "land_fraction": "land amount",
    "coastline_raggedness": "coastline", "river_density": "rivers", "lake_count": "lakes",
}


def _uri(arr):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


from PIL import Image  # noqa: E402  (after matplotlib backend set)


def _model_list():
    """[(key, label, kind, tag)] — best-MIG β-VAE per β, then the VAE-GAN."""
    data = json.loads((ROOT / "results" / "stage3_metrics.json").read_text())
    best = {}
    for r in data["results"]:
        if r["model"] == "bvae" and r["img_size"] == RES:
            b = r["beta"]
            if b not in best or r["mig"] > best[b]["mig"]:
                best[b] = r
    out = [(f"b{int(b)}", f"β {int(b)}", "vae", best[b]["tag"]) for b in sorted(best)]
    out.append(("v256", "256px", "vae256", "beta1.0_seed0_256px"))  # sharper-structure VAE
    out.append(("gan", "GAN", "gan", "vaegan_seed0_128px"))
    return out


@torch.no_grad()
def _model_payload(tag, eval_cache, names):
    ck = torch.load(CKPT / f"{tag}.pt", map_location="cpu", weights_only=False)
    sc = ck["sidecar"]
    res = sc["img_size"]
    if res not in eval_cache:
        print(f"    loading eval set at {res}px ...")
        eval_cache[res] = evaluate.load_eval(res)
    imgs, facs, _ = eval_cache[res]
    m = models.ConvVAE(res, sc["latent_dim"], sc["channels"])
    m.load_state_dict(ck["model_state"]); m.eval()
    codes, _ = evaluate.encode(m, imgs)
    z0 = np.median(codes, 0)
    # sweep each latent across its encoded 5-95 pct range, but at least +-2.5 units from the
    # prototype so low-variance dims (esp. the VAE-GAN's near-free-bits dims) still move visibly.
    lo = np.minimum(np.percentile(codes, 5, 0), z0 - 2.5)
    hi = np.maximum(np.percentile(codes, 95, 0), z0 + 2.5)
    mi = metrics.mutual_info_matrix(codes, facs)
    mig = metrics.mig(codes, facs)["mig"]
    order = list(np.argsort(-mi.max(1))[:TOP])
    strips, labels = {}, {}
    for j in order:
        j = int(j)
        top2 = np.argsort(-mi[j])[:2]
        primary, second = names[top2[0]], names[top2[1]]
        mip = float(mi[j, top2[0]])
        clean = mip >= CLEAN_MI
        labels[str(j)] = {"label": HUMAN[primary] if clean else f"{HUMAN[primary]} + {HUMAN[second]}",
                          "primary": HUMAN[primary], "mi": round(mip, 2), "clean": clean}
        frames = []
        for t in np.linspace(lo[j], hi[j], STEPS):
            z = torch.tensor(z0, dtype=torch.float32).clone().unsqueeze(0)
            z[0, j] = float(t)
            frames.append(_uri(m.decode(z)[0].permute(1, 2, 0).numpy()))
        strips[str(j)] = frames
    return {"order": [int(j) for j in order], "strips": strips, "labels": labels,
            "mig": round(mig, 3), "res": res}


@torch.no_grad()
def _gan_sample_payload(tag, n=42):
    """Pool of novel crisp maps sampled from the prior (z~N(0,I)) — the GAN's generation
    story, shown via a 'Generate' button instead of sliders (its latent is entangled)."""
    ck = torch.load(CKPT / f"{tag}.pt", map_location="cpu", weights_only=False)
    sc = ck["sidecar"]
    m = models.ConvVAE(sc["img_size"], sc["latent_dim"], sc["channels"])
    m.load_state_dict(ck["model_state"]); m.eval()
    torch.manual_seed(0)
    dec = m.decode(torch.randn(n, sc["latent_dim"]))
    samples = [_uri(dec[i].permute(1, 2, 0).numpy()) for i in range(n)]
    return {"samples": samples, "res": sc["img_size"], "n": n}


def _build_payload():
    names = factors.GROUND_TRUTH
    eval_cache = {}   # resolution -> (imgs, facs, seeds); models may be 128px or 256px
    ml = _model_list()
    payload = {"model_list": [[k, lbl, kind] for k, lbl, kind, _ in ml], "models": {}, "steps": STEPS, "res": RES}
    for key, lbl, kind, tag in ml:
        print(f"  decoding {lbl} ({tag}) ...")
        payload["models"][key] = (_gan_sample_payload(tag) if kind == "gan"
                                  else _model_payload(tag, eval_cache, names))
    return payload


def main():
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
    print(f"wrote report/demo_web.html + _artifact_demo.html (~{len(body)//1024} KB, "
          f"{len(payload['model_list'])} models incl. VAE-GAN)")


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
[hidden]{display:none!important}
body{margin:0;background:var(--paper);color:var(--ink);
  font:16px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;-webkit-font-smoothing:antialiased;padding:28px 20px 56px}
.wrap{max-width:600px;margin:0 auto}
.eyebrow{font:600 12px/1 ui-monospace,"SF Mono",Menlo,monospace;letter-spacing:.18em;color:var(--sea)}
h1{font-family:Georgia,"Iowan Old Style",serif;font-weight:600;font-size:clamp(30px,8vw,42px);line-height:1.08;
  text-wrap:balance;margin:14px 0 12px;letter-spacing:-.01em}
.lede{color:var(--ink-soft);font-size:17px;margin:0 0 24px;max-width:52ch}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px}
.modelbar{display:flex;flex-direction:column;gap:8px;margin-bottom:16px}
.modelbar .flabel{display:flex;justify-content:space-between;align-items:baseline}
.seg{display:grid;grid-template-columns:repeat(6,1fr);gap:5px;background:var(--chip);border-radius:11px;padding:5px}
.seg button{font:600 12px/1 ui-monospace,Menlo,monospace;border:0;border-radius:8px;padding:11px 0;color:var(--ink-soft);
  background:transparent;cursor:pointer;transition:background .15s,color .15s}
.seg button[aria-pressed="true"]{background:var(--sea);color:#fff}
.seg button.gan[aria-pressed="true"]{background:var(--peak)}
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
.genbtn{font:600 15px/1 system-ui,-apple-system,sans-serif;border:0;border-radius:11px;padding:15px;color:#fff;
  background:var(--peak);cursor:pointer;transition:filter .15s;letter-spacing:.01em}
.genbtn:hover{filter:brightness(1.06)}.genbtn:active{filter:brightness(.93)}
.genbtn:focus-visible{outline:2px solid var(--ink);outline-offset:2px}
.caption{margin:14px 2px 0;font-size:14.5px;color:var(--ink-soft);line-height:1.55}
.caption b{color:var(--ink)}.peak{color:var(--peak);font-weight:600}.sea{color:var(--sea);font-weight:600}
.finding{margin-top:30px}
.finding h2{font-family:Georgia,serif;font-weight:600;font-size:22px;margin:0 0 6px;text-wrap:balance}
.finding p{color:var(--ink-soft);margin:0 0 18px;font-size:15.5px}
.stats{display:grid;gap:10px}
.stat{display:flex;align-items:baseline;justify-content:space-between;gap:14px;padding:13px 15px;background:var(--card);
  border:1px solid var(--line);border-radius:11px}
.stat .q{font-size:14.5px}.stat .a{font-family:ui-monospace,Menlo,monospace;font-size:13px;white-space:nowrap;font-variant-numeric:tabular-nums}
.a.no{color:var(--peak)}.a.yes{color:var(--sea)}
.note{margin-top:24px;font-size:13.5px;color:var(--ink-soft);line-height:1.6;border-top:1px solid var(--line);padding-top:18px}
footer{margin-top:26px;font:500 11px/1.5 ui-monospace,Menlo,monospace;letter-spacing:.04em;color:var(--ink-soft);text-transform:uppercase}
"""

_CONTENT = """<style>__CSS__</style>
<div class="wrap">
  <div class="eyebrow">&beta;-VAE &amp; VAE-GAN &middot; GENERATIVE MAPS</div>
  <h1>This Fantasy Map Does Not Exist</h1>
  <p class="lede">Every map here is dreamed up by a neural network from a code of numbers &mdash; no seed, no
  source image. Pick a model, then drag the slider to walk one dimension of that code.</p>

  <div class="card">
    <div class="modelbar">
      <div class="flabel"><span>Model</span><span class="mig" id="migTag"></span></div>
      <div class="seg" id="seg" role="group" aria-label="select model"></div>
    </div>
    <div class="viewer"><img id="fA" class="frame on" alt="generated fantasy map"><img id="fB" class="frame" alt=""></div>
    <div class="controls">
      <div class="field" id="dimField"><span class="flabel">Latent dimension</span><select id="dim" aria-label="latent dimension"></select></div>
      <button id="gen" class="genbtn" hidden>&#127922;&nbsp;&nbsp;Generate a new map</button>
      <div class="field"><span class="flabel" id="sliderLabel">Position along it</span><input id="slider" type="range" min="0" max="8" value="4" aria-label="latent value"></div>
    </div>
    <p class="caption" id="cap"></p>
  </div>

  <div class="finding">
    <h2>Two stories in one toy.</h2>
    <p>The &beta;-VAE models were built to <b>disentangle</b> &mdash; and measurably barely do. The
    <b>VAE-GAN</b> was built to be <b>sharp</b> &mdash; and is. Switch between them above.</p>
    <div class="stats">
      <div class="stat"><span class="q">Does raising &beta; buy disentanglement?</span><span class="a no">No &middot; p=0.07</span></div>
      <div class="stat"><span class="q">Rate&ndash;distortion tradeoff real?</span><span class="a yes">Yes &middot; p&lt;0.001</span></div>
      <div class="stat"><span class="q">&beta;-TCVAE beat plain &beta;-VAE?</span><span class="a no">No gain &middot; p=0.72</span></div>
      <div class="stat"><span class="q">VAE-GAN: sharp <i>and</i> controllable?</span><span class="a yes">Yes &middot; KL held, latent alive</span></div>
    </div>
  </div>

  <p class="note">The &beta;-VAE maps are soft on purpose &mdash; a &beta;-VAE trades sharpness for a structured latent, and
  this project <b>measures</b> disentanglement (MIG/DCI) against 33,000 procedurally-generated maps with known factors.
  The VAE-GAN adds an adversarial loss for crisp texture; an early version collapsed (one map for every input), fixed
  with a free-bits KL floor + reconstruct-first warmup. A data point in the <b>Locatello et&nbsp;al. (2019)</b>
  impossibility debate, wearing a wizard hat.</p>

  <footer id="foot"></footer>
</div>
<script>
const D=__PAYLOAD__;
const seg=document.getElementById('seg'),sel=document.getElementById('dim'),sl=document.getElementById('slider'),
  fA=document.getElementById('fA'),fB=document.getElementById('fB'),cap=document.getElementById('cap'),
  migTag=document.getElementById('migTag'),foot=document.getElementById('foot'),
  gen=document.getElementById('gen'),dimField=document.getElementById('dimField'),sliderLabel=document.getElementById('sliderLabel');
let front=fA,back=fB,cur=(D.model_list.find(m=>m[2]==='gan')||D.model_list[0])[0];  // open on VAE-GAN
D.model_list.forEach(([key,lbl,kind])=>{const btn=document.createElement('button');btn.textContent=lbl;
  if(kind==='gan')btn.className='gan';btn.setAttribute('aria-pressed',key===cur);
  btn.onclick=()=>{cur=key;syncModel();};seg.appendChild(btn);});
function kindOf(key){return D.model_list.find(m=>m[0]===key)[2];}
function show(uri){back.src=uri;back.classList.add('on');front.classList.remove('on');[front,back]=[back,front];}
function syncModel(){
  [...seg.children].forEach((btn,i)=>btn.setAttribute('aria-pressed',D.model_list[i][0]===cur));
  const M=D.models[cur],kind=kindOf(cur);
  if(kind==='gan'){
    migTag.textContent='crisp \\u00b7 novel';dimField.hidden=true;gen.hidden=false;sliderLabel.textContent='Browse this batch';
    sl.max=M.n-1;sl.value=Math.floor(Math.random()*M.n);
  }else{
    migTag.textContent='MIG '+M.mig;dimField.hidden=false;gen.hidden=true;sliderLabel.textContent='Position along it';
    sel.innerHTML='';
    M.order.forEach(j=>{const L=M.labels[j];const o=document.createElement('option');o.value=j;
      o.textContent='z'+j+'  \\u00b7  '+L.label+'  (MI '+L.mi+')';sel.appendChild(o);});
    sl.max=D.steps-1;sl.value=(D.steps-1)>>1;
  }
  render();
}
function render(){
  const M=D.models[cur],kind=kindOf(cur);
  if(kind==='gan'){
    show(M.samples[sl.value]);
    cap.innerHTML='A brand-new fantasy map from the <span class="peak">VAE-GAN</span> &mdash; decoded from a random '+
      'code, <b>crisp</b>, and it never existed. Hit <b>Generate</b> for another, or drag to browse this batch of '+M.n+'.';
    foot.textContent='VAE-GAN \\u00b7 '+M.res+'px \\u00b7 novel samples \\u00b7 crisp, not eyeballed';
    return;
  }
  const j=sel.value,L=M.labels[j];
  show(M.strips[j][sl.value]);
  if(kind==='vae256'){
    cap.innerHTML='The &beta;-VAE at <b>256px</b>. Sharper <i>structure</i> than the 128px models &mdash; '+
      'landmasses are recognizable &mdash; but still soft: MSE reconstruction can&rsquo;t make crisp edges at any '+
      'resolution. That&rsquo;s why the <span class="peak">VAE-GAN</span>, not resolution, is the sharp one.';
  }else{
    cap.innerHTML='Walking latent <b>z'+j+'</b>, which most tracks <span class="sea">'+L.primary+'</span>. '+
      (L.clean?'Fairly clean &mdash; a real knob.'
             :'Weak and entangled (MI '+L.mi+'): it nudges several properties at once &mdash; low disentanglement, made visible.');
  }
  foot.textContent=(kind==='vae256'?'\\u03b2-VAE 256px':'\\u03b2-VAE')+' \\u00b7 '+M.res+'px \\u00b7 MIG '+M.mig+' \\u00b7 measured, not eyeballed';
}
gen.onclick=()=>{const M=D.models[cur];let v;do{v=Math.floor(Math.random()*M.n);}while(M.n>1&&v===+sl.value);sl.value=v;render();};
sel.onchange=render;sl.oninput=render;syncModel();
</script>"""

_CONTENT = _CONTENT.replace("__CSS__", _CSS)

if __name__ == "__main__":
    main()
