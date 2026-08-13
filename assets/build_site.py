#!/usr/bin/env python3
"""Build the Kuber project page: a professional, self-contained technical landing page.

Embeds the simulation PNGs as base64 and inlines the SVG figures + architecture diagram,
so the output is a single portable HTML file. Writes site/index.html and, if a path arg is
given, an artifact-body copy.
"""
import base64, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SIM = os.path.join(HERE, "sim")


def png_uri(name):
    with open(os.path.join(SIM, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def svg_inline(name):
    s = open(os.path.join(HERE, name)).read()
    s = re.sub(r'\swidth="[\d.]+"', '', s, count=1)
    s = re.sub(r'\sheight="[\d.]+"', '', s, count=1)
    s = s.replace("<svg ", '<svg style="width:100%;height:auto;display:block" ', 1)
    return s


STYLE = r"""
*,*::before,*::after{box-sizing:border-box}
:root{
  --ground:#FBFBFC; --surface:#FFFFFF; --card:#FFFFFF; --ink:#0B1220; --ink2:#26303C; --muted:#5B6672;
  --line:#E6EAEF; --accent:#1F4E79; --accent2:#2F6EA8; --ember:#C2410C; --shadow:0 1px 2px rgba(16,24,40,.06);
  --navbg:rgba(251,251,252,.82);
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --ground:#0B0E13; --surface:#12161D; --card:#FFFFFF; --ink:#EAEEF3; --ink2:#C4CDD8; --muted:#93A0AE;
  --line:#202832; --accent:#7CB2E8; --accent2:#9AC6F0; --ember:#F0863C; --shadow:0 1px 2px rgba(0,0,0,.4);
  --navbg:rgba(11,14,19,.78);
}}
:root[data-theme="dark"]{
  --ground:#0B0E13; --surface:#12161D; --card:#FFFFFF; --ink:#EAEEF3; --ink2:#C4CDD8; --muted:#93A0AE;
  --line:#202832; --accent:#7CB2E8; --accent2:#9AC6F0; --ember:#F0863C; --shadow:0 1px 2px rgba(0,0,0,.4);
  --navbg:rgba(11,14,19,.78);
}
html{-webkit-text-size-adjust:100%; scroll-behavior:smooth}
body{margin:0; background:var(--ground); color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  font-size:16.5px; line-height:1.6; -webkit-font-smoothing:antialiased; letter-spacing:-0.006em}
a{color:var(--accent); text-decoration:none}
.wrap{max-width:1080px; margin:0 auto; padding:0 24px}

/* nav */
.nav{position:sticky; top:0; z-index:50; background:var(--navbg); backdrop-filter:saturate(1.4) blur(10px);
  border-bottom:1px solid var(--line)}
.nav-in{max-width:1080px; margin:0 auto; padding:11px 24px; display:flex; align-items:center; gap:20px}
.brand{display:flex; align-items:center; gap:9px; font-weight:750; font-size:1.06rem; color:var(--ink); letter-spacing:-.02em}
.mark{width:18px; height:18px; border-radius:5px; background:linear-gradient(135deg,var(--ember),var(--accent))}
.nav-links{margin-left:auto; display:flex; align-items:center; gap:20px; font-size:.93rem}
.nav-links a{color:var(--ink2); font-weight:500}
.nav-links a:hover{color:var(--accent)}
.nav-links .btn-sm{border:1px solid var(--line); border-radius:8px; padding:.34em .8em; color:var(--ink)}
.nav-links .btn-sm:hover{border-color:var(--accent)}
@media (max-width:720px){.nav-links a:not(.btn-sm){display:none}}

/* hero */
.hero{text-align:center; padding:76px 0 30px}
.eyebrow{font-size:.79rem; font-weight:650; letter-spacing:.14em; text-transform:uppercase; color:var(--accent)}
.hero h1{font-size:clamp(2.3rem,5.4vw,3.75rem); line-height:1.04; font-weight:800; letter-spacing:-.032em;
  margin:.34em auto .28em; max-width:16ch; text-wrap:balance}
.lede{font-size:clamp(1.05rem,2.2vw,1.28rem); color:var(--ink2); max-width:60ch; margin:0 auto 1.7em; text-wrap:balance}
.cta{display:flex; gap:12px; justify-content:center; flex-wrap:wrap; margin-bottom:16px}
.btn{display:inline-flex; align-items:center; gap:.4em; padding:.66em 1.25em; border-radius:10px; font-weight:600;
  font-size:.98rem; border:1px solid var(--line); background:var(--surface); color:var(--ink); transition:.15s}
.btn:hover{border-color:var(--accent); transform:translateY(-1px)}
.btn.primary{background:var(--accent); border-color:var(--accent); color:#fff}
:root[data-theme="dark"] .btn.primary, :root:not([data-theme="light"]) .btn.primary{color:#08121d}
.btn.primary:hover{filter:brightness(1.06); color:#fff}
:root[data-theme="dark"] .btn.primary:hover{color:#08121d}

/* stat tiles */
.stats{display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:38px 0 8px}
.stat{border:1px solid var(--line); border-radius:14px; background:var(--surface); padding:18px 16px; text-align:left; box-shadow:var(--shadow)}
.stat .num{font-size:1.72rem; font-weight:800; letter-spacing:-.03em; color:var(--accent); font-variant-numeric:tabular-nums}
.stat .lbl{font-size:.82rem; color:var(--muted); line-height:1.35; margin-top:4px}
@media (max-width:820px){.stats{grid-template-columns:repeat(2,1fr)}}
@media (max-width:460px){.stats{grid-template-columns:1fr}}

/* sections */
section{padding:52px 0; border-top:1px solid var(--line)}
section:first-of-type{border-top:0}
h2{font-size:clamp(1.6rem,3vw,2.1rem); font-weight:780; letter-spacing:-.028em; margin:.22em 0 .3em; text-wrap:balance}
.section-lede{font-size:1.08rem; color:var(--ink2); max-width:66ch; margin:0 0 8px}
p{margin:0 0 1em; max-width:70ch; color:var(--ink2)}

/* figures — always-light cards */
figure.fig{margin:22px 0 0; background:var(--card); border:1px solid #E6EAEF; border-radius:16px;
  padding:16px 16px 6px; box-shadow:0 1px 3px rgba(16,24,40,.06)}
figure.fig.tight{padding:12px}
figure.fig img{width:100%; height:auto; display:block; border-radius:8px}
figure.fig svg{width:100%; height:auto; display:block}
figcaption{font-size:.88rem; color:#5B6672; padding:12px 4px 8px; line-height:1.5; border-top:1px solid #EEF1F5; margin-top:10px}
a.fig-link{display:block; color:inherit}
a.fig-link:hover figure.fig{border-color:var(--accent)}
a.fig-link figcaption{color:var(--accent); font-weight:600}
.grid2{display:grid; grid-template-columns:1fr 1fr; gap:20px}
.grid2 figure.fig{margin-top:0}
@media (max-width:760px){.grid2{grid-template-columns:1fr}}

/* footer */
footer{border-top:1px solid var(--line); padding:30px 0 60px; color:var(--muted); font-size:.9rem}
.foot-in{max-width:1080px; margin:0 auto; padding:0 24px; display:flex; flex-wrap:wrap; gap:12px; justify-content:space-between}
.foot-in a{color:var(--ink2)}
@media (prefers-reduced-motion:reduce){*{transition:none!important; scroll-behavior:auto}}
"""

CONTENT = """
<header class="nav"><div class="nav-in">
  <a class="brand" href="#top"><span class="mark"></span>Kuber</a>
  <nav class="nav-links">
    <a href="#how">How it works</a>
    <a href="#results">Results</a>
    <a href="#dataset">Dataset</a>
    <a href="demo.html">Demo</a>
    <a href="https://github.com/ShubhJain007/Kuber/blob/main/paper/kuber.pdf">Paper</a>
    <a class="btn-sm" href="https://github.com/ShubhJain007/Kuber">GitHub</a>
  </nav>
</div></header>

<main class="wrap">
  <section class="hero" id="top">
    <div class="eyebrow">Open framework &middot; Conjugate heat transfer</div>
    <h1>Neural surrogates for coupled fluid&ndash;heat simulation</h1>
    <p class="lede">Kuber predicts the full temperature and flow field of a heatsink or cold plate in a
    sub-second &mdash; geometry-general, and state of the art on a public benchmark with no domain
    adaptation.</p>
    <div class="cta">
      <a class="btn primary" href="demo.html">Interactive demo &rarr;</a>
      <a class="btn" href="https://github.com/ShubhJain007/Kuber/blob/main/paper/kuber.pdf">Read the paper</a>
      <a class="btn" href="https://github.com/ShubhJain007/Kuber">GitHub</a>
    </div>
    <div class="stats">
      <div class="stat"><div class="num">12.14&thinsp;K</div><div class="lbl">temperature RMSE on SIMSHIFT &mdash; beats the prior best, no domain adaptation</div></div>
      <div class="stat"><div class="num">10,000&times;</div><div class="lbl">up to &mdash; faster than the CFD it learns from</div></div>
      <div class="stat"><div class="num">2</div><div class="lbl">device classes &mdash; heatsinks &amp; cold plates &mdash; from one model</div></div>
      <div class="stat"><div class="num">~14&thinsp;M</div><div class="lbl">parameters, geometry-general (arbitrary CAD)</div></div>
    </div>
    <figure class="fig">__CMPHS__<figcaption>Heatsink &mdash; Kuber prediction vs. CFD ground truth (&plusmn;2.11&thinsp;K temperature agreement).</figcaption></figure>
    <figure class="fig">__CMPCP__<figcaption>Cold plate &mdash; Kuber prediction vs. CFD ground truth (&plusmn;1.33&thinsp;K, 445&times; faster than the CFD solver).</figcaption></figure>
  </section>

  <section id="how">
    <div class="eyebrow">How it works</div>
    <h2>One transformer, from geometry to field</h2>
    <p class="section-lede">SurfaceGeoTransolver ingests raw boundary geometry &mdash; a surface point cloud with
    normals &mdash; plus physics conditioning, and predicts <em>(U, T, p)</em> at any query point. No analytic
    signed-distance field is required, so it applies to arbitrary CAD.</p>
    <figure class="fig tight">__ARCH__<figcaption>The surface encoder turns the boundary into geometry tokens;
    local kNN cross-attention gives every query node a geometry descriptor; concatenated with the physics
    conditioning it feeds a GeoTransolver physics-attention core (~14&thinsp;M parameters).</figcaption></figure>
  </section>

  <section id="results">
    <div class="eyebrow">Results</div>
    <h2>State of the art, without domain adaptation</h2>
    <p class="section-lede">On the public SIMSHIFT heatsink split (train fin counts 5&ndash;8 &rarr; test 10&ndash;12),
    Kuber leads on temperature &mdash; with none of the unsupervised domain adaptation every published baseline uses.
    Every number is measured and reproducible.</p>
    <figure class="fig">__LEADERBOARD__<figcaption>SIMSHIFT heatsink leaderboard (medium / out-of-distribution).
    Baselines include UDA; Kuber uses none. Lower is better.</figcaption></figure>
    <div class="grid2">
      <figure class="fig">__INDIST__<figcaption>The leading number is zero-shot: the test fin counts never appear in training.</figcaption></figure>
      <figure class="fig">__STABILITY__<figcaption>Predicted &nabla;T at fin tips/corners stays at or below physical &mdash; no gradient explosion.</figcaption></figure>
      <figure class="fig">__VALUE__<figcaption>Pretraining on the self-generated corpus lowers error further.</figcaption></figure>
      <figure class="fig">__SPEED__<figcaption>Sub-second inference vs. a median 22-minute CFD solve (log scale).</figcaption></figure>
    </div>
  </section>

  <section id="multigeo">
    <div class="eyebrow">One model, two device classes</div>
    <h2>Heatsinks and cold plates from one set of weights</h2>
    <p class="section-lede">Distinguished only by a device flag and fluid/BC conditioning, a single model spans a
    buoyancy-driven heatsink in air and a forced-liquid cold plate. Held-out cold plates 3.11&thinsp;K, heatsinks
    5.13&thinsp;K on our corpus (not comparable to the SIMSHIFT numbers).</p>
    <figure class="fig">__MULTIGEO__<figcaption>One SurfaceGeoTransolver, evaluated per device class on held-out cases. Per-device prediction-vs-ground-truth comparisons are at the top; explore the <a href="demo.html">interactive viewer</a>.</figcaption></figure>
  </section>

  <section id="dataset">
    <div class="eyebrow">Dataset</div>
    <h2>A self-generated OpenFOAM corpus</h2>
    <p class="section-lede">Parametric geometry &rarr; OpenFOAM <code>buoyantSimpleFoam</code> &rarr; per-node fields,
    resumable and convergence-gated. Zero cases from SIMSHIFT or any licensed source.</p>
    <figure class="fig">__CORPUS__<figcaption>Corpus coverage &mdash; fluids, regimes, shapes, device classes.</figcaption></figure>
  </section>
</main>

<footer><div class="foot-in">
  <div>Kuber &mdash; an open framework for conjugate&ndash;heat-transfer AI. Built by the Kuber.ai team.</div>
  <div><a href="https://github.com/ShubhJain007/Kuber">GitHub</a> &middot; <a href="demo.html">Demo</a> &middot;
  <a href="https://github.com/ShubhJain007/Kuber/blob/main/paper/kuber.pdf">Paper</a> &middot; PolyForm Noncommercial 1.0.0</div>
</div></footer>
"""

CONTENT = (CONTENT
           .replace("__CMPHS__", '<img src="' + png_uri("heat-sink-comparison.png") + '" alt="Heatsink — Kuber prediction vs CFD ground truth">')
           .replace("__CMPCP__", '<img src="' + png_uri("cold-plate-comparison.png") + '" alt="Cold plate — Kuber prediction vs CFD ground truth">')
           .replace("__ARCH__", '<img src="' + png_uri("architecture.png") + '" alt="SurfaceGeoTransolver architecture">')
           .replace("__LEADERBOARD__", svg_inline("fig_leaderboard.svg"))
           .replace("__INDIST__", svg_inline("fig_indist_vs_ood.svg"))
           .replace("__STABILITY__", svg_inline("fig_stability.svg"))
           .replace("__VALUE__", svg_inline("fig_value_of_data.svg"))
           .replace("__SPEED__", svg_inline("fig_speed.svg"))
           .replace("__CORPUS__", svg_inline("fig_corpus.svg"))
           .replace("__MULTIGEO__", svg_inline("fig_multigeo.svg")))

TITLE = "Kuber"
full = (f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{TITLE}</title>\n<style>{STYLE}</style>\n</head>\n<body>\n{CONTENT}\n</body>\n</html>\n")
body_only = f"<title>{TITLE}</title>\n<style>{STYLE}</style>\n{CONTENT}\n"

os.makedirs(os.path.join(ROOT, "site"), exist_ok=True)
open(os.path.join(ROOT, "site", "index.html"), "w").write(full)
if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(body_only)
print("wrote site/index.html", f"({len(full)//1024} KB)")
