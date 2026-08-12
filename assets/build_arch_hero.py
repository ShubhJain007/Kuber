#!/usr/bin/env python3
"""Compose the SurfaceGeoTransolver hero architecture figure (paper-style):
real input geometry -> network blocks -> real predicted field. Emits an HTML that is
screenshotted to assets/fig_architecture.png. Usage: build_arch_hero.py <panel_dir> <out_html>
"""
import base64, os, sys

PDIR = sys.argv[1] if len(sys.argv) > 1 else "."
OUT_HTML = sys.argv[2] if len(sys.argv) > 2 else "arch.html"


def uri(name):
    with open(os.path.join(PDIR, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


HTML = """<meta charset="utf-8"><style>
*{box-sizing:border-box;margin:0}
body{background:#fff;font-family:system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
.fig{width:2160px;padding:34px 40px 40px;background:#fff}
.hd{margin:0 0 6px 4px}
.hd h1{font-size:30px;font-weight:800;letter-spacing:-.02em;color:#0B1220}
.hd p{font-size:16px;color:#5B6672;margin-top:2px}
.row{display:flex;align-items:center;gap:6px;margin-top:14px}
.card{background:#fff;border:1px solid #E3E8EE;border-radius:16px;padding:14px;box-shadow:0 2px 8px rgba(16,24,40,.06);width:400px;flex:0 0 auto}
.card.out{width:440px}
.card img{width:100%;height:280px;object-fit:contain;display:block}
.cap{font-size:15px;font-weight:700;color:#14181F;margin-top:6px;text-align:center}
.sub{font-size:12.5px;color:#5B6672;text-align:center;margin-top:1px}
.chips{display:flex;flex-direction:column;gap:6px;margin-top:10px}
.chip{font-size:12.5px;background:#EEF2F6;color:#2A3746;border:1px solid #E3E8EE;border-radius:8px;padding:6px 10px;text-align:center}
.chip b{color:#1F4E79}
.arr{display:flex;flex-direction:column;align-items:center;flex:0 0 auto;min-width:66px}
.arr .lab{font-size:12px;color:#5B6672;margin-bottom:3px;white-space:nowrap;text-align:center;line-height:1.2}
.arr svg{display:block}
.blk{background:#1F4E79;color:#fff;border-radius:14px;padding:18px 16px;text-align:center;flex:0 0 auto;min-width:132px;box-shadow:0 2px 8px rgba(31,78,121,.28)}
.blk .bt{font-size:17px;font-weight:750;line-height:1.12}
.blk .bs{font-size:12px;opacity:.85;margin-top:5px;line-height:1.25}
.blk.core{padding-top:12px;min-width:212px;background:#1D4E7C}
.stack{display:flex;flex-direction:column;gap:3px;margin:0 auto 9px;width:120px}
.stack span{height:6px;border-radius:3px;background:rgba(255,255,255,.35)}
.stack span:nth-child(2){background:rgba(255,255,255,.55)}
.stack span:nth-child(3){background:rgba(255,255,255,.75)}
.mult{display:inline-block;margin-top:8px;font-size:12px;font-weight:700;background:rgba(255,255,255,.16);border-radius:20px;padding:2px 10px}
.tokens{display:flex;gap:4px;justify-content:center;margin-bottom:8px}
.tokens i{width:9px;height:9px;border-radius:2px;background:rgba(255,255,255,.6)}
</style>
<div class="fig">
  <div class="hd"><h1>SurfaceGeoTransolver</h1>
  <p>Geometry-general conjugate-heat-transfer surrogate &mdash; predicts the full field at every query point</p></div>
  <div class="row">

    <div class="card">
      <img src="__IN__" alt="surface geometry">
      <div class="cap">Input geometry</div>
      <div class="sub">surface point cloud + outward normals &nbsp;[N<sub>s</sub>&times;3]</div>
      <div class="chips">
        <div class="chip">Query points &nbsp;<b>[N&times;3]</b></div>
        <div class="chip">Physics conditioning &nbsp;<b>&rho; &mu; C<sub>p</sub> Pr u<sub>in</sub> BC device</b></div>
      </div>
    </div>

    __ARR__ARROW__

    <div class="blk"><div class="tokens"><i></i><i></i><i></i><i></i></div>
      <div class="bt">Surface<br>encoder</div><div class="bs">self-attention</div></div>

    __ARR__geometry tokens__

    <div class="blk"><div class="bt">Local kNN<br>cross-attention</div><div class="bs">k = 16</div></div>

    __ARR__descriptor &oplus; conditioning__

    <div class="blk core">
      <div class="stack"><span></span><span></span><span></span></div>
      <div class="bt">GeoTransolver core</div>
      <div class="bs">physics attention &middot; 256 hidden<br>64 slices &middot; multiscale ball-query</div>
      <div class="mult">&times; 12 layers</div>
    </div>

    __ARR__field [N&times;5]__

    <div class="card out">
      <img src="__OUT__" alt="predicted field">
      <div class="cap">Predicted field</div>
      <div class="sub">(U<sub>x</sub>, U<sub>y</sub>, U<sub>z</sub>, T, p<sub>rgh</sub>) at every query point</div>
    </div>

  </div>
</div>"""


def arrow(label):
    lab = "" if label == "ARROW" else f'<span class="lab">{label}</span>'
    svg = ('<svg width="52" height="16" viewBox="0 0 52 16">'
           '<path d="M0 8 H44" stroke="#94A3B8" stroke-width="2.4"/>'
           '<path d="M44 2 L52 8 L44 14 Z" fill="#94A3B8"/></svg>')
    return f'<div class="arr">{lab}{svg}</div>'


out = HTML.replace("__IN__", uri("arch_input.png")).replace("__OUT__", uri("arch_output.png"))
while "__ARR__" in out:
    i = out.index("__ARR__")
    j = out.index("__", i + 7)
    label = out[i + 7:j]
    out = out[:i] + arrow(label) + out[j + 2:]

open(OUT_HTML, "w").write(out)
print("wrote", OUT_HTML)
