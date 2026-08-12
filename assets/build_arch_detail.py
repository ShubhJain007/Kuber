#!/usr/bin/env python3
"""Compose a detailed, transformer-paper-style architecture figure: real input geometry ->
expanded network modules (each showing its internal layers) -> real predicted field.
Emits HTML screenshotted to assets/sim/architecture.png. Usage: <panel_dir> <out_html>
"""
import base64, os, sys

PDIR = sys.argv[1] if len(sys.argv) > 1 else "."
OUT_HTML = sys.argv[2] if len(sys.argv) > 2 else "arch.html"


def uri(name):
    with open(os.path.join(PDIR, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def arr():
    return ('<div class="arr"><svg width="46" height="16" viewBox="0 0 46 16">'
            '<path d="M0 8 H38" stroke="#94A3B8" stroke-width="2.4"/>'
            '<path d="M38 2 L46 8 L38 14 Z" fill="#94A3B8"/></svg></div>')


CSS = """
*{box-sizing:border-box;margin:0}
body{background:#fff;font-family:system-ui,-apple-system,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
.fig{width:2320px;padding:34px 40px 40px;background:#fff}
.hd h1{font-size:30px;font-weight:800;letter-spacing:-.02em;color:#0B1220}
.hd p{font-size:16px;color:#5B6672;margin:2px 0 0 2px}
.row{display:flex;align-items:center;gap:4px;margin-top:16px}
.card{background:#fff;border:1px solid #E3E8EE;border-radius:16px;padding:14px;box-shadow:0 2px 8px rgba(16,24,40,.06);width:360px;flex:0 0 auto}
.card.out{width:392px}
.card img{width:100%;height:250px;object-fit:contain;display:block}
.cap{font-size:15px;font-weight:700;color:#14181F;margin-top:6px;text-align:center}
.sub{font-size:12px;color:#5B6672;text-align:center;margin-top:1px}
.chips{display:flex;flex-direction:column;gap:6px;margin-top:10px}
.chip{font-size:12px;background:#EEF2F6;color:#2A3746;border:1px solid #E3E8EE;border-radius:8px;padding:6px 10px;text-align:center}
.chip b{color:#1F4E79}
.arr{display:flex;align-items:center;flex:0 0 auto;padding:0 2px}
.mod{background:#EAF1F8;border:1.6px solid #C7DAEC;border-radius:16px;padding:12px 12px 10px;display:flex;flex-direction:column;gap:6px;min-width:216px;flex:0 0 auto;align-self:stretch;justify-content:center}
.mh{display:flex;align-items:center;justify-content:space-between;gap:8px;margin-bottom:3px}
.mh b{color:#123A5E;font-size:15.5px;font-weight:800;line-height:1.08}
.xn{background:#1F4E79;color:#fff;font-size:11px;font-weight:700;border-radius:20px;padding:2px 9px;white-space:nowrap}
.lay{background:#fff;border:1px solid #DCE6F0;border-radius:9px;padding:7px 10px;font-size:12.5px;color:#26303C;text-align:center;line-height:1.22}
.lay small{color:#6B7887;font-size:11px}
.lay.attn{border-color:#E7B08A;background:#FFF6F0;font-weight:650;color:#7A3A12}
.lay.attn small{color:#B4632F}
.lay.norm{background:#F1F5F9;color:#64748B;font-size:11.5px;padding:5px 10px;border-color:#E3E8EE}
.lay.head{background:#1D4E7C;color:#fff;border-color:#1D4E7C;font-weight:650}
.lay.head small{color:#C3D6EA}
.io{font-size:11.5px;color:#5B6672;text-align:center;margin-top:3px}
.io b{color:#1F4E79}
"""

HTML = """<meta charset="utf-8"><style>__CSS__</style>
<div class="fig">
  <div class="hd"><h1>SurfaceGeoTransolver</h1>
  <p>Geometry-general conjugate-heat-transfer surrogate &mdash; internal layers shown</p></div>
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

    __ARR__

    <div class="mod">
      <div class="mh"><b>Surface<br>encoder</b><span class="xn">&times; 2</span></div>
      <div class="lay">Input embed <small>pts &oplus; normals &rarr; 128</small></div>
      <div class="lay attn">Multi-Head Self-Attention <small>4 heads</small></div>
      <div class="lay norm">Add &amp; Norm</div>
      <div class="lay">Feed-Forward <small>MLP</small></div>
      <div class="lay norm">Add &amp; Norm</div>
      <div class="io">geometry tokens <b>[N<sub>s</sub>&times;128]</b></div>
    </div>

    __ARR__

    <div class="mod">
      <div class="mh"><b>Local<br>cross-attention</b></div>
      <div class="lay">kNN gather <small>k = 16 surface tokens</small></div>
      <div class="lay attn">Cross-Attention <small>4 heads &middot; Q: node, KV: tokens</small></div>
      <div class="io">descriptor <b>[N&times;64]</b> &nbsp;&oplus;&nbsp; conditioning</div>
    </div>

    __ARR__

    <div class="mod">
      <div class="mh"><b>GeoTransolver<br>core</b><span class="xn">&times; 12</span></div>
      <div class="lay attn">Physics Attention <small>64 learned slices</small></div>
      <div class="lay norm">Add &amp; Norm</div>
      <div class="lay">Multiscale ball-query <small>r = 0.05 / 0.25</small></div>
      <div class="lay">Feed-Forward <small>256 hidden</small></div>
      <div class="lay norm">Add &amp; Norm</div>
      <div class="lay head">Linear head <small>&rarr; 5 channels</small></div>
      <div class="io">field <b>[N&times;5]</b></div>
    </div>

    __ARR__

    <div class="card out">
      <img src="__OUT__" alt="predicted field">
      <div class="cap">Predicted field</div>
      <div class="sub">(U<sub>x</sub>, U<sub>y</sub>, U<sub>z</sub>, T, p<sub>rgh</sub>) at every query point</div>
    </div>

  </div>
</div>"""

out = (HTML.replace("__CSS__", CSS)
       .replace("__IN__", uri("arch_input.png"))
       .replace("__OUT__", uri("arch_output.png"))
       .replace("__ARR__", arr()))
open(OUT_HTML, "w").write(out)
print("wrote", OUT_HTML)
