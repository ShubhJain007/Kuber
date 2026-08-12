#!/usr/bin/env python3
"""Render clean input/output panels for the SurfaceGeoTransolver hero architecture figure.

  arch_input.png   surface point cloud + a few outward normals (the model's geometry input)
  arch_output.png  the predicted temperature field (colored point cloud) with a colorbar

Run in a matplotlib+numpy env; output dir is argv[1] (default: assets/sim).
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

HERE = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(os.path.dirname(HERE), "data_sample")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "sim")
os.makedirs(OUT, exist_ok=True)

f = sorted(p for p in os.listdir(DS) if p.startswith("bsf_out_ov_air_natural") and p.endswith(".npz"))[0]
d = np.load(os.path.join(DS, f), allow_pickle=True)
coords = np.asarray(d["coords"], float)
T = np.asarray(d["T"], float).ravel()
sp = np.asarray(d["surf_pts"], float)
sn = np.asarray(d["surf_normals"], float)
ELEV, AZIM = 20, -60


def clean(ax, pts):
    ax.set_axis_off()
    ext = pts.max(0) - pts.min(0)
    ax.set_box_aspect(tuple(ext / ext.max()))
    ax.view_init(elev=ELEV, azim=AZIM)


# ---- INPUT: surface point cloud + normals ----
fig = plt.figure(figsize=(6.4, 6.0))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], c="#3E5C78", s=7, alpha=0.85, linewidths=0, depthshade=True)
rng = np.random.default_rng(0)
idx = rng.choice(len(sp), 110, replace=False)
L = (sp.max(0) - sp.min(0)).max() * 0.06
ax.quiver(sp[idx, 0], sp[idx, 1], sp[idx, 2], sn[idx, 0], sn[idx, 1], sn[idx, 2],
          length=L, color="#C2410C", linewidth=0.8, normalize=True)
clean(ax, sp)
fig.subplots_adjust(0, 0, 1, 1)
fig.savefig(os.path.join(OUT, "arch_input.png"), dpi=170, transparent=True)
plt.close(fig)
print("wrote arch_input.png")

# ---- OUTPUT: predicted temperature field + colorbar ----
fig = plt.figure(figsize=(6.8, 6.0))
ax = fig.add_subplot(111, projection="3d")
lo, hi = np.percentile(T, [2, 98])
norm = np.clip((T - lo) / (hi - lo + 1e-9), 0, 1)
colors = plt.get_cmap("inferno")(norm)
colors[:, 3] = 0.12 + 0.8 * norm
order = np.argsort(norm)
ax.scatter(coords[order, 0], coords[order, 1], coords[order, 2], c=colors[order], s=5, linewidths=0, depthshade=False)
clean(ax, coords)
sm = cm.ScalarMappable(cmap="inferno", norm=Normalize(lo, hi))
sm.set_array([])
cb = fig.colorbar(sm, ax=ax, shrink=0.5, pad=0.0, aspect=16)
cb.set_label("T [K]", fontsize=12)
cb.ax.tick_params(labelsize=9)
fig.subplots_adjust(0, 0, 0.9, 1)
fig.savefig(os.path.join(OUT, "arch_output.png"), dpi=170, transparent=True)
plt.close(fig)
print("wrote arch_output.png  (T range %.0f-%.0f K)" % (T.min(), T.max()))
