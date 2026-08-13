"""Analytic surface geometry for the SIMSHIFT parametric heatsink.

AB-UPT-style geometry input: a SURFACE point cloud + outward NORMALS (no mesh
connectivity), generated analytically from the heatsink parameters — so it works at
any resolution with no raw CFD data. Replaces the scalar SDF with an explicit boundary
representation the model can encode and cross-attend to.

Geometry (physical coords, from metadata): base plate L(x) x W(y) x height1(z) centered
at (0,0), sitting on z=0; `fins` fins of thickness_fins along y, gap between, running the
full length x, height2 tall, on top of the base (z in [height1, height1+height2]).

Fluid-facing solid faces sampled (the base BOTTOM at z=0 is the mounting face, excluded):
  base: top + 4 sides ;  each fin: top + 2 long sides (gap-facing) + 2 ends.
"""
from __future__ import annotations

import numpy as np


def _faces(cond):
    """Return list of (origin[3], u[3], v[3], normal[3]) rectangles for the solid boundary.
    A face is sampled as origin + a*u + b*v for a,b ~ U(0,1); `normal` is the outward unit."""
    L, W = float(cond["length"]), float(cond["width"])
    t, g, f = float(cond["thickness_fins"]), float(cond["gap"]), int(cond["fins"])
    h1, h2 = float(cond["height1"]), float(cond["height2"])
    F = []
    x0, y0 = -L / 2, -W / 2

    # --- base plate (z in [0, h1]); bottom face z=0 excluded (mounting/heat source) ---
    F.append(((x0, y0, h1), (L, 0, 0), (0, W, 0), (0, 0, 1)))        # top
    F.append(((x0, y0, 0), (0, W, 0), (0, 0, h1), (-1, 0, 0)))       # x- side
    F.append((( L / 2, y0, 0), (0, W, 0), (0, 0, h1), (1, 0, 0)))    # x+ side
    F.append(((x0, y0, 0), (L, 0, 0), (0, 0, h1), (0, -1, 0)))       # y- side
    F.append(((x0,  W / 2, 0), (L, 0, 0), (0, 0, h1), (0, 1, 0)))    # y+ side

    # --- fins (z in [h1, h1+h2]) ---
    for k in range(f):
        ys = y0 + k * (t + g)
        F.append(((x0, ys, h1 + h2), (L, 0, 0), (0, t, 0), (0, 0, 1)))       # fin top
        F.append(((x0, ys, h1), (L, 0, 0), (0, 0, h2), (0, -1, 0)))          # fin y- (gap)
        F.append(((x0, ys + t, h1), (L, 0, 0), (0, 0, h2), (0, 1, 0)))       # fin y+ (gap)
        F.append(((x0, ys, h1), (0, t, 0), (0, 0, h2), (-1, 0, 0)))          # fin x- end
        F.append((( L / 2, ys, h1), (0, t, 0), (0, 0, h2), (1, 0, 0)))       # fin x+ end
    return F


def heatsink_surface(cond, n_surf=2048, seed=0):
    """Sample exactly `n_surf` points on the heatsink solid boundary with outward normals.

    Points are allocated to faces proportional to face area (uniform surface density).
    Returns (points[n_surf,3], normals[n_surf,3]) as float32; normals are unit outward.
    """
    rng = np.random.default_rng(seed)
    faces = _faces(cond)
    origins = np.array([f[0] for f in faces], np.float64)
    us = np.array([f[1] for f in faces], np.float64)
    vs = np.array([f[2] for f in faces], np.float64)
    normals = np.array([f[3] for f in faces], np.float64)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    areas = np.linalg.norm(us, axis=1) * np.linalg.norm(vs, axis=1)

    # allocate points per face proportional to area (largest-remainder to hit exactly n_surf)
    frac = areas / areas.sum() * n_surf
    counts = np.floor(frac).astype(int)
    rem = n_surf - counts.sum()
    if rem > 0:
        order = np.argsort(-(frac - counts))[:rem]
        counts[order] += 1

    pts, nrm = [], []
    for i, c in enumerate(counts):
        if c == 0:
            continue
        ab = rng.random((c, 2))
        p = origins[i] + ab[:, 0:1] * us[i] + ab[:, 1:2] * vs[i]
        pts.append(p)
        nrm.append(np.tile(normals[i], (c, 1)))
    points = np.concatenate(pts, axis=0).astype(np.float32)
    norms = np.concatenate(nrm, axis=0).astype(np.float32)
    # shuffle so ordering carries no information (encoder must be permutation-invariant)
    perm = rng.permutation(points.shape[0])
    return points[perm], norms[perm]
