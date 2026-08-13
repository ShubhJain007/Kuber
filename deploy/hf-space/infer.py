"""Inference engine for the Kuber.ai thermal-surrogate demo.

Loads a trained GeoTransolver/SurfaceGeoTransolver checkpoint once, holds the
model + normalizers + the in-distribution test cases in memory, and predicts the
full 5-channel field (U_x,U_y,U_z,T,p_rgh) for a chosen case in ~sub-second.

Reuses the exact training-time input construction via src.train_simshift.load_dataset
(so normalization + analytic geometry match training). De-normalizes to physical
units with pred*ystd + ymean (per make_demo_frames.py).
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import torch

from kuber.train_simshift import (load_dataset, _build_model, _dims_from, _forward,
                                _import_geotransolver, _norm_frame, _apply_frame)
from kuber.surface_geom import heatsink_surface
from kuber.edge_proof import _arg_defaults

CHANNELS = ["U_x", "U_y", "U_z", "T", "p_rgh"]
_TIDX = 3
_PR2FLUID = {0.705: "air", 6.1: "water", 292.0: "oil", 29.0: "glycol"}


def _fluid_name(cond):
    return _PR2FLUID.get(round(float(cond.get("Pr", 0.0)), 3), "coolant")


# Pr, rho, Cp, mu  (matches adapt_bsf / adapt_coldplate so demo custom == training physics)
_FLUID_PROPS = {"air": (0.705, 1.18, 1004.4, 1.831e-5), "water": (6.1, 997.0, 4181.0, 8.9e-4),
                "oil": (292.0, 850.0, 1900.0, 2.0e-2), "glycol": (29.0, 1070.0, 3300.0, 3.5e-3)}


def _duct_surface(L, W, H, n_surf, seed=0):
    """4 long wetted walls of the duct (x in [0,L], y in [0,H], z in [0,W]) + inward normals."""
    rng = np.random.default_rng(seed)
    faces = [(1, 0.0, (0, 1, 0)), (1, H, (0, -1, 0)), (2, 0.0, (0, 0, 1)), (2, W, (0, 0, -1))]
    hi = (L, H, W); areas = [L * W, L * W, L * H, L * H]; tot = sum(areas) + 1e-12
    P, Nn = [], []
    for (ax, val, nrm), a in zip(faces, areas):
        k = max(1, int(round(n_surf * a / tot))); pts = np.zeros((k, 3)); pts[:, ax] = val
        o = [i for i in range(3) if i != ax]
        pts[:, o[0]] = rng.uniform(0, hi[o[0]], k); pts[:, o[1]] = rng.uniform(0, hi[o[1]], k)
        P.append(pts); Nn.append(np.tile(np.asarray(nrm, np.float64), (k, 1)))
    P = np.concatenate(P); Nn = np.concatenate(Nn)
    sel = rng.choice(len(P), n_surf, replace=len(P) < n_surf)
    return P[sel].astype(np.float32), Nn[sel].astype(np.float32)


def _heatsink_boxes_norm(cond, lo, scale):
    """Heatsink solid (base slab + fins) as [lo,hi] boxes in the per-case [0,1] frame."""
    L, W = cond["length"], cond["width"]
    t, g, f = cond["thickness_fins"], cond["gap"], int(cond["fins"])
    h1, h2 = cond["height1"], cond["height2"]
    boxes = [((-L / 2, -W / 2, 0.0), (L / 2, W / 2, h1))]          # base slab
    y0 = -W / 2
    for k in range(f):                                            # fins
        ys = y0 + k * (t + g)
        boxes.append(((-L / 2, ys, h1), (L / 2, ys + t, h1 + h2)))
    out = []
    for lo_b, hi_b in boxes:
        a = _apply_frame(np.array(lo_b)[None], lo, scale)[0]
        b = _apply_frame(np.array(hi_b)[None], lo, scale)[0]
        out.append([a.tolist(), b.tolist()])
    return out


def _streamlines(coords, vel, temp, n_seed=46, max_steps=90, step=0.014, G=44):
    """Trace streamlines through the predicted velocity field.

    coords: [N,3] in the per-case [0,1] frame (isotropic), vel: [N,3] physical
    velocity (direction is frame-invariant), temp: [N] physical K. Rasterizes the
    scattered field onto a coarse GxGxG grid, seeds near the inflow (low x), and
    integrates forward. Returns a list of {pts:[[x,y,z]..], T:[..]} in the same
    [0,1] frame for the frontend to render + color by temperature.
    """
    mn = coords.min(0); mx = coords.max(0)
    span = np.maximum(mx - mn, 1e-6)

    def to_g(p):
        return np.clip((p - mn) / span * (G - 1), 0, G - 1)

    gi = to_g(coords).astype(int)
    Vg = np.zeros((G, G, G, 3)); Tg = np.zeros((G, G, G)); Cg = np.zeros((G, G, G))
    np.add.at(Vg, (gi[:, 0], gi[:, 1], gi[:, 2]), vel)
    np.add.at(Tg, (gi[:, 0], gi[:, 1], gi[:, 2]), temp)
    np.add.at(Cg, (gi[:, 0], gi[:, 1], gi[:, 2]), 1.0)
    nz = Cg > 0
    Vg[nz] /= Cg[nz, None]; Tg[nz] /= Cg[nz]

    def sample(p):
        i = np.round(to_g(p)).astype(int)
        return Vg[i[0], i[1], i[2]], Tg[i[0], i[1], i[2]], Cg[i[0], i[1], i[2]] > 0

    # seed near the inflow plane (low x), spread over the populated y,z
    xlo = np.percentile(coords[:, 0], 12)
    pool = np.where(coords[:, 0] <= xlo)[0]
    if pool.size < n_seed:
        pool = np.arange(coords.shape[0])
    seeds = coords[np.random.default_rng(1).choice(pool, min(n_seed, pool.size), replace=False)]

    lines = []
    for s0 in seeds:
        p = s0.astype(np.float64).copy(); pts = [p.copy()]; Ts = [float(sample(p)[1])]
        for _ in range(max_steps):
            v, t, ok = sample(p)
            nv = np.linalg.norm(v)
            if not ok or nv < 1e-7:
                break
            p = p + step * v / nv
            if np.any(p < mn - 0.03) or np.any(p > mx + 0.03):
                break
            pts.append(p.copy()); Ts.append(float(t))
        if len(pts) >= 5:
            lines.append({"pts": np.round(np.array(pts), 4).tolist(),
                          "T": [round(x, 2) for x in Ts]})
    return lines


def _tgrid(coords, temp, fill, G=40):
    """Rasterize the scattered temperature field onto a GxGxG grid (empty cells =
    ambient `fill`), so the frontend can render slice planes / isosurfaces.
    Returns grid dims + bounds (in the [0,1] frame) + a flat C-order int array."""
    mn = coords.min(0); mx = coords.max(0); span = np.maximum(mx - mn, 1e-6)
    gi = np.clip((coords - mn) / span * (G - 1), 0, G - 1).astype(int)
    S = np.zeros((G, G, G)); C = np.zeros((G, G, G))
    np.add.at(S, (gi[:, 0], gi[:, 1], gi[:, 2]), temp)
    np.add.at(C, (gi[:, 0], gi[:, 1], gi[:, 2]), 1.0)
    nz = C > 0; S[nz] /= C[nz]; S[~nz] = fill
    return {"G": G, "mn": [round(float(x), 4) for x in mn], "span": [round(float(x), 4) for x in span],
            "T": [int(round(float(x))) for x in S.ravel()]}


def _boxes_phys(cond):
    """Heatsink boxes (base slab + fins) as [lo,hi] in PHYSICAL metres from conditions."""
    L, W = float(cond["length"]), float(cond["width"])
    t, g, f = float(cond["thickness_fins"]), float(cond["gap"]), int(cond["fins"])
    h1, h2 = float(cond["height1"]), float(cond["height2"])
    boxes = [((-L / 2, -W / 2, 0.0), (L / 2, W / 2, h1))]
    y0 = -W / 2
    for k in range(f):
        ys = y0 + k * (t + g)
        boxes.append(((-L / 2, ys, h1), (L / 2, ys + t, h1 + h2)))
    return boxes


def _align_boxes(boxes, surf_bb):
    """Translate origin-centered heatsink fin boxes so their bounding-box center matches the
    stored surface bbox center — placing the reconstructed solid at its TRUE location in the
    CFD mesh frame so it sits inside the fluid point cloud (the coords we return are mesh-frame).
    Returns JSON-ready [[lo,hi],...]. Without a stored surface, returns the boxes unchanged."""
    lo = np.min([b[0] for b in boxes], axis=0)
    hi = np.max([b[1] for b in boxes], axis=0)
    if surf_bb is None:
        return [[list(map(float, b[0])), list(map(float, b[1]))] for b in boxes]
    sp_lo, sp_hi = surf_bb
    off = (np.asarray(sp_lo) + np.asarray(sp_hi)) / 2.0 - (np.asarray(lo) + np.asarray(hi)) / 2.0
    return [[(np.asarray(b[0]) + off).tolist(), (np.asarray(b[1]) + off).tolist()] for b in boxes]


def _stl_from_cond(cond, name="heatsink"):
    """ASCII STL of the heatsink solid (axis-aligned boxes -> 12 triangles each).

    `name` becomes the STL `solid <name>` header — for library geometries we embed
    the case id (e.g. `kuber_case_0304`) so a re-uploaded STL round-trips back to its
    known in-distribution config (editable + CFD ground truth available)."""
    faces = [(0, 3, 2), (0, 2, 1), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (3, 7, 6), (3, 6, 2), (0, 4, 7), (0, 7, 3), (1, 2, 6), (1, 6, 5)]
    out = ["solid " + name]
    for lo, hi in _boxes_phys(cond):
        x0, y0, z0 = lo; x1, y1, z1 = hi
        v = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
             (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        for (a, b, c) in faces:
            out.append(" facet normal 0 0 0\n  outer loop")
            for p in (v[a], v[b], v[c]):
                out.append("   vertex %g %g %g" % p)
            out.append("  endloop\n endfacet")
    out.append("endsolid " + name)
    return "\n".join(out)


def _csv_field(coords, pred):
    """Point-cloud field export (ANSYS/Fluent/CFD-Post ingestible CSV)."""
    lines = ["x,y,z,T_K,Ux_m_s,Uy_m_s,Uz_m_s,p_rgh_Pa"]
    for i in range(coords.shape[0]):
        c = coords[i]; p = pred[i]
        lines.append("%.6g,%.6g,%.6g,%.4f,%.5f,%.5f,%.5f,%.3f"
                     % (c[0], c[1], c[2], p[3], p[0], p[1], p[2], p[4]))
    return "\n".join(lines)


def _vtk_field(coords, pred):
    """Legacy VTK POLYDATA point cloud (ParaView / EnSight / most CFD post-processors)."""
    n = coords.shape[0]
    L = ["# vtk DataFile Version 3.0", "Kuber.ai neural CFD surrogate field", "ASCII",
         "DATASET POLYDATA", "POINTS %d float" % n]
    L += ["%.6g %.6g %.6g" % (coords[i][0], coords[i][1], coords[i][2]) for i in range(n)]
    L.append("POINT_DATA %d" % n)
    L += ["SCALARS temperature float 1", "LOOKUP_TABLE default"] + ["%.4f" % pred[i, 3] for i in range(n)]
    L += ["SCALARS pressure float 1", "LOOKUP_TABLE default"] + ["%.3f" % pred[i, 4] for i in range(n)]
    L += ["VECTORS velocity float"] + ["%.5f %.5f %.5f" % (pred[i, 0], pred[i, 1], pred[i, 2]) for i in range(n)]
    return "\n".join(L)


class Engine:
    def __init__(self, ckpt, data, splits, difficulty="medium", device=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        GeoT = _import_geotransolver(None)
        ck = torch.load(ckpt, map_location=self.dev, weights_only=False)
        cfg = ck["meta"]["config"]
        self.cond_keys = ck["meta"]["cond_keys"]
        self.geom_mode = cfg.get("geom_mode", "sdf" if cfg.get("use_sdf", False) else "none")
        n_surf = cfg.get("n_surf", 2048)
        self.data_dir = Path(data)
        self.extent_feats = cfg.get("extent_feats", False)
        # build inputs + normalizers exactly as training did (multi-geometry: surface + physics cond + extent)
        self.dataset, self.meta = load_dataset(data, splits, difficulty,
                                               cond_keys=self.cond_keys,
                                               geom_mode=self.geom_mode, n_surf=n_surf,
                                               extent_feats=self.extent_feats)
        d = _dims_from(cfg, _arg_defaults())
        self.model, self.is_ref = _build_model(self.geom_mode, self.meta["cond_dim"],
                                                self.meta["feat_dim"], d, self.dev, GeoT)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.ymean = np.asarray(self.meta["ymean"], np.float64)
        self.ystd = np.asarray(self.meta["ystd"], np.float64)
        # normalizers exposed by load_dataset (used to build inputs for custom geometries)
        self.cmean = np.asarray(self.meta["cmean"], np.float64)
        self.cstd = np.asarray(self.meta["cstd"], np.float64)
        self.ext_mean = self.meta.get("ext_mean"); self.ext_std = self.meta.get("ext_std")
        self.n_surf = n_surf
        self.split = self.dataset["src_test"]                     # in-distribution cases (both device classes)
        self.ids = list(self.split["ids"])
        self.conds, self.bbox, self.surf_bb = [], [], []
        for sid in self.ids:                                      # per-case conditions + physical bbox
            z = np.load(self.data_dir / f"{sid}.npz", allow_pickle=True)
            self.conds.append(json.loads(str(z["conditions"])))
            c = np.asarray(z["coords"], np.float64); self.bbox.append((c.min(0), c.max(0)))
            if "surf_pts" in z.files:                             # true solid location in the mesh frame
                sp = np.asarray(z["surf_pts"], np.float64); self.surf_bb.append((sp.min(0), sp.max(0)))
            else:
                self.surf_bb.append(None)
        self.devs = [int(round(float(c.get("device", 0.0)))) for c in self.conds]  # 0=heatsink 1=cold plate
        # warm up CUDA (kernel compile/cache) so the FIRST user prediction is fast, not a cold start
        try:
            if self.dev.type == "cuda":
                self.predict(0); torch.cuda.synchronize()
        except Exception:
            pass

    def _case_boxes(self, i):
        """Physical [lo,hi] boxes: heatsink -> fin boxes from conditions, translated to the solid's
        true location in the CFD mesh frame (so they sit inside the fluid cloud); cold plate ->
        channel duct (coords bbox)."""
        if self.devs[i] == 0 and "fins" in self.conds[i]:
            return _align_boxes(_boxes_phys(self.conds[i]), self.surf_bb[i])
        lo, hi = self.bbox[i]
        return [[lo.tolist(), hi.tolist()]]

    def list_cases(self):
        out = []
        for i, c in enumerate(self.conds):
            lo, hi = self.bbox[i]; ext = (hi - lo) * 1000.0            # mm
            if self.devs[i] == 0:                                      # heatsink (wall-temp BC, air)
                out.append({"index": i, "id": self.ids[i], "device": "heatsink",
                            "title": "Heatsink · %d fins" % int(c.get("fins", 0)),
                            "sub": "%.0f×%.0f mm · wall %.0f K · %s" % (float(c.get("length", 0)) * 1000, float(c.get("width", 0)) * 1000, float(c.get("solidTemp", 0)), _fluid_name(c)),
                            "fins": int(c.get("fins", 0)), "solidTemp": round(float(c.get("solidTemp", 0)), 1),
                            "envTemp": round(float(c.get("envTemp", 300)), 1), "fluid": _fluid_name(c)})
            else:                                                      # cold plate (heat-flux BC, liquid, forced)
                out.append({"index": i, "id": self.ids[i], "device": "coldplate",
                            "title": "Cold plate · %s" % _fluid_name(c),
                            "sub": "%.0f×%.0f×%.1f mm · q=%.0f kW/m² · %.2f m/s" % (ext[0], ext[2], ext[1], float(c.get("heatFlux", 0)) / 1000.0, float(c.get("u_in", 0))),
                            "heatFlux": round(float(c.get("heatFlux", 0))), "u_in": round(float(c.get("u_in", 0)), 2),
                            "envTemp": round(float(c.get("envTemp", 300)), 1), "fluid": _fluid_name(c),
                            "Lmm": round(float(ext[0]), 1), "Wmm": round(float(ext[2]), 1), "Hmm": round(float(ext[1]), 2)})
        return out

    def stl_case(self, i):
        """ASCII STL of a preset in-distribution case, with its case id embedded in the
        solid header so a re-upload maps back to the known config (editable + GT)."""
        return _stl_from_cond(self.conds[i], name=f"kuber_{self.ids[i]}")

    def geometry(self, i):
        """Geometry boxes WITHOUT running the model — UI preview before Predict (physical frame)."""
        return {"case_id": self.ids[i], "device": "heatsink" if self.devs[i] == 0 else "coldplate",
                "boxes": self._case_boxes(i)}

    def _resolve_cond(self, ov):
        """Build a full conditions dict from user overrides, seeded by the nearest
        in-distribution case (for the volume frame + any untouched fields)."""
        tgt = int(round(float(ov.get("fins") or 8)))
        hs = [i for i in range(len(self.ids)) if self.devs[i] == 0 and "fins" in self.conds[i]]
        base = min(hs, key=lambda i: abs(int(self.conds[i]["fins"]) - tgt)) if hs else 0
        cond = dict(self.conds[base])
        for k in ("fins", "length", "width", "height1", "height2", "gap", "thickness_fins", "solidTemp", "envTemp"):
            if ov.get(k) is not None:
                cond[k] = float(ov[k])
        cond["fins"] = max(1, int(round(cond["fins"])))
        return cond, base

    @torch.no_grad()
    def _infer_cond(self, cond, base):
        vol = np.asarray(np.load(self.data_dir / f"{self.ids[base]}.npz", allow_pickle=True)["coords"], np.float64)
        lo, scale = _norm_frame(vol); c01 = _apply_frame(vol, lo, scale)
        sp_phys, sn = heatsink_surface(cond, n_surf=self.n_surf)
        sp01 = _apply_frame(sp_phys.astype(np.float64), lo, scale)
        cvec = np.array([cond.get(k, 0.0) for k in self.cond_keys], np.float64)
        condn = ((cvec - self.cmean) / self.cstd).astype(np.float32)
        feats = np.broadcast_to(condn, (c01.shape[0], len(self.cond_keys))).astype(np.float32)
        if self.extent_feats and self.ext_mean is not None:       # append physical bbox extents (aspect/scale)
            extn = ((np.log(scale + 1e-12) - np.asarray(self.ext_mean)) / np.asarray(self.ext_std)).astype(np.float32)
            feats = np.concatenate([feats, np.broadcast_to(extn, (c01.shape[0], 3)).astype(np.float32)], axis=1)
        pos = torch.tensor(c01[None], device=self.dev); fx = torch.tensor(feats[None], device=self.dev)
        sp = torch.tensor(sp01[None], device=self.dev); snt = torch.tensor(sn[None].astype(np.float32), device=self.dev)
        if self.dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        pred = _forward(self.model, self.geom_mode, pos, fx, sp, snt)
        if self.dev.type == "cuda": torch.cuda.synchronize()
        dt = time.time() - t0
        return vol, c01, lo, scale, pred[0].cpu().numpy() * self.ystd + self.ymean, dt

    def predict_custom(self, ov, max_points=16384):
        cond, base = self._resolve_cond(ov)
        vol, c01, lo, scale, pred, dt = self._infer_cond(cond, base)
        N = c01.shape[0]
        pool = np.where(c01[:, 2] < 0.45)[0]
        if pool.size < 2000: pool = np.arange(N)
        idx = pool if pool.size <= max_points else np.random.default_rng(0).choice(pool, max_points, replace=False)
        cc, pv = vol[idx], pred[idx]                               # PHYSICAL coords (uniform aspect)
        vmag = np.linalg.norm(pv[:, :3], axis=1)
        return {
            "case_id": "custom", "fins": int(cond["fins"]), "custom": True,
            "extrapolated": bool(not (5 <= cond["fins"] <= 8)),
            "n_points": int(cc.shape[0]), "infer_ms": round(dt * 1000, 1),
            "coords": np.round(cc, 4).astype(np.float32).tolist(),
            "fields": {
                "T": {"pred": np.round(pv[:, _TIDX], 2).tolist(), "gt": np.round(pv[:, _TIDX], 2).tolist(), "unit": "K"},
                "velocity": {"pred": np.round(vmag, 4).tolist(), "gt": np.round(vmag, 4).tolist(), "unit": "m/s"},
                "p_rgh": {"pred": np.round(pv[:, 4], 1).tolist(), "gt": np.round(pv[:, 4], 1).tolist(), "unit": "Pa"}},
            "boxes": _align_boxes(_boxes_phys(cond), self.surf_bb[base]),
            "metrics": {"peak_T": round(float(pred[:, _TIDX].max()), 2), "peak_T_gt": round(float(pred[:, _TIDX].max()), 2),
                        "T_rmse": 0.0, "dP": round(float(pred[:, 4].max() - pred[:, 4].min()), 1),
                        "solidTemp": round(float(cond["solidTemp"]), 1), "envTemp": round(float(cond.get("envTemp", 300)), 1)}}

    @torch.no_grad()
    def predict_custom_cp(self, ov, max_points=16384):
        """Custom COLD PLATE: build a duct (channel dims) + fluid + heat flux from the editor,
        run the same multi-geometry model. Volume = a base cold-plate's fluid points rescaled to
        the new duct; surface = the 4 duct walls; conditioning = physics + device=1 + extents."""
        fluid = ov.get("fluid", "water")
        pr, rho, cp, mu = _FLUID_PROPS.get(fluid, _FLUID_PROPS["water"])
        L = float(ov.get("L", 0.12)); W = float(ov.get("W", 0.02)); H = float(ov.get("H", 0.003))
        q = float(ov.get("q", 2e5)); u_in = float(ov.get("u_in", 1.0)); T_in = float(ov.get("T_in", 300.0))
        cond = dict(rho=rho, mu=mu, Cp=cp, Pr=pr, u_in=u_in, envTemp=T_in, solidTemp=0.0, heatFlux=q, device=1.0)
        base = next(i for i in range(len(self.ids)) if self.devs[i] == 1)   # a cold plate for the fluid-point layout
        b = np.asarray(np.load(self.data_dir / f"{self.ids[base]}.npz", allow_pickle=True)["coords"], np.float64)
        ext = b.max(0) - b.min(0); ext[ext < 1e-9] = 1.0
        vol = (b - b.min(0)) / ext * np.array([L, H, W])                    # rescale fluid points to the new duct
        sp_phys, sn = _duct_surface(L, W, H, self.n_surf)
        lo, scale = _norm_frame(vol); c01 = _apply_frame(vol, lo, scale); sp01 = _apply_frame(sp_phys.astype(np.float64), lo, scale)
        cvec = np.array([cond.get(k, 0.0) for k in self.cond_keys], np.float64)
        feats = np.broadcast_to(((cvec - self.cmean) / self.cstd).astype(np.float32), (c01.shape[0], len(self.cond_keys))).astype(np.float32)
        if self.extent_feats and self.ext_mean is not None:
            extn = ((np.log(scale + 1e-12) - np.asarray(self.ext_mean)) / np.asarray(self.ext_std)).astype(np.float32)
            feats = np.concatenate([feats, np.broadcast_to(extn, (c01.shape[0], 3)).astype(np.float32)], axis=1)
        pos = torch.tensor(c01[None], device=self.dev); fx = torch.tensor(feats[None], device=self.dev)
        sp = torch.tensor(sp01[None], device=self.dev); snt = torch.tensor(sn[None].astype(np.float32), device=self.dev)
        if self.dev.type == "cuda": torch.cuda.synchronize()
        t0 = time.time(); pred = _forward(self.model, self.geom_mode, pos, fx, sp, snt)
        if self.dev.type == "cuda": torch.cuda.synchronize()
        dt = time.time() - t0
        pred = pred[0].cpu().numpy() * self.ystd + self.ymean
        N = vol.shape[0]; idx = np.arange(N) if N <= max_points else np.random.default_rng(0).choice(N, max_points, replace=False)
        c = vol[idx]; pv = pred[idx]; vmag = np.linalg.norm(pv[:, :3], axis=1)
        return {"case_id": "custom", "device": "coldplate", "custom": True, "extrapolated": False,
                "n_points": int(c.shape[0]), "infer_ms": round(dt * 1000, 1),
                "coords": np.round(c, 5).astype(np.float32).tolist(),
                "fields": {"T": {"pred": np.round(pv[:, _TIDX], 2).tolist(), "gt": np.round(pv[:, _TIDX], 2).tolist(), "unit": "K"},
                           "velocity": {"pred": np.round(vmag, 4).tolist(), "gt": np.round(vmag, 4).tolist(), "unit": "m/s"},
                           "p_rgh": {"pred": np.round(pv[:, 4], 1).tolist(), "gt": np.round(pv[:, 4], 1).tolist(), "unit": "Pa"}},
                "boxes": [[vol.min(0).tolist(), vol.max(0).tolist()]],
                "metrics": {"peak_T": round(float(pred[:, _TIDX].max()), 2), "peak_T_gt": round(float(pred[:, _TIDX].max()), 2),
                            "T_rmse": 0.0, "dP": round(float(pred[:, 4].max() - pred[:, 4].min()), 1),
                            "solidTemp": 0.0, "envTemp": round(T_in, 1), "heatFlux": round(q), "u_in": round(u_in, 2), "fluid": fluid}}

    def export(self, ov, fmt):
        cond, base = self._resolve_cond(ov)
        if fmt == "stl":
            return _stl_from_cond(cond), "model/stl", "heatsink.stl"
        vol, c01, lo, scale, pred, dt = self._infer_cond(cond, base)
        if fmt == "csv":
            return _csv_field(vol, pred), "text/csv", "kuber_field.csv"
        if fmt == "vtk":
            return _vtk_field(vol, pred), "text/plain", "kuber_field.vtk"
        raise ValueError("unknown format: " + fmt)

    @torch.no_grad()
    def predict(self, i, max_points=16384):
        s = self.split
        pos = torch.tensor(s["coords"][i:i + 1], device=self.dev)
        fx = torch.tensor(s["feats"][i:i + 1], device=self.dev)
        sp = torch.tensor(s["surf_pts"][i:i + 1], device=self.dev) if self.geom_mode == "surface" else None
        sn = torch.tensor(s["surf_normals"][i:i + 1], device=self.dev) if self.geom_mode == "surface" else None
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        if self.is_ref:
            pred = self.model.predict(fx if fx.shape[-1] > 0 else None, pos, sp, sn, sample=False)
        else:
            pred = _forward(self.model, self.geom_mode, pos, fx, sp, sn)
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0

        pred = pred[0].cpu().numpy() * self.ystd + self.ymean     # [N,5] physical
        gt = s["y"][i] * self.ystd + self.ymean
        coords = s["coords"][i]                                   # normalized [0,1]
        N = coords.shape[0]
        # concentrate points in the near-field (heatsink + plume sit at low z),
        # so the render is dense where the temperature actually varies
        pool = np.arange(N) if self.devs[i] == 1 else np.where(coords[:, 2] < 0.45)[0]  # cold plate: whole channel; heatsink: near-field
        if pool.size < 2000:
            pool = np.arange(N)
        idx = (pool if pool.size <= max_points
               else np.random.default_rng(0).choice(pool, max_points, replace=False))
        vol_phys = np.asarray(np.load(self.data_dir / f"{self.ids[i]}.npz",
                                      allow_pickle=True)["coords"], np.float64)
        c = vol_phys[idx]                                          # PHYSICAL coords (uniform aspect)
        pv, gv = pred[idx], gt[idx]
        vmagP = np.linalg.norm(pv[:, :3], axis=1)
        vmagG = np.linalg.norm(gv[:, :3], axis=1)
        boxes = self._case_boxes(i)                                # PHYSICAL boxes (heatsink fins OR cold-plate duct)

        T_rmse = float(np.sqrt(np.mean((pred[:, _TIDX] - gt[:, _TIDX]) ** 2)))
        return {
            "case_id": self.ids[i], "device": "heatsink" if self.devs[i] == 0 else "coldplate",
            "fins": int(self.conds[i].get("fins", 0)),
            "n_points": int(c.shape[0]), "infer_ms": round(dt * 1000, 1),
            "coords": np.round(c, 4).astype(np.float32).tolist(),
            "fields": {
                "T":        {"pred": np.round(pred[idx, _TIDX], 2).tolist(),
                             "gt": np.round(gt[idx, _TIDX], 2).tolist(), "unit": "K"},
                "velocity": {"pred": np.round(vmagP, 4).tolist(),
                             "gt": np.round(vmagG, 4).tolist(), "unit": "m/s"},
                "p_rgh":    {"pred": np.round(pv[:, 4], 1).tolist(),
                             "gt": np.round(gv[:, 4], 1).tolist(), "unit": "Pa"},
            },
            "boxes": boxes,
            "metrics": {
                "peak_T": round(float(pred[:, _TIDX].max()), 2),
                "peak_T_gt": round(float(gt[:, _TIDX].max()), 2),
                "T_rmse": round(T_rmse, 2),
                "dP": round(float(pred[:, 4].max() - pred[:, 4].min()), 1),
                "solidTemp": round(float(self.conds[i].get("solidTemp", 0)), 1),
                "envTemp": round(float(self.conds[i].get("envTemp", 300)), 1),
                "heatFlux": round(float(self.conds[i].get("heatFlux", 0))),
                "u_in": round(float(self.conds[i].get("u_in", 0)), 2),
                "fluid": _fluid_name(self.conds[i]),
            },
        }


if __name__ == "__main__":     # smoke test
    import sys
    eng = Engine(ckpt="outputs/geot_medium_surface.pt",
                 data="/home/shubhj/simshift/npz",
                 splits="/home/shubhj/simshift/npz/splits.json")
    print(f"[engine] device={eng.dev} geom_mode={eng.geom_mode} cases={len(eng.ids)}", flush=True)
    r = eng.predict(0)
    print(f"[predict] case={r['case_id']} fins={r['fins']} pts={r['n_points']} "
          f"infer={r['infer_ms']}ms peakT={r['metrics']['peak_T']}K "
          f"(gt {r['metrics']['peak_T_gt']}K) T_rmse={r['metrics']['T_rmse']}K dP={r['metrics']['dP']}Pa",
          flush=True)
    if eng.dev.type == "cuda":
        print(f"[vram] {torch.cuda.max_memory_allocated()/1e9:.2f} GB", flush=True)
