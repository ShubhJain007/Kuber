"""Train GeoTransolver (PhysicsNeMo multiscale ball-query) on the SIMSHIFT heatsink
benchmark, following the official recipe: train on src.train, early-stop on src.val,
evaluate on src.test (in-distribution) AND tgt.test (OOD geometry shift).

Data: per-case npz with coords[3], U[3], T[1], p_rgh[1] (16384 pts), conditions JSON.
Splits: splits.json -> {easy,medium,hard} -> {src,tgt} -> {train,val,test}.

Model I/O (study_plan.md sec 5):
  local_positions [B,N,3] = per-case coords normalized to [0,1]
  local_embedding [B,N,C] = normalized condition scalars, broadcast per node
  geometry        [B,N,3] = per-case coords (drives the multiscale ball-query)
  output          [B,N,5] = U(3), T(1), p_rgh(1), per-channel z-normalized targets

Usage (in `pnemo` env):
  # time one epoch first (do this on the pod before committing to a full run)
  python -u -m kuber.train_simshift --data <npz_dir> --splits <splits.json> --time_one_epoch
  # full run
  python -u -m kuber.train_simshift --data <npz_dir> --splits <splits.json> --difficulty medium
  # laptop smoke
  python -u -m kuber.train_simshift --data <npz_dir> --splits <splits.json> --smoke
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from kuber.surface_geom import heatsink_surface


def _import_geotransolver(pnemo_path: str | None):
    if pnemo_path and pnemo_path not in sys.path:
        sys.path.insert(0, pnemo_path)
    from physicsnemo.experimental.models.geotransolver.geotransolver import GeoTransolver
    return GeoTransolver


# ----------------------------------------------------------------------------- data
def _norm_frame(coords):
    """Per-case min-max frame: returns (lo[3], scale[3]) so (x-lo)/scale maps to [0,1]."""
    lo = coords.min(0)
    hi = coords.max(0)
    scale = np.where(hi > lo, hi - lo, 1.0)
    return lo, scale


def _apply_frame(coords, lo, scale):
    return ((coords - lo) / scale).astype(np.float32)


def _norm01(coords):
    """Per-case min-max normalize each axis to [0,1]."""
    lo, scale = _norm_frame(coords)
    return _apply_frame(coords, lo, scale)


def _box_sdf(p, lo, hi):
    """Signed distance from points p[N,3] to an axis-aligned box [lo,hi] (neg inside)."""
    c = (lo + hi) / 2.0
    h = (hi - lo) / 2.0
    q = np.abs(p - c) - h
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=1)
    inside = np.minimum(np.max(q, axis=1), 0.0)
    return outside + inside


def heatsink_sdf(coords_phys, cond):
    """Analytic signed distance to the parametric heatsink solid (base plate + fins),
    computed in PHYSICAL coords. Validated: >=~96% of the fluid-only points read as
    outside; the small negatives are near-wall cell-size noise, clamped to 0 by callers.

    Geometry (from metadata): base plate L(x) x W(y) x height1(z) centered at (0,0,z=0);
    `fins` fins of thickness_fins along y, gap between, running full length x,
    height2 tall, sitting on the base (z in [height1, height1+height2]).
    """
    L, W = cond["length"], cond["width"]
    t, g, f = cond["thickness_fins"], cond["gap"], int(cond["fins"])
    h1, h2 = cond["height1"], cond["height2"]
    sdf = np.full(coords_phys.shape[0], 1e9, dtype=np.float64)
    # base plate
    sdf = np.minimum(sdf, _box_sdf(coords_phys,
                                   np.array([-L / 2, -W / 2, 0.0]),
                                   np.array([L / 2, W / 2, h1])))
    # fins along width (y), full length (x), on top of base
    y0 = -W / 2
    for k in range(f):
        ys = y0 + k * (t + g)
        sdf = np.minimum(sdf, _box_sdf(coords_phys,
                                       np.array([-L / 2, ys, h1]),
                                       np.array([L / 2, ys + t, h1 + h2])))
    return sdf


def heatsink_sdf_grad(coords_phys, cond, eps=1e-4):
    """Directional SDF: signed distance + UNIT gradient (direction away from nearest surface).
    grad = normalize(d SDF / dx) via central differences. For exterior points grad equals the
    outward normal of the nearest surface point -> gives the model 'which way the wall is', so
    windward and leeward points at equal distance are distinguishable (needed for flow).
    Returns (sdf[N], grad[N,3])."""
    sdf = heatsink_sdf(coords_phys, cond)
    g = np.zeros((coords_phys.shape[0], 3), np.float64)
    for ax in range(3):
        d = np.zeros(3); d[ax] = eps
        g[:, ax] = (heatsink_sdf(coords_phys + d, cond) - heatsink_sdf(coords_phys - d, cond)) / (2 * eps)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return sdf, (g / np.where(n > 1e-9, n, 1.0)).astype(np.float32)


def load_dataset(data_dir, splits_path, difficulty, cond_keys=None, geom_mode="sdf",
                 n_surf=2048, drop_geom_scalars=False, extent_feats=False):
    """Return dict of tensors per split + normalization stats.

    geom_mode:
      "sdf"     -> per-node analytic SDF scalar appended to `feats` (GeoTransolver).
      "surface" -> analytic surface point cloud + normals per case (SurfaceGeoTransolver);
                   surface is mapped into the SAME per-case [0,1] frame as the volume so
                   KNN / relative positions are consistent.
      "none"    -> condition scalars only, no geometry feature.

    drop_geom_scalars: keep only the BC scalar `solidTemp` in cond_keys (drop fins/gap/
      thickness/height) — tests whether global geometry scalars hurt OOD.

    All normalizers are fit on src.train ONLY (no leakage; difficulty-invariant).
    All cases share N=16384; loaded into host RAM.
    """
    data_dir = Path(data_dir)
    splits = json.loads(Path(splits_path).read_text())[difficulty]
    use_sdf = geom_mode == "sdf"           # scalar SDF (1 ch)
    use_dsdf = geom_mode == "dsdf"         # directional SDF: signed dist + unit grad (4 ch)

    needed = set()
    for domain in ("src", "tgt"):
        for part in ("train", "val", "test"):
            needed.update(splits[domain][part])
    src_train = splits["src"]["train"]

    # --- pass 1: read raw conditions ---------------------------------------------
    raw, all_cond = {}, {}
    for sid in sorted(needed):
        p = data_dir / f"{sid}.npz"
        if not p.exists():
            raise FileNotFoundError(f"missing npz for {sid}: {p}")
        with np.load(p, allow_pickle=True) as d:   # materialize + close: don't leak fds
            raw[sid] = {k: d[k] for k in d.files}   # (1.5k+ cases would exceed `ulimit -n`)
        all_cond[sid] = json.loads(str(raw[sid]["conditions"]))

    # condition columns: non-constant over src.train (optionally drop geometry scalars)
    if cond_keys is None:
        keys = sorted({k for c in all_cond.values() for k in c})
        mat = np.array([[all_cond[s].get(k, np.nan) for k in keys] for s in src_train], np.float64)
        stds = np.nanstd(mat, axis=0)
        cond_keys = [k for k, sd in zip(keys, stds) if sd > 1e-12]
        if drop_geom_scalars:
            cond_keys = [k for k in cond_keys if k == "solidTemp"]
    if cond_keys:
        cond_mat = np.array([[all_cond[s].get(k, 0.0) for k in cond_keys] for s in src_train], np.float64)
        cmean, cstd = cond_mat.mean(0), cond_mat.std(0) + 1e-9
    else:
        cmean = cstd = np.zeros(0, np.float64)

    # per-case physical bbox extents (log, z-scored over src.train) — restores the aspect
    # ratio / absolute scale that PER-AXIS [0,1] framing erases (critical for thin cold-plate
    # channels where a 36:1 duct would otherwise look identical to a cube). device-agnostic.
    if extent_feats:
        ext_mat = np.log(np.array([_norm_frame(np.asarray(raw[s]["coords"], np.float64))[1]
                                   for s in src_train], np.float64) + 1e-12)
        ext_mean, ext_std = ext_mat.mean(0), ext_mat.std(0) + 1e-9
    else:
        ext_mean = ext_std = None

    # target normalization: per-channel z-score over src.train
    stack = np.concatenate([np.concatenate([raw[s]["U"], raw[s]["T"], raw[s]["p_rgh"]], 1)
                            for s in src_train], axis=0)
    ymean = stack.mean(0).astype(np.float32)
    ystd = (stack.std(0) + 1e-8).astype(np.float32)

    def _sdf_of(sid):
        # prefer SDF precomputed by the adapter (our box geometry, any shape/frame)
        if "sdf" in raw[sid]:
            return np.clip(np.asarray(raw[sid]["sdf"], np.float64), 0.0, None).astype(np.float32)
        try:                                                    # analytic heatsink SDF
            return np.clip(heatsink_sdf(np.asarray(raw[sid]["coords"], np.float64),
                                        all_cond[sid]), 0.0, None).astype(np.float32)
        except Exception:                                       # non-heatsink geometry (e.g. cold plate)
            return np.zeros(np.asarray(raw[sid]["coords"]).shape[0], np.float32)  # SDF n/a; unused in surface mode
    def _dsdf_of(sid):  # signed sdf + unit gradient (direction)
        if "sdf_grad" in raw[sid]:
            return (np.asarray(raw[sid]["sdf"], np.float64),
                    np.asarray(raw[sid]["sdf_grad"], np.float32))
        return heatsink_sdf_grad(np.asarray(raw[sid]["coords"], np.float64), all_cond[sid])
    if use_sdf:
        sdf_stack = np.concatenate([_sdf_of(s) for s in src_train])
        sdf_mean, sdf_std = float(sdf_stack.mean()), float(sdf_stack.std() + 1e-8)
    elif use_dsdf:  # z-score the SIGNED distance; gradient is already unit (~[-1,1])
        sdf_stack = np.concatenate([_dsdf_of(s)[0] for s in src_train])
        sdf_mean, sdf_std = float(sdf_stack.mean()), float(sdf_stack.std() + 1e-8)
    else:
        sdf_mean, sdf_std = 0.0, 1.0

    def build(ids):
        if not ids:
            return None
        coords, feats, ys, surf_p, surf_n, sdfs = [], [], [], [], [], []
        for sid in ids:
            d = raw[sid]
            vol_phys = np.asarray(d["coords"], np.float64)
            lo, scale = _norm_frame(vol_phys)                           # per-case frame
            c = _apply_frame(vol_phys, lo, scale)                       # volume in [0,1]
            sdfs.append(_sdf_of(sid))                                   # per-point dist-to-solid (m)
            N = c.shape[0]
            if cond_keys:
                cond = np.asarray([all_cond[sid].get(k, 0.0) for k in cond_keys], np.float64)
                condn = ((cond - cmean) / cstd).astype(np.float32)
                fnode = np.broadcast_to(condn, (N, len(cond_keys))).astype(np.float32)
            else:
                fnode = np.zeros((N, 0), np.float32)
            if extent_feats:                                           # + physical bbox extents (aspect/scale)
                extn = ((np.log(scale + 1e-12) - ext_mean) / ext_std).astype(np.float32)
                fnode = np.concatenate([fnode, np.broadcast_to(extn, (N, 3)).astype(np.float32)], axis=1)
            if use_sdf:
                sdfn = ((_sdf_of(sid) - sdf_mean) / sdf_std)[:, None].astype(np.float32)
                fnode = np.concatenate([fnode, sdfn], axis=1)
            elif use_dsdf:
                sdf_v, grad = _dsdf_of(sid)
                sdfn = ((sdf_v - sdf_mean) / sdf_std)[:, None].astype(np.float32)
                fnode = np.concatenate([fnode, sdfn, grad], axis=1)     # [N, C+4]
            if geom_mode == "surface":
                if "surf_pts" in raw[sid]:        # our precomputed cloud
                    sp_phys = np.asarray(raw[sid]["surf_pts"], np.float64)
                    sn = np.asarray(raw[sid]["surf_normals"], np.float32)
                else:
                    sp_phys, sn = heatsink_surface(all_cond[sid], n_surf=n_surf)
                surf_p.append(_apply_frame(sp_phys.astype(np.float64), lo, scale))  # same frame
                surf_n.append(sn.astype(np.float32))                    # physical unit normals
            y = np.concatenate([d["U"], d["T"], d["p_rgh"]], axis=1)
            yn = ((y - ymean) / ystd).astype(np.float32)
            coords.append(c); feats.append(fnode); ys.append(yn)
        out = {"coords": np.stack(coords), "feats": np.stack(feats), "y": np.stack(ys),
               "sdf": np.stack(sdfs), "ids": list(ids)}
        if geom_mode == "surface":
            out["surf_pts"] = np.stack(surf_p)
            out["surf_normals"] = np.stack(surf_n)
        return out

    data = {k: build(splits[dom][part]) for k, (dom, part) in {
        "src_train": ("src", "train"), "src_val": ("src", "val"),
        "src_test": ("src", "test"), "tgt_test": ("tgt", "test")}.items()}
    feat_dim = len(cond_keys) + (3 if extent_feats else 0) + (1 if use_sdf else 0) + (4 if use_dsdf else 0)
    meta = {"cond_keys": cond_keys, "cond_dim": len(cond_keys), "ymean": ymean, "ystd": ystd,
            "use_sdf": use_sdf, "geom_mode": geom_mode, "n_surf": n_surf,
            "sdf_mean": sdf_mean, "sdf_std": sdf_std, "feat_dim": feat_dim,
            "cmean": cmean, "cstd": cstd, "extent_feats": extent_feats,
            "ext_mean": ext_mean, "ext_std": ext_std,   # exposed for the demo Engine (custom-geometry inputs)
            "channels": ["U_x", "U_y", "U_z", "T", "p_rgh"], "N": data["src_train"]["coords"].shape[1]}
    return data, meta


# --------------------------------------------------------------------------- metrics
def _subsample_idx(N, n_points, seed):
    if n_points >= N:
        return np.arange(N)
    return np.sort(np.random.default_rng(seed).choice(N, n_points, replace=False))


def _forward(model, geom_mode, pos, fx, sp=None, sn=None):
    """Dispatch to GeoTransolver (sdf/none) or SurfaceGeoTransolver (surface)."""
    if geom_mode == "surface":
        cond = fx if fx.shape[-1] > 0 else None
        return model(cond, pos, sp, sn)
    return model(local_embedding=fx, local_positions=pos, geometry=pos)


_TIDX = 3                          # temperature channel in [U_x,U_y,U_z,T,p_rgh]


def _roughness_np(field, coords, k=8, cap=1024, seed=0):
    """Mean local slope of a field on a point cloud: mean_j |f_i-f_j| / |x_i-x_j| over k-NN.
    A fidelity proxy for high-frequency content (MSE models over-smooth -> lower roughness).
    field:[M], coords:[M,3] -> scalar."""
    M = field.shape[0]
    if M > cap:
        sel = np.random.default_rng(seed).choice(M, cap, replace=False)
        field, coords = field[sel], coords[sel]
        M = cap
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    dist = np.sqrt(np.take_along_axis(d2, nn, axis=1)) + 1e-9
    df = np.abs(field[nn] - field[:, None])
    return float((df / dist).mean())


@torch.no_grad()
def evaluate(model, split, dev, bs, n_points, ymean, ystd, channels, geom_mode="sdf",
             is_refiner=False, fidelity=False):
    """Per-channel nRMSE + physical RMSE. If fidelity=True, also near-wall T-RMSE and the
    T roughness ratio (pred/gt local-slope; <1 = over-smoothed). Fidelity is skipped during
    per-epoch val (it's only needed for final/eval-only reporting)."""
    if split is None:
        return None
    model.eval()
    idx = _subsample_idx(split["coords"].shape[1], n_points, seed=123)
    # near-wall band = closest 15% of points by SDF (percentile is robust to mesh density,
    # unlike a fixed metric threshold — the CFD mesh clusters near the heatsink)
    nw_thresh = float(np.percentile(split["sdf"][:, idx], 15)) if fidelity else None
    se_norm = np.zeros(5); se_phys = np.zeros(5); ss_norm = np.zeros(5); count = 0
    se_T_nw = 0.0; cnt_nw = 0                                           # near-wall T (physical)
    rough_pred = 0.0; rough_gt = 0.0; nrough = 0                        # T roughness accumulators
    S = split["coords"].shape[0]
    for s in range(0, S, bs):
        sl = slice(s, s + bs)
        pos = torch.tensor(split["coords"][sl][:, idx], device=dev)
        fx = torch.tensor(split["feats"][sl][:, idx], device=dev)
        y = torch.tensor(split["y"][sl][:, idx], device=dev)
        sp = torch.tensor(split["surf_pts"][sl], device=dev) if geom_mode == "surface" else None
        sn = torch.tensor(split["surf_normals"][sl], device=dev) if geom_mode == "surface" else None
        if is_refiner:
            cond = fx if fx.shape[-1] > 0 else None
            pred = model.predict(cond, pos, sp, sn, sample=False)       # PDE-Refiner inference
        else:
            pred = _forward(model, geom_mode, pos, fx, sp, sn)
        err = (pred - y).cpu().numpy()
        gt = y.cpu().numpy()
        se_norm += (err ** 2).sum(axis=(0, 1))
        ss_norm += (gt ** 2).sum(axis=(0, 1))
        se_phys += ((err * ystd[None, None, :]) ** 2).sum(axis=(0, 1))
        count += err.shape[0] * err.shape[1]

        if fidelity:  # near-wall T-RMSE + T roughness ratio (physical T)
            sdf_b = split["sdf"][sl][:, idx]                            # [b,M] metres
            errT = err[..., _TIDX] * ystd[_TIDX]                        # [b,M] physical K
            nw = sdf_b < nw_thresh
            se_T_nw += float((errT[nw] ** 2).sum()); cnt_nw += int(nw.sum())
            coords_b = split["coords"][sl][:, idx]
            predT = pred[..., _TIDX].cpu().numpy() * ystd[_TIDX]
            gtT = gt[..., _TIDX] * ystd[_TIDX]
            for bi in range(coords_b.shape[0]):
                rough_pred += _roughness_np(predT[bi], coords_b[bi])
                rough_gt += _roughness_np(gtT[bi], coords_b[bi])
                nrough += 1

    rmse_norm = np.sqrt(se_norm / count)
    rmse_phys = np.sqrt(se_phys / count)
    rel_l2 = np.sqrt(se_norm / (ss_norm + 1e-12))
    out = {"nRMSE": dict(zip(channels, rmse_norm.tolist())),
           "RMSE_phys": dict(zip(channels, rmse_phys.tolist())),
           "relL2": dict(zip(channels, rel_l2.tolist()))}
    if fidelity:
        out["T_RMSE_nearwall"] = float(np.sqrt(se_T_nw / max(cnt_nw, 1)))
        out["nearwall_frac"] = cnt_nw / max(count, 1)
        out["T_roughness_ratio"] = rough_pred / max(rough_gt, 1e-12)
    return out


def _fmt(m):
    s = (f"T_nRMSE={m['nRMSE']['T']:.4f} T_RMSE={m['RMSE_phys']['T']:.3f}K | "
         f"U nRMSE={np.mean([m['nRMSE'][c] for c in ('U_x','U_y','U_z')]):.4f} | "
         f"p_rgh nRMSE={m['nRMSE']['p_rgh']:.4f} | mean_nRMSE={np.mean(list(m['nRMSE'].values())):.4f}")
    if "T_RMSE_nearwall" in m:                                    # fidelity metrics
        s += (f" || nearwall_T_RMSE={m['T_RMSE_nearwall']:.3f}K "
              f"(frac {m['nearwall_frac']:.2f}) | T_rough_ratio={m['T_roughness_ratio']:.3f}")
    return s


# --------------------------------------------------------------------------- wsd lr
class WSDScheduler:
    """Warmup - Stable - Decay learning-rate controller (the modern flexible-horizon
    schedule used by DeepSeek/MiniCPM etc.).

    - warmup : lr ramps 0 -> peak linearly over `warmup_epochs`
    - stable : lr held at peak; training continues at full LR *as long as val keeps
               improving*. Decay is triggered when val plateaus for `stable_patience`
               epochs, OR when we approach `max_epochs` (so we always anneal before the cap).
    - decay  : lr cosine-anneals peak -> min_lr over `decay_epochs`, then STOP. WSD
               typically gets its largest val drop here.

    Call `set_lr(ep)` before each train epoch, then `update(improved)` after validating.
    `update` returns True when training should stop.
    """

    def __init__(self, opt, peak_lr, warmup_epochs, decay_epochs, stable_patience,
                 min_lr, max_epochs):
        assert warmup_epochs >= 1 and decay_epochs >= 1 and stable_patience >= 1, \
            "warmup/decay/stable_patience must all be >= 1"
        assert 0 < min_lr < peak_lr, f"need 0 < min_lr({min_lr}) < peak_lr({peak_lr})"
        assert max_epochs > warmup_epochs + decay_epochs, \
            "max_epochs must exceed warmup+decay so a stable phase can exist"
        self.opt = opt
        self.peak, self.min_lr = peak_lr, min_lr
        self.warmup, self.decay_len, self.stable_patience = warmup_epochs, decay_epochs, stable_patience
        self.max_epochs = max_epochs
        self.phase = "warmup"
        self.stable_bad = 0
        self.decay_step = 0
        self.cur_lr = 0.0

    def state_dict(self):
        return {"phase": self.phase, "stable_bad": self.stable_bad,
                "decay_step": self.decay_step, "cur_lr": self.cur_lr}

    def load_state_dict(self, s):
        self.phase, self.stable_bad = s["phase"], s["stable_bad"]
        self.decay_step, self.cur_lr = s["decay_step"], s["cur_lr"]

    def _apply(self, lr):
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.cur_lr = lr

    def set_lr(self, ep):
        if self.phase == "warmup":
            self._apply(self.peak * ep / self.warmup)
        elif self.phase == "stable":
            self._apply(self.peak)
        elif self.phase == "decay":
            prog = self.decay_step / self.decay_len                  # 0 .. ->1
            self._apply(self.min_lr + 0.5 * (self.peak - self.min_lr) * (1 + np.cos(np.pi * prog)))
        return self.cur_lr

    def update(self, ep, improved):
        """Advance the phase machine. Returns True if training should stop."""
        if self.phase == "warmup":
            if ep >= self.warmup:
                self.phase = "stable"; self.stable_bad = 0
                print(f"[WSD] warmup done @ ep{ep} -> STABLE (lr={self.peak:.2e})", flush=True)
        elif self.phase == "stable":
            self.stable_bad = 0 if improved else self.stable_bad + 1
            near_cap = ep >= self.max_epochs - self.decay_len
            if self.stable_bad >= self.stable_patience or near_cap:
                why = "val plateaued" if self.stable_bad >= self.stable_patience else "approaching max_epochs"
                self.phase = "decay"; self.decay_step = 0
                print(f"[WSD] {why} @ ep{ep} -> DECAY over {self.decay_len} epochs", flush=True)
        elif self.phase == "decay":
            self.decay_step += 1
            if self.decay_step >= self.decay_len:
                print(f"[WSD] decay complete @ ep{ep} -> STOP", flush=True)
                return True
        return False


# --------------------------------------------------------------------------- model
def _dims_from(cfg, a):
    """Model hyperparams, preferring a checkpoint's stored config, falling back to args."""
    g = lambda k, d: cfg.get(k, d)
    return dict(n_hidden=g("n_hidden", a.n_hidden), n_layers=g("n_layers", a.n_layers),
                n_head=g("n_head", a.n_head), slice_num=g("slice_num", a.slice_num),
                n_hidden_local=g("n_hidden_local", a.n_hidden_local),
                radii=g("radii", a.radii), neighbors=g("neighbors", a.neighbors),
                no_local=g("no_local", a.no_local),
                d_surf=g("d_surf", a.d_surf), d_geo=g("d_geo", a.d_geo),
                surf_layers=g("surf_layers", a.surf_layers), surf_head=g("surf_head", a.surf_head),
                geo_head=g("geo_head", a.geo_head), knn_k=g("knn_k", a.knn_k),
                geom_wiring=g("geom_wiring", a.geom_wiring),
                refine=g("refine", a.refine), refine_steps=g("refine_steps", a.refine_steps),
                sigma_min=g("sigma_min", a.sigma_min))


def _build_model(geom_mode, cond_dim, feat_dim, d, dev, GeoTransolver):
    """Return (model, is_refiner). Refiner is only valid with geom_mode='surface'."""
    refine = d.get("refine", False)
    if geom_mode == "surface":
        from kuber.surface_geotransolver import SurfaceGeoTransolver
        net = SurfaceGeoTransolver(
            cond_dim=feat_dim, out_dim=5, geom_wiring=d["geom_wiring"], refiner_inputs=refine,  # feat_dim = cond + extent (+sdf); all per-node scalars
            d_surf=d["d_surf"], n_surf_layers=d["surf_layers"], n_surf_head=d["surf_head"],
            d_geo=d["d_geo"], geo_head=d["geo_head"], k=d["knn_k"],
            n_hidden=d["n_hidden"], n_layers=d["n_layers"], n_head=d["n_head"], slice_num=d["slice_num"],
            include_local_features=not d["no_local"], n_hidden_local=d["n_hidden_local"],
            radii=d["radii"], neighbors=d["neighbors"]).to(dev)
        if refine:
            from kuber.pde_refiner import PDERefiner
            return PDERefiner(net, num_steps=d["refine_steps"], sigma_min=d["sigma_min"], out_dim=5).to(dev), True
        return net, False
    assert not refine, "--refine requires --geom_mode surface"
    core = GeoTransolver(
        functional_dim=feat_dim, out_dim=5, geometry_dim=3,
        n_layers=d["n_layers"], n_hidden=d["n_hidden"], n_head=d["n_head"], slice_num=d["slice_num"],
        use_te=False, include_local_features=not d["no_local"],
        n_hidden_local=d["n_hidden_local"], radii=d["radii"], neighbors_in_radius=d["neighbors"]).to(dev)
    return core, False


# ------------------------------------------------------------------------- eval-only
def run_eval_only(a, dev, GeoTransolver):
    """Load a trained checkpoint and evaluate on <difficulty>'s src.test + tgt.test.
    Normalizer is src.train-only (difficulty-invariant), so a medium-trained model is
    evaluated exactly on easy/hard tgt.test."""
    ck = torch.load(a.eval_only, map_location=dev, weights_only=False)
    mck = ck["meta"]; cfg = mck.get("config", {})
    cond_keys = mck["cond_keys"]
    # geom_mode: prefer stored; fall back to legacy use_sdf flag on old checkpoints
    geom_mode = cfg.get("geom_mode", "sdf" if cfg.get("use_sdf", False) else "none")
    no_local = cfg.get("no_local", False)
    n_surf = cfg.get("n_surf", a.n_surf)
    print(f"[eval_only] ckpt={a.eval_only} difficulty={a.difficulty} geom_mode={geom_mode} "
          f"no_local={no_local} cond_keys={cond_keys}", flush=True)
    data, meta = load_dataset(a.data, a.splits, a.difficulty, cond_keys=cond_keys,
                              geom_mode=geom_mode, n_surf=n_surf,
                              extent_feats=cfg.get("extent_feats", False))
    ymean, ystd, channels = meta["ymean"], meta["ystd"], meta["channels"]
    model, is_refiner = _build_model(geom_mode, meta["cond_dim"], meta["feat_dim"], _dims_from(cfg, a), dev, GeoTransolver)
    model.load_state_dict(ck["model"])
    n_points = cfg.get("n_points", a.n_points)
    mid = evaluate(model, data["src_test"], dev, a.batch_size, n_points, ymean, ystd, channels, geom_mode, is_refiner, fidelity=True)
    mood = evaluate(model, data["tgt_test"], dev, a.batch_size, n_points, ymean, ystd, channels, geom_mode, is_refiner, fidelity=True)
    print(f"\n==== EVAL ({a.difficulty}, {'noLS' if no_local else 'multiscale'}, {geom_mode}) ====")
    print(f"IN-DIST  src.test : {_fmt(mid)}")
    print(f"OOD      tgt.test : {_fmt(mood)}")
    gap = np.mean(list(mood["nRMSE"].values())) - np.mean(list(mid["nRMSE"].values()))
    print(f"OOD gap (mean nRMSE) = {gap:+.4f}")
    tag = f"{a.difficulty}{'_noLS' if no_local else ''}_{geom_mode}"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"results_{tag}_eval.json").write_text(json.dumps(
        {"difficulty": a.difficulty, "multiscale": not no_local, "geom_mode": geom_mode,
         "eval_of": str(a.eval_only), "in_dist": mid, "ood": mood, "ood_gap": float(gap)}, indent=2))
    print(f"[saved] results_{tag}_eval.json", flush=True)


# ----------------------------------------------------------------------------- train
def run(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    GeoTransolver = _import_geotransolver(a.pnemo_path)
    if a.eval_only:
        return run_eval_only(a, dev, GeoTransolver)

    # config (smoke shrinks everything for a laptop E2E check)
    geom_mode = a.geom_mode
    if a.smoke:
        n_points, n_hidden, n_layers, slice_num, bs, epochs = 2048, 64, 4, 32, 2, 3
        n_hidden_local = 16
        d_surf, d_geo, surf_layers, knn_k = 32, 16, 1, 6
    else:
        n_points, n_hidden, n_layers, slice_num, bs, epochs = (
            a.n_points, a.n_hidden, a.n_layers, a.slice_num, a.batch_size, a.epochs)
        n_hidden_local = a.n_hidden_local
        d_surf, d_geo, surf_layers, knn_k = a.d_surf, a.d_geo, a.surf_layers, a.knn_k

    # record the ACTUAL dims used (smoke overrides them) so eval_only rebuilds correctly
    used_cfg = {**vars(a), "n_points": n_points, "n_hidden": n_hidden, "n_layers": n_layers,
                "slice_num": slice_num, "n_hidden_local": n_hidden_local,
                "d_surf": d_surf, "d_geo": d_geo, "surf_layers": surf_layers, "knn_k": knn_k}

    print(f"[load] difficulty={a.difficulty} data={a.data} geom_mode={geom_mode}", flush=True)
    ckeys = a.cond_keys.split(",") if getattr(a, "cond_keys", None) else None
    data, meta = load_dataset(a.data, a.splits, a.difficulty, cond_keys=ckeys, geom_mode=geom_mode,
                              n_surf=a.n_surf, drop_geom_scalars=a.drop_geom_scalars,
                              extent_feats=a.extent_feats)
    ymean, ystd, channels = meta["ymean"], meta["ystd"], meta["channels"]
    print(f"[data] N={meta['N']} cond_dim={meta['cond_dim']} feat_dim={meta['feat_dim']} "
          f"cond_keys={meta['cond_keys']} geom_mode={geom_mode} n_surf={meta['n_surf']}", flush=True)
    for k, v in data.items():
        print(f"       {k}: {0 if v is None else v['coords'].shape[0]} cases", flush=True)
    print(f"[cfg]  dev={dev} n_points={n_points} n_hidden={n_hidden} n_layers={n_layers} "
          f"slice={slice_num} bs={bs} multiscale={not a.no_local}", flush=True)

    dims = dict(n_hidden=n_hidden, n_layers=n_layers, n_head=a.n_head, slice_num=slice_num,
                n_hidden_local=n_hidden_local, radii=a.radii, neighbors=a.neighbors, no_local=a.no_local,
                d_surf=d_surf, d_geo=d_geo, surf_layers=surf_layers, surf_head=a.surf_head,
                geo_head=a.geo_head, knn_k=knn_k, geom_wiring=a.geom_wiring,
                refine=a.refine, refine_steps=a.refine_steps, sigma_min=a.sigma_min)
    model, is_refiner = _build_model(geom_mode, meta["cond_dim"], meta["feat_dim"], dims, dev, GeoTransolver)
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] {'SurfaceGeoTransolver' if geom_mode == 'surface' else 'GeoTransolver'}"
          f"{'+PDERefiner' if is_refiner else ''} params={nparams:.2f}M", flush=True)

    if a.init_from:  # warm-restart: load weights, fresh optimizer + WSD cycle
        sd = torch.load(a.init_from, map_location=dev, weights_only=False)["model"]
        model.load_state_dict(sd)
        print(f"[init_from] warm-started weights from {a.init_from}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    # WSD schedule params (smoke uses tiny values so all 3 phases show in a few epochs)
    if a.smoke:
        warmup_ep, decay_ep, stable_pat, max_ep = 1, 2, 1, 8
    else:
        warmup_ep, decay_ep, stable_pat, max_ep = a.warmup, a.decay, a.stable_patience, epochs
    wsd = WSDScheduler(opt, peak_lr=a.lr, warmup_epochs=warmup_ep, decay_epochs=decay_ep,
                       stable_patience=stable_pat, min_lr=a.min_lr, max_epochs=max_ep)
    print(f"[wsd]  peak_lr={a.lr:.2e} warmup={warmup_ep} stable_patience={stable_pat} "
          f"decay={decay_ep} min_lr={a.min_lr:.2e} max_epochs={max_ep}", flush=True)

    tr = data["src_train"]
    Str = tr["coords"].shape[0]

    def train_epoch(seed):
        model.train()
        idx = _subsample_idx(tr["coords"].shape[1], n_points, seed)     # resample points/epoch
        order = np.random.default_rng(seed).permutation(Str)
        tot, n = 0.0, 0
        for s in range(0, Str, bs):
            b = order[s:s + bs]
            pos = torch.tensor(tr["coords"][b][:, idx], device=dev)
            fx = torch.tensor(tr["feats"][b][:, idx], device=dev)
            y = torch.tensor(tr["y"][b][:, idx], device=dev)
            sp = torch.tensor(tr["surf_pts"][b], device=dev) if geom_mode == "surface" else None
            sn = torch.tensor(tr["surf_normals"][b], device=dev) if geom_mode == "surface" else None
            if is_refiner:
                cond = fx if fx.shape[-1] > 0 else None
                loss = model.loss(cond, pos, sp, sn, y)                 # PDE-Refiner training loss
            else:
                pred = _forward(model, geom_mode, pos, fx, sp, sn)
                loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * b.shape[0]; n += b.shape[0]
        return tot / n

    # --- time-one-epoch mode: measure real per-epoch cost, then exit ------------
    if a.time_one_epoch:
        if dev == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        tl = train_epoch(0)
        if dev == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        peak = torch.cuda.max_memory_allocated() / 1e9 if dev == "cuda" else 0
        print(f"\n[TIME] 1 epoch = {dt:.1f}s | train_loss={tl:.4e} | peak_VRAM={peak:.1f}GB "
              f"| est 100ep={dt*100/60:.1f}min 200ep={dt*200/60:.1f}min", flush=True)
        return

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.difficulty}{'_noLS' if a.no_local else ''}_{geom_mode}{'_noscal' if a.drop_geom_scalars else ''}"
    ckpt = out / f"geot_{tag}.pt"
    resume_pt = out / "resume.pt"
    best_val = float("inf"); best_ep = 0; start_ep = 1
    if getattr(a, "resume", False) and resume_pt.exists():   # crash recovery: continue mid-run
        rs = torch.load(resume_pt, map_location=dev, weights_only=False)
        model.load_state_dict(rs["model"]); opt.load_state_dict(rs["opt"]); wsd.load_state_dict(rs["wsd"])
        best_val, best_ep, start_ep = rs["best_val"], rs["best_ep"], rs["epoch"] + 1
        print(f"[resume] continuing from ep {rs['epoch']} -> {start_ep} (best@{best_ep}={best_val:.4f}, wsd={wsd.phase})", flush=True)
    for ep in range(start_ep, max_ep + 1):
        lr = wsd.set_lr(ep)                                   # WSD sets LR for this epoch
        t0 = time.time()
        tl = train_epoch(ep)
        vm = evaluate(model, data["src_val"], dev, bs, n_points, ymean, ystd, channels, geom_mode, is_refiner)
        vloss = np.mean(list(vm["nRMSE"].values()))
        improved = vloss < best_val - 1e-5
        if improved:
            best_val, best_ep = vloss, ep
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "meta": {
                "cond_keys": meta["cond_keys"], "ymean": ymean, "ystd": ystd,
                "channels": channels, "config": used_cfg}}, ckpt)
        # diagnostic: log OOD (tgt.test) trajectory to see if it peaks before the in-dist best
        ood_str = ""
        if a.track_ood and ep >= a.ood_start and ep % a.ood_every == 0 and data["tgt_test"] is not None:
            om = evaluate(model, data["tgt_test"], dev, bs, n_points, ymean, ystd, channels, geom_mode, is_refiner)
            ood_str = (f" || OOD T {om['nRMSE']['T']:.3f} mean "
                       f"{np.mean(list(om['nRMSE'].values())):.3f}")
        print(f"ep {ep:3d} | {time.time()-t0:5.1f}s | lr {lr:.2e} {wsd.phase:6s} | "
              f"train {tl:.4e} | val {_fmt(vm)} | best@{best_ep}({best_val:.4f})"
              f"{' *' if improved else ''}{ood_str}", flush=True)
        stop = wsd.update(ep, improved)                       # returns True -> decay done, stop
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "wsd": wsd.state_dict(),
                    "epoch": ep, "best_val": best_val, "best_ep": best_ep}, resume_pt)  # crash-safe: resume from here
        if stop:
            break

    # --- final eval on best checkpoint: in-dist (src.test) + OOD (tgt.test) -----
    print(f"\n[eval] loading best checkpoint (ep {best_ep}) for src.test + tgt.test", flush=True)
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model"])
    mid = evaluate(model, data["src_test"], dev, bs, n_points, ymean, ystd, channels, geom_mode, is_refiner, fidelity=True)
    mood = (evaluate(model, data["tgt_test"], dev, bs, n_points, ymean, ystd, channels, geom_mode, is_refiner, fidelity=True)
            if data["tgt_test"] is not None else None)
    print(f"\n==== RESULTS ({a.difficulty}, {'noLS' if a.no_local else 'multiscale'}, {geom_mode}) ====")
    print(f"IN-DIST  src.test : {_fmt(mid)}")
    if mood is not None:
        print(f"OOD      tgt.test : {_fmt(mood)}")
        gap = float(np.mean(list(mood["nRMSE"].values())) - np.mean(list(mid["nRMSE"].values())))
        print(f"OOD gap (mean nRMSE, OOD - in-dist) = {gap:+.4f}")
    else:
        gap = 0.0
        print("OOD      tgt.test : (none — pretrain split, no OOD held out)")
    results = {"difficulty": a.difficulty, "multiscale": not a.no_local, "geom_mode": geom_mode,
               "drop_geom_scalars": a.drop_geom_scalars,
               "best_ep": best_ep, "params_M": nparams, "in_dist": mid, "ood": mood, "ood_gap": float(gap)}
    (out / f"results_{tag}.json").write_text(json.dumps(results, indent=2))
    print(f"[saved] {ckpt} + results json", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="dir of case_XXXX.npz")
    p.add_argument("--splits", required=True, help="splits.json")
    p.add_argument("--difficulty", default="medium", choices=["easy", "medium", "hard"])
    p.add_argument("--pnemo_path", default=None, help="path to physicsnemo clone if not pip-installed")
    p.add_argument("--out", default="outputs")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--time_one_epoch", action="store_true")
    p.add_argument("--eval_only", default=None, help="path to a checkpoint: eval it on this difficulty's src.test+tgt.test and exit")
    p.add_argument("--init_from", default=None, help="warm-restart: load model weights from this checkpoint, then train fresh (new optimizer + WSD)")
    p.add_argument("--track_ood", action="store_true", help="diagnostic: eval tgt.test each epoch and log OOD trajectory")
    p.add_argument("--ood_every", type=int, default=1, help="log OOD every K epochs when --track_ood")
    p.add_argument("--ood_start", type=int, default=0, help="start OOD logging from this epoch")
    p.add_argument("--no_local", action="store_true", help="GEOT-noLS ablation (multiscale off)")
    p.add_argument("--geom_mode", choices=["sdf", "dsdf", "surface", "none"], default="sdf",
                   help="geometry feature: scalar SDF | directional SDF (dist+grad, 4ch) | learned surface branch | none")
    p.add_argument("--drop_geom_scalars", action="store_true",
                   help="keep only solidTemp in conditions (drop fins/gap/thickness/height) — OOD test")
    p.add_argument("--cond_keys", default=None,
                   help="comma-separated conditioning keys to force (overrides auto-select), "
                        "e.g. fins,gap,height2,thickness_fins,solidTemp — for pretrain/finetune transfer")
    p.add_argument("--extent_feats", action="store_true",
                   help="append per-case physical bbox extents (log, z-scored) to conditioning — "
                        "restores aspect/scale erased by per-axis framing (multi-geometry: thin cold plates)")
    p.add_argument("--resume", action="store_true",
                   help="crash recovery: if <out>/resume.pt exists, continue mid-run (model+opt+WSD+epoch). "
                        "Saved every epoch; combine with --init_from for first-launch warm start.")
    p.add_argument("--n_surf", type=int, default=2048, help="analytic surface points per case (surface mode)")
    # surface-branch dims (geom_mode=surface)
    p.add_argument("--d_surf", type=int, default=128, help="surface encoder token dim")
    p.add_argument("--surf_layers", type=int, default=2, help="surface encoder self-attn layers")
    p.add_argument("--surf_head", type=int, default=4, help="surface encoder heads")
    p.add_argument("--d_geo", type=int, default=64, help="per-point geometry descriptor dim")
    p.add_argument("--geo_head", type=int, default=4, help="geometry cross-attention heads")
    p.add_argument("--knn_k", type=int, default=16, help="nearest surface tokens per volume point")
    p.add_argument("--geom_wiring", choices=["concat", "deep"], default="concat",
                   help="surface conditioning: concat (shallow) or deep (AB-UPT native geometry branch on surface cloud)")
    # PDE-Refiner (generative denoising refinement; requires --geom_mode surface)
    p.add_argument("--refine", action="store_true", help="wrap model in PDE-Refiner (denoising refinement)")
    p.add_argument("--refine_steps", type=int, default=3, help="K refinement/denoising steps")
    p.add_argument("--sigma_min", type=float, default=1e-2, help="PDE-Refiner min noise amplitude")
    # full config (study_plan.md sec 6)
    p.add_argument("--n_points", type=int, default=16384)
    p.add_argument("--n_hidden", type=int, default=256)
    p.add_argument("--n_layers", type=int, default=12)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--slice_num", type=int, default=64)
    p.add_argument("--n_hidden_local", type=int, default=32)
    p.add_argument("--radii", type=float, nargs="+", default=[0.05, 0.25])
    p.add_argument("--neighbors", type=int, nargs="+", default=[8, 32])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3, help="peak LR")
    # WSD schedule (convergence-driven, flexible horizon)
    p.add_argument("--warmup", type=int, default=5, help="linear warmup epochs")
    p.add_argument("--stable_patience", type=int, default=25,
                   help="epochs of no val improvement at peak LR before triggering decay")
    p.add_argument("--decay", type=int, default=30, help="cosine decay epochs (peak->min_lr) then stop")
    p.add_argument("--min_lr", type=float, default=1e-6, help="LR floor at end of decay")
    p.add_argument("--epochs", type=int, default=1000, help="safety cap; decay auto-triggers before it")
    run(p.parse_args())
