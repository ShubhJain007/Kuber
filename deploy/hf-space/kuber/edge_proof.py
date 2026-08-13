"""Numerical-stability proof: no gradient explosion at edges + generalization to
never-seen geometries (OOD tgt.test).

Reuses the trainer's exact model construction / data loading / forward so the
predictions are identical to the reported eval. Adds a per-node temperature
gradient-magnitude estimate (k-NN local slope) and compares predicted vs CFD
ground-truth gradients in three regions:
  - edge/near-wall band  (closest 15% of nodes by |SDF| — fin tips, corners, walls)
  - steep-gradient region (top 5% of nodes by TRUE |grad T| — where a blow-up shows)
  - bulk                 (everything else)

Proof of "no gradient explosion":
  * max|gradT|_pred / max|gradT|_gt  ~<= 1        (pred never spikes past physical)
  * p99.9 ratio and edge mean ratio  ~ 1          (faithful, not over-smoothed, not exploding)
  * explosion_frac (nodes with |gradT|_pred > 2*max_gt) == 0
  * zero NaN/Inf in the predicted field

Runs on CPU by default (does not touch the GPU).

Usage (from the repo root, in your PhysicsNeMo python env):
  python -m kuber.edge_proof --ckpt <ckpt.pt> --data <dir> --splits <splits.json> \
      --difficulty medium --out edge_proof_surface.json --device cpu
"""
import argparse, json
from types import SimpleNamespace
import numpy as np
import torch

from kuber.train_simshift import (load_dataset, _build_model, _dims_from, _forward,
                                 _import_geotransolver)

TIDX = 3  # temperature channel in [U_x, U_y, U_z, T, p_rgh]


def _arg_defaults():
    """Mirror the trainer argparser defaults so _dims_from's fallbacks are defined."""
    return SimpleNamespace(
        n_points=16384, n_hidden=256, n_layers=12, n_head=8, slice_num=64,
        n_hidden_local=32, radii=[0.05, 0.25], neighbors=[8, 32], no_local=False,
        n_surf=2048, d_surf=128, surf_layers=2, surf_head=4, d_geo=64, geo_head=4,
        knn_k=16, geom_wiring="concat", refine=False, refine_steps=3, sigma_min=1e-2,
        batch_size=1)


def grad_mag(field, coords, k=8, chunk=2048):
    """Per-node local gradient magnitude proxy: mean_j |f_i - f_j| / |x_i - x_j| over kNN.
    Chunked torch kNN (no scipy); self is the nearest neighbour and is dropped.
    Distances are floored at 0.5x the median nearest-neighbour spacing so that a pair of
    near-coincident subsampled nodes cannot manufacture a spurious 1/eps slope (that would
    be an artefact of the ESTIMATOR, not the field)."""
    c = torch.as_tensor(coords, dtype=torch.float32)
    f = torch.as_tensor(field, dtype=torch.float32)
    M = c.shape[0]
    dists, idxs = torch.empty(M, k), torch.empty(M, k, dtype=torch.long)
    for s in range(0, M, chunk):
        d = torch.cdist(c[s:s + chunk], c)                    # [b,M]
        dd, ii = torch.topk(d, k + 1, dim=1, largest=False)
        dists[s:s + chunk], idxs[s:s + chunk] = dd[:, 1:], ii[:, 1:]   # drop self
    floor = 0.5 * float(dists[:, 0].median())                 # characteristic node spacing
    dists = dists.clamp_min(max(floor, 1e-9))
    df = (f[idxs] - f[:, None]).abs()                         # [M,k]
    return (df / dists).mean(dim=1).numpy()


@torch.no_grad()
def analyze(model, split, geom_mode, dev, ystd, tag):
    if split is None:
        return None
    model.eval()
    S, N = split["coords"].shape[0], split["coords"].shape[1]
    acc = {k: [] for k in ("edge_ratio", "bulk_ratio", "steep_p999_ratio", "max_ratio",
                            "explosion_frac", "value_overshoot_frac", "value_worst_overshoot",
                            "edge_gp", "edge_gg", "T_rmse_edge", "nan_inf")}
    for i in range(S):
        pos = torch.tensor(split["coords"][i:i + 1], device=dev)
        fx = torch.tensor(split["feats"][i:i + 1], device=dev)
        sp = torch.tensor(split["surf_pts"][i:i + 1], device=dev) if geom_mode == "surface" else None
        sn = torch.tensor(split["surf_normals"][i:i + 1], device=dev) if geom_mode == "surface" else None
        pred = _forward(model, geom_mode, pos, fx, sp, sn)[0].cpu().numpy()   # [N,5] normalized
        y = split["y"][i]                                                     # [N,5] normalized

        acc["nan_inf"].append(int((~np.isfinite(pred)).sum()))

        coords = split["coords"][i].astype(np.float64)
        sdf = split["sdf"][i]
        predT = pred[:, TIDX] * ystd[TIDX]                                    # physical K (offset cancels in grad)
        gtT = y[:, TIDX] * ystd[TIDX]
        gp = grad_mag(predT, coords)
        gg = grad_mag(gtT, coords)

        nw = sdf < np.percentile(sdf, 15)                                     # edge / near-wall band
        steep = gg >= np.percentile(gg, 95)                                   # steepest true gradients
        bulk = ~nw
        gg_max = max(gg.max(), 1e-9)

        acc["edge_ratio"].append(gp[nw].mean() / max(gg[nw].mean(), 1e-9))
        acc["bulk_ratio"].append(gp[bulk].mean() / max(gg[bulk].mean(), 1e-9))
        acc["steep_p999_ratio"].append(np.percentile(gp, 99.9) / max(np.percentile(gg, 99.9), 1e-9))
        acc["max_ratio"].append(gp.max() / gg_max)                      # distance-floored -> real
        acc["explosion_frac"].append(float((gp > 2.0 * gg_max).mean()))
        acc["edge_gp"].append(float(gp[nw].mean()))
        acc["edge_gg"].append(float(gg[nw].mean()))
        # value stability: predicted T must stay inside the physical value range set by the BCs
        # (a blow-up would push T beyond the wall/ambient envelope). Fraction of nodes outside
        # the CFD T-range (+/-10% margin), and the worst overshoot as a fraction of that range.
        gt_lo, gt_hi = float(y[:, TIDX].min()), float(y[:, TIDX].max())
        rng = gt_hi - gt_lo + 1e-9
        pT = pred[:, TIDX]
        acc["value_overshoot_frac"].append(float(((pT > gt_hi + 0.1 * rng) | (pT < gt_lo - 0.1 * rng)).mean()))
        acc["value_worst_overshoot"].append(max(0.0, float(max(pT.max() - gt_hi, gt_lo - pT.min()) / rng)))
        # temperature RMSE in the edge band (physical K)
        errT = (pred[:, TIDX] - y[:, TIDX]) * ystd[TIDX]
        acc["T_rmse_edge"].append(float(np.sqrt((errT[nw] ** 2).mean())))

    summ = {k: float(np.mean(v)) for k, v in acc.items() if k != "nan_inf"}
    summ["nan_inf_total"] = int(np.sum(acc["nan_inf"]))
    summ["cases"] = S
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--difficulty", default="medium")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    GeoTransolver = _import_geotransolver(None)
    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    mck = ck["meta"]; cfg = mck.get("config", {})
    cond_keys = mck["cond_keys"]
    geom_mode = cfg.get("geom_mode", "sdf" if cfg.get("use_sdf", False) else "none")
    n_surf = cfg.get("n_surf", 2048)
    print(f"[edge_proof] ckpt={a.ckpt} geom_mode={geom_mode} cond_keys={cond_keys}", flush=True)

    data, meta = load_dataset(a.data, a.splits, a.difficulty, cond_keys=cond_keys,
                              geom_mode=geom_mode, n_surf=n_surf)
    ystd = meta["ystd"]
    dims = _dims_from(cfg, _arg_defaults())
    model, is_refiner = _build_model(geom_mode, meta["cond_dim"], meta["feat_dim"], dims, dev, GeoTransolver)
    model.load_state_dict(ck["model"])
    nparams = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[edge_proof] params={nparams:.2f}M  running on {dev}", flush=True)

    res = {"ckpt": a.ckpt, "geom_mode": geom_mode, "params_M": nparams, "difficulty": a.difficulty}
    res["in_dist"] = analyze(model, data["src_test"], geom_mode, dev, ystd, "in_dist")
    res["ood"] = analyze(model, data["tgt_test"], geom_mode, dev, ystd, "ood")

    for tag in ("in_dist", "ood"):
        m = res[tag]
        print(f"\n==== {tag} ({m['cases']} cases) ====")
        print(f"  edge |gradT| ratio (pred/gt) : {m['edge_ratio']:.3f}   (1.0=faithful; <1 over-smooth; >>1 explode)")
        print(f"  bulk |gradT| ratio (pred/gt) : {m['bulk_ratio']:.3f}")
        print(f"  steep-region p99.9 ratio     : {m['steep_p999_ratio']:.3f}   (<=~1 => no explosion at peaks)")
        print(f"  max |gradT| ratio (pred/gt)  : {m['max_ratio']:.3f}   (dist-floored; <=~1 => no spurious spike)")
        print(f"  explosion fraction           : {m['explosion_frac']:.2e}  (nodes with |gradT|_pred > 2x physical max)")
        print(f"  value overshoot fraction     : {m['value_overshoot_frac']:.2e}  (T outside CFD range +/-10%)")
        print(f"  worst value overshoot        : {m['value_worst_overshoot']:.3f}   (fraction of CFD T-range)")
        print(f"  edge T-RMSE (physical)       : {m['T_rmse_edge']:.3f} K")
        print(f"  NaN/Inf in predictions       : {m['nan_inf_total']}")

    open(a.out, "w").write(json.dumps(res, indent=2))
    print(f"\n[saved] {a.out}", flush=True)


if __name__ == "__main__":
    main()
