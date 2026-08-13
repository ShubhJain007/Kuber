# Training

How a Kuber surrogate is trained — preprocessing, geometry conditioning, the objective, the
schedule, and how to reproduce it. The model is **SurfaceGeoTransolver** (~14.3 M parameters); the
architecture is in [`MODEL.md`](MODEL.md), the corpus in [`DATASET.md`](DATASET.md).

## What the model learns

From a geometry and its operating conditions, predict the **steady-state fluid field** as a
5-channel vector per node — velocity `(U_x, U_y, U_z)`, temperature `T`, and reduced pressure
`p_rgh` — on the case's 16 384-node point cloud. One model is trained across **both device classes**
(buoyancy-driven heatsinks and forced-convection cold plates); geometry enters *only* through the
surface point cloud, so nothing in the input is device-specific.

## Preprocessing

Every case is turned into model inputs exactly the same way at train and inference time:

1. **Fixed-size clouds.** Each case is a 16 384-node fluid cloud and a 2 048-point solid-surface cloud
   with outward normals (the geometry input).
2. **Per-case unit frame.** Coordinates are mapped into a per-case `[0, 1]` box (per-axis min–max).
   This makes the local kNN / relative-position geometry features scale-invariant, so a big and a
   small heatsink present the same *local* structure to the network. The surface cloud is mapped with
   the **same** per-case frame as the volume, so surface and volume stay registered.
3. **Aspect features (`extent_feats`).** Per-axis framing erases absolute scale and aspect ratio —
   which matters for thin cold-plate ducts (median channel aspect ≈ 36:1, up to 162:1) that would
   otherwise collapse to a cube. We append the log of the case's physical bounding-box extents
   (z-scored over the training set) to the conditioning to restore aspect/scale. Device-agnostic.
4. **Target normalization.** The five targets are z-scored **per channel using statistics computed on
   the source-domain training split only** (`ymean`, `ystd`) — no leakage from val/test or from the
   out-of-distribution target domain. Predictions are de-normalized to physical units (K, m/s, Pa)
   for reporting.
5. **Conditioning normalization.** The physical condition scalars (below) are z-scored over the
   training split (`cmean`, `cstd`).

## Conditioning — one unified vector across device classes

The network is conditioned on physics + a device flag, never on device-specific geometry scalars
(those don't transfer between a finned heatsink and a duct). The unified vector is:

```
rho, mu, Cp, Pr      fluid properties
u_in                 inlet velocity (0 for natural convection)
envTemp              ambient / inlet temperature (K)
solidTemp            wall-temperature BC   (heatsink; 0 when a flux BC is used)
heatFlux             heat-flux BC          (cold plate; 0 when a wall-temp BC is used)
device               0 = heatsink, 1 = cold plate
+ log bbox extents   (3 numbers, from extent_feats)
```

**Geometry is read from the surface cloud, not from scalars.** This is the central design choice: a
learned, local, orientation-aware surface descriptor (encoder → k-nearest-neighbour cross-attention;
see [`MODEL.md`](MODEL.md)) replaces the scalar signed-distance field. Locality is what lets a model
trained on 5–8 fins generalize to 10–12 — a denser fin array is locally "more of the same" fin-gap
patches — and it is what lets a *single* model span heatsinks and cold plates, since both are just
surfaces.

## Objective

Mean-squared error on the five z-normalized targets, averaged over all nodes and the batch:

```
L(θ) = (1 / B·N) · Σ_{b,j} ‖ ŷ_{bj} − y_{bj} ‖²
```

An optional **PDE-Refiner** head (a denoising-refinement objective) can be swapped in where
high-frequency fidelity or calibrated uncertainty is needed; all reported numbers use the one-shot
model. See the paper for details.

## Optimizer & schedule

| setting | value |
|---|---|
| optimizer | Adam, learning rate `1e-3` |
| batch | 4 cases per step (16 384 query nodes each) |
| schedule | **Warmup–Stable–Decay (WSD)** |
| — warmup | 5 epochs, linear |
| — stable | held at `1e-3` until the validation mean-nRMSE plateaus (patience 25 epochs) |
| — decay | 30-epoch cosine decay to `1e-6` |
| checkpoint | best validation mean-nRMSE |
| params | ~14.3 M |

WSD lets us hold a high, constant learning rate through the long "stable" phase and only spend the
cosine decay once validation has actually plateaued — so training length adapts to the run instead of
being fixed up front. Selection is always on the **source-domain validation** split; the
out-of-distribution target split is touched only at final evaluation.

## Compute

A single modern data-center GPU. The multi-geometry model (~3 k cases) runs at roughly **4 minutes
per epoch** and reaches its best validation checkpoint in ~110 epochs (a few GPU-hours end to end).
The SIMSHIFT single-domain models are smaller and faster. Inference is **sub-second to a few seconds
per case on CPU** — the entire reason the surrogate exists (versus CFD-hours).

## What we do *not* do (and disclose)

- **No unsupervised domain adaptation (UDA).** The model never sees the target (test) distribution,
  labelled or unlabelled. The published SIMSHIFT baselines use UDA; Kuber does not — see
  [`RESULTS.md`](RESULTS.md).
- **No test-set leakage.** All normalizers and model selection use the source-domain train/val only.
- **Pretraining is opt-in and disclosed.** One ablation pretrains on the broader corpus and
  fine-tunes — and we never pretrain on any held-out evaluation split. That row is labelled
  everywhere it appears.

## Reproduce

The training entry point is [`../kuber/train_simshift.py`](../kuber/train_simshift.py). The headline
SIMSHIFT run and the multi-geometry run are wrapped as one-command recipes in the
[README](../README.md#recipes); each recipe prints the exact configuration (geometry mode,
conditioning, schedule) it used. A result only counts if it ships with the exact command and
environment that produced it — see [`BENCHMARK.md`](BENCHMARK.md).
