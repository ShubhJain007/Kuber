# The Kuber corpus

A self-generated OpenFOAM conjugate-heat-transfer (CHT) corpus for electronics cooling —
**0 cases from SIMSHIFT or any licensed/scraped source**. Everything here is produced by the
pipeline in [`../datagen/`](../datagen) and is free for research and other noncommercial use.

## Why we generate it ourselves

Training a CHT surrogate needs thousands of solved conjugate-heat-transfer cases with clean,
redistributable provenance. That data did not exist for us to use:

- **The public benchmark ships no generator.** SIMSHIFT publishes an evaluation split but not the
  pipeline that produced it, so there is no license-clean way to *scale* it.
- **Commercial datasets are encumbered.** The heatsink / cold-plate CFD data we could otherwise find
  is Fluent/COMSOL-derived — licensing that forbids redistribution or model training.

So we built an OpenFOAM pipeline (GPL solver; the *output data is ours*) that reproduces the SIMSHIFT
heatsink distribution and extends it to new fluids, regimes, and a second device class. Every case is
self-generated — **0 cases from SIMSHIFT or any licensed/scraped source** — which is what lets us
release the corpus for research and hold contributors to the same "no licensed data" rule.

## What a case is

One `.npz` per case, sampled to a fixed **16 384-node** point cloud in the fluid domain:

| key | shape | meaning |
|---|---|---|
| `coords` | `[16384, 3]` | node positions (metres), physical frame |
| `U` | `[16384, 3]` | velocity `(U_x, U_y, U_z)` (m/s) |
| `T` | `[16384, 1]` | temperature (K) |
| `p_rgh` | `[16384, 1]` | reduced pressure `p − ρgh` (Pa) |
| `surf_pts` | `[2048, 3]` | solid-boundary surface point cloud (the geometry input) |
| `surf_normals` | `[2048, 3]` | outward unit normals at `surf_pts` |
| `sdf` | `[16384]` | signed distance to the solid (heatsink cases; negative inside) — optional |
| `sdf_grad` | `[16384, 3]` | unit SDF gradient (directional-SDF mode) — optional |
| `conditions` | `()` object | JSON dict of operating conditions (see below) |

`conditions` carries the physics the model is conditioned on. Across device classes the unified keys are:

```
rho, mu, Cp, Pr          # fluid properties
u_in                     # inlet velocity (0 for natural convection)
envTemp                  # ambient / inlet temperature (K)
solidTemp                # wall temperature BC (heatsink; K)   — 0 when a flux BC is used
heatFlux                 # heat-flux BC (cold plate; W/m^2)    — 0 when a wall-temp BC is used
device                   # 0 = heatsink, 1 = cold plate
# heatsinks additionally carry geometry scalars: fins, gap, height1, height2, length, width, thickness_fins
```

## Coverage

| axis | coverage |
|---|---|
| fluids | air, water, mineral oil (Pr≈292), glycol — conditioned on Pr, ρ, Cp, μ |
| regimes | natural + forced convection |
| shapes | fins, plate, cube, pin-fin arrays (heatsinks); straight-channel ducts (cold plates) |
| conditions | ambient 290–310 K, wall 340–400 K, fin count 5–14, varied geometry |
| device classes | heatsink (wall-temp BC, buoyancy) and cold plate (heat-flux BC, forced liquid) |
| fidelity | ~1.4 M cells/case (snap-level-2 + 3 prism layers), subsampled to 16 384 nodes |

**Honest limits.** The corpus is air-dominated; oil/glycol are thin (~30–40 cases each), so the model
is strongest on air. The primary OOD axis is fin count; fluids/shapes appear in both train and test.
Cold plates are straight rectangular channels only — serpentine, pin-fin, and parallel-microchannel
topologies are on the roadmap. More liquid data and leave-one-out splits are the roadmap too.

## How it is generated

Pipeline (`../datagen/`), fully resumable, gated by convergence + a physics filter:

```
parametric generator  →  STL  →  blockMesh + snappyHexMesh (+ prism layers)  →  buoyantSimpleFoam  →  .npz
     gen_bsf.py           make_stl.py      mesh_geom.py                          run_sweep_bsf.py    to_npz_bsf.py
```

- **Solver.** OpenFOAM `buoyantSimpleFoam` — a steady, single-region *compressible* solver with the
  `kOmegaSST` turbulence model and air as a `perfectGas`. The solid is not meshed as a second region
  (the multi-region conjugate solver was a throughput dead end); it is modelled as its boundary — a
  **fixed-temperature heated wall** (heatsinks, 340–400 K) or a **heat-flux surface** (cold plates).
  Heatsinks sit in an open air box so the buoyant plume leaves cleanly.
- **The pressure subtlety (the fix that made it work).** Because the solver is compressible, pressure
  must be **absolute** (`p_rgh ≈ 1e5` Pa), not gauge-zero. Initialising at gauge 0 drives the density
  toward zero and blows the solve up — this single detail separated garbage from a stable corpus.
  Startup is further stabilised with temperature/velocity limiters (`fvOptions`) and field relaxation
  (`p_rgh` 0.3, `U` 0.2, `h` 0.5).
- **Meshing.** `blockMesh` background + `snappyHexMesh` to carve the solid, with **3 prism layers** to
  resolve the near-wall thermal boundary layer (~1.4–2 M cells for heatsinks; high-fin OOD cases are
  the largest). Cold plates are structured `blockMesh` rectangular ducts (fast, ~1 min/case).
- **Sampling.** Latin-hypercube over geometry **and** operating conditions — heatsink fin count 5–14
  with width-derived gaps (plus plate and cube variants), channel L/W/H for cold plates, and fluid +
  flow/Reynolds + heat-flux / wall-temperature — which is what fills the coverage table above.
- **Validity gate.** A case is written only after the continuity residual converges (heatsinks
  ~`1e-5`; cold plates `<5e-3`), then a physics filter drops unphysical results — e.g. we removed 156
  cold plates whose coolant exceeded 550 K, since `buoyantSimpleFoam` has no boiling model and those
  temperatures fall outside the physical range.
- **Resumable.** Re-run the same command to continue; `status.json` is rewritten after every case, and
  the raw solve is pruned once the `.npz` is written.

### Mesh-convergence check

Prism layers recover the near-wall hot spot to within **0.1 K of a fine mesh at ~2.7× fewer cells**,
which is why the production corpus uses snap-level-2 + 3 layers:

| mesh | cells | T-max (hot spot) |
|---|---|---|
| snap2 (no layers) | 124 k | 359.5 K |
| **snap2 + 3 prism layers** | 142 k | **378.8 K** |
| snap3 (fine) | 382 k | 378.9 K |

## Sample

[`../data_sample/`](../data_sample) contains 6 ready-to-load cases (3 heatsinks, 3 cold plates) plus a
sample split file. Load one:

```python
import numpy as np, json, os
d = np.load("data_sample/" + sorted(os.listdir("data_sample"))[0], allow_pickle=True)
print({k: getattr(d[k], "shape", None) for k in d.files})
print(json.loads(str(d["conditions"])))
```

## License

The corpus is generated with OpenFOAM (GPL solver, but the *output data* is yours) and is released
under the repository's PolyForm Noncommercial License 1.0.0 (noncommercial use; commercial use
requires a separate license). If you extend it, please keep the no-licensed-data rule
(see [`../CONTRIBUTING.md`](../CONTRIBUTING.md)).
