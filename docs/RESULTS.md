# Results — Kuber SurfaceGeoTransolver

All results are for **SurfaceGeoTransolver**, Kuber's production model (full-geometry input: a surface
point cloud + normals + physics conditioning; ~14.3 M parameters). Alternative geometry encodings
(scalar SDF, directional SDF, conditions-only) are available in the code as `--geom_mode` options but
are not the reported model. Every number here is measured; machine-readable copies are in
[`../results/simshift_medium.json`](../results/simshift_medium.json) and
[`../results/leaderboard.csv`](../results/leaderboard.csv).

Predicted field: `(U_x, U_y, U_z, T, p_rgh)` per query node. Metric definitions: [`BENCHMARK.md`](BENCHMARK.md).

---

## 1. SIMSHIFT heatsink benchmark (public)

Official split, **medium** difficulty: train on fin counts 5–8, test on the shifted target domain
(fin counts 10–12). Baseline numbers are quoted from the SIMSHIFT paper (Table 2); **those numbers
include unsupervised domain adaptation (UDA)** — their best case. SurfaceGeoTransolver uses **no UDA**
and is trained from scratch on SIMSHIFT's 222 training cases.

![SIMSHIFT heatsink leaderboard — temperature RMSE](../assets/fig_leaderboard.svg)

| model | UDA | Temp RMSE (K) ↓ | Velocity RMSE (m/s) ↓ | params |
|---|:---:|:---:|:---:|:---:|
| **Kuber — SurfaceGeoTransolver** | ✗ | **12.14** | 0.044 | 14.3 M |
| UPT (prev. published best) | ✓ | 12.41 | 0.039 | ~14 M¹ |
| Transolver | ✓ | 13.43 | 0.041 | ~14 M¹ |
| PointNet | ✓ | 17.43 | 0.044 | ~14 M¹ |

¹ The paper reports no parameter counts; its released configs show all three baselines are
comparably sized (~10–15 M). The advantage is the geometry conditioning and training recipe at the
same parameter budget, without the UDA the baselines use.

On temperature — the engineering-critical field — SurfaceGeoTransolver at 12.14 K is below the
previous published best (UPT, 12.41 K), which itself includes UDA (its no-UDA result is ~13.03 K).
Velocity (0.044 m/s) is competitive but not the lowest; the model trades a little velocity accuracy
for its temperature lead. The SIMSHIFT paper publishes only temperature and velocity RMSE on the
target domain, so those are the only two head-to-head columns possible.

### 1.1 Source (in-distribution) vs. target (OOD) detail

The paper reports no source-domain per-field numbers, so this view is ours. **Mean nRMSE** (averaged
over all five fields) is the paper's primary model-selection metric.

| domain | T-RMSE (K) | Velocity RMSE (m/s) | p_rgh RMSE (Pa) | T-nRMSE | mean nRMSE |
|---|:---:|:---:|:---:|:---:|:---:|
| source (in-distribution) | 4.29 | 0.025 | 203 | 0.179 | 0.287 |
| target (out-of-distribution) | 12.14 | 0.044 | 2303 | 0.506 | 0.671 |

The leaderboard number **is** the out-of-distribution number: the target fin counts never appear in
training, so 12.14 K is a zero-shot result. In-distribution temperature error is 4.29 K.

![In-distribution vs out-of-distribution temperature RMSE](../assets/fig_indist_vs_ood.svg)

---

## 2. Multi-geometry — one model, heatsinks and cold plates

The same architecture handles two device classes from a single set of weights: **heatsinks**
(wall-temperature BC, buoyancy-driven, air) and **cold plates** (heat-flux BC, forced liquid),
distinguished only by a device flag plus fluid/BC conditioning. Evaluated per class on held-out cases
from Kuber's corpus — **there is no public cold-plate CHT benchmark**. These numbers are on our own
corpus and are **not comparable** to the SIMSHIFT numbers above (different training set and test
distribution). Machine-readable: [`../results/multigeo.json`](../results/multigeo.json).

![Multi-geometry — one model, heatsinks + cold plates](../assets/fig_multigeo.svg)

| held-out class | T-RMSE (K) | T-nRMSE | velocity nRMSE | mean nRMSE |
|---|:---:|:---:|:---:|:---:|
| cold plates | 3.11 | 0.039 | 0.028 | 0.028 |
| heatsinks | 5.13 | 0.065 | 0.216 | 0.145 |
| in-distribution (both classes) | 1.72 | 0.022 | 0.034 | 0.027 |

One model spans both physics regimes — a wall-temperature, buoyancy-driven heatsink in air and a
heat-flux, forced-liquid cold plate. The cold-plate mean nRMSE (0.028) is close to the
in-distribution number: it generalizes to held-out cold plates with almost no degradation. Cold
plates are currently straight-channel; topology diversity (serpentine, pin-fin, parallel) is on the
roadmap.

---

## 3. Value of the self-generated corpus

Does pretraining on Kuber's OpenFOAM corpus add value on top of SIMSHIFT's own training set? A clean
A/B with the **same** model (surface, reduced "transfer" conditioning — `solidTemp` only so weights
transfer cleanly — no UDA): once from scratch, once pretrained on the corpus then fine-tuned on
SIMSHIFT. Only the pretraining differs. Temperature RMSE (K), target domain unless noted.

![Value of the corpus — from scratch vs pretrained](../assets/fig_value_of_data.svg)

| distribution shift | from scratch | pretrained on our corpus | Δ |
|---|:---:|:---:|:---:|
| easy | 8.99 | **7.28** | **−1.71 K (−19 %)** |
| medium | 12.94 | **12.38** | −0.56 K (−4.3 %) |
| hard | 14.42 | 14.43 | ±0.0 K |
| in-distribution (src.test) | 4.63 | **4.09** | −0.54 K (−12 %) |

Pretraining helps materially at easy shift (−19 %) and in-distribution (−12 %), modestly at medium,
and is neutral at the hardest shift (the corpus does not yet cover that regime). Mean nRMSE improves
at every level (easy 0.426→0.379, medium 0.615→0.580, hard 0.716→0.709). This is direct evidence the
self-generated data adds value: identical architecture and fine-tuning, the only added ingredient is
the corpus.

**Why it helps — and why it isn't leakage.** With only a few hundred fine-tuning cases, initialization
matters: pretraining on the broader OpenFOAM corpus supplies a physics prior (temperature–flow
coupling, near-wall behavior, boundary-layer structure) that is hard to learn from the small target set
alone — hence the largest gains at the easy shift and in-distribution, tapering at the hardest shift the
corpus doesn't cover. Critically, **no held-out evaluation data is ever seen during pretraining or
fine-tuning**: the pretraining corpus is our own OpenFOAM data, entirely disjoint from the SIMSHIFT
benchmark and its splits; the SIMSHIFT target (OOD) split is used only for evaluation; and in the
multi-geometry setting the held-out per-device cases are excluded from training. The gains reflect
transfer of physics, not exposure to the test distribution.

---

## 4. Numerical stability — no gradient explosion

A deployable surrogate must reproduce steep near-wall gradients without over-smoothing them or
emitting unphysical spikes. Measured predicted-vs-CFD local temperature-gradient magnitude on the
edge/near-wall band (closest 15 % of nodes — tips, corners, walls) and at the steepest peaks.

![Stability — edge temperature-gradient fidelity](../assets/fig_stability.svg)

| metric (temperature ∇ at edges) | in-distribution | out-of-distribution |
|---|:---:|:---:|
| edge-band ∇T ratio (pred / CFD) — 1.0 = faithful | 0.969 | 0.746 |
| steepest-peak (p99.9) ∇T ratio | 0.938 | 0.725 |
| max ∇T ratio (distance-floored) | 0.923 | 0.827 |
| **explosion fraction** (nodes with ∇T > 2× CFD max) | **0** | **0** |
| value overshoot (T outside CFD range ±10 %) | 0 | 0.7 % |
| **NaN / Inf in the predicted field** | **0** | **0** |

Every gradient ratio is ≤ 1.0, in- and out-of-distribution — the surrogate never produces a
steeper-than-physical gradient, so explosion is ruled out by measurement, and there are zero NaN/Inf.
It is not over-smoothing either: the in-distribution edge ratio (0.97 ≈ 1.0) reproduces the true edge
structure; on OOD it errs slightly conservative (0.75, marginally smoother — the safe direction).

Reproduce: `python -m kuber.edge_proof --ckpt <model.pt> --data <simshift> --splits <splits> --difficulty medium`

---

## 5. Speed

![Speed — surrogate vs CFD, log scale](../assets/fig_speed.svg)

| solver | time / case | source |
|---|:---:|---|
| OpenFOAM CFD — low fin count | ~2.7 min | measured |
| OpenFOAM CFD — median (601 cases) | ~22 min | measured (range 161–7051 s) |
| OpenFOAM CFD — high fin count | ~117 min | measured |
| **SurfaceGeoTransolver (inference)** | **~0.3 s** | estimate\* |

Up to **10,000× faster than CFD** (the slowest high-fin-count cases at ≈0.3 s vs ≈117 min), and
inference is geometry-independent — CFD cost grows with mesh size; a forward pass does not.
\*Inference latency is an estimate pending exact per-GPU timing.

---

## 6. Data fidelity — mesh convergence

Same geometry, three meshes. Prism layers recover the near-wall hot spot to within 0.1 K of a fine
mesh at ~2.7× fewer cells, which is why the production corpus uses snap-level-2 + 3 layers.

![Mesh convergence of the hot spot](../assets/fig_mesh_convergence.svg)

| mesh | cells | T-max (hot spot) |
|---|:---:|:---:|
| snap2 (no layers) | 124 k | 359.5 K |
| **snap2 + 3 prism layers** | 142 k | **378.8 K** |
| snap3 (fine) | 382 k | 378.9 K |

Dataset coverage and the full data contract: [`DATASET.md`](DATASET.md).

---

## 7. Caveats

- **Fluid imbalance.** The corpus is air-dominated (oil/glycol are thin, ~30–40 cases each), so the
  model is strongest on air.
- **OOD axis is fin count.** Fluids and shapes appear in both train and test. Stronger studies
  (leave-one-fluid-out, leave-one-shape-out) are on the roadmap.
- **Hardest shift.** Pretraining on the corpus gives no gain at the widest fin-count gap — more/broader
  data is the lever.
- **Inference latency** is a sub-second estimate, not yet an exact per-GPU measurement.

Reproduce everything: `datagen/` generates the data; `python -m kuber.train_simshift --geom_mode surface`
trains the model; `python -m kuber.edge_proof` runs the stability proof.
