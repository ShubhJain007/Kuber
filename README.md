<div align="center">

# Kuber

**An open framework for conjugate–heat-transfer AI — build, train, and operate neural surrogates for any coupled fluid–heat problem.**

The full stack, from physics-data generation to a deployable, geometry-general surrogate — for **any coupled fluid–heat (conjugate heat transfer) problem**: Datacenters, CPUs, GPUs, heatsinks, cold plates, heat exchangers, power electronics, battery packs, HVAC, turbomachinery cooling. First domain with published results: **electronics cooling**, where Kuber beats the previous best on the SIMSHIFT heatsink benchmark with no domain adaptation.

[![Interactive demo](https://img.shields.io/badge/%F0%9F%A7%8A%20interactive%20demo-live-1F4E79.svg)](https://shubhjain007.github.io/Kuber/demo.html)
[![Project page](https://img.shields.io/badge/%F0%9F%8C%A1%EF%B8%8F%20project%20page-live-2E7D5B.svg)](https://shubhjain007.github.io/Kuber/)
[![Paper](https://img.shields.io/badge/paper-PDF-B31B1B.svg)](paper/kuber.pdf)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShubhJain007/Kuber/blob/main/notebooks/quickstart.ipynb)
[![tests](https://github.com/ShubhJain007/Kuber/actions/workflows/tests.yml/badge.svg)](https://github.com/ShubhJain007/Kuber/actions/workflows/tests.yml)
[![License: PolyForm NC](https://img.shields.io/badge/license-PolyForm%20Noncommercial%201.0.0-2E7D5B.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-1F4E79.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-1F4E79.svg)](https://pytorch.org/)
[![Built on PhysicsNeMo](https://img.shields.io/badge/built%20on-NVIDIA%20PhysicsNeMo-1F4E79.svg)](https://github.com/NVIDIA/physicsnemo)

</div>

**Kuber prediction vs. CFD ground truth** — the surface temperature field, surrogate beside CFD:

| | |
|:---:|:---:|
| ![Heatsink simulation — Kuber vs CFD](assets/sim/heat-sink-comparison.png) | ![Cold plate simulation — Kuber vs CFD](assets/sim/cold-plate-comparison.png) |
| **Heatsink simulation** — ±2.11 K temperature agreement, **7,000× faster** than CFD | **Cold plate simulation** — ±1.33 K temperature agreement, **445× faster** than CFD |

> **Read the [technical report (PDF)](paper/kuber.pdf).** The [project page](https://shubhjain007.github.io/Kuber/) and an [interactive ground-truth-vs-prediction viewer](https://shubhjain007.github.io/Kuber/demo.html) (3D solid geometry + fluid field, heatsink & cold plate) are served from **GitHub Pages**.

## Key Features

- **Geometry-general.** SurfaceGeoTransolver ingests raw boundary geometry — a surface point cloud with normals — and predicts the full field `(Uₓ, U_y, U_z, T, p_rgh)` at any query point. No analytic signed-distance field, so it works on arbitrary CAD.
- **State of the art, no domain adaptation.** 12.14 K temperature RMSE on the public SIMSHIFT heatsink out-of-distribution split — beating the previous published best (UPT, 12.41 K) — while every baseline relies on unsupervised domain adaptation and Kuber uses none.
- **One model, many device classes.** Heatsinks (wall-temperature BC, buoyancy-driven, air) and cold plates (heat-flux BC, forced liquid) from a single set of weights, distinguished only by a device flag and physics conditioning.
- **Reproducible data engine.** Parametric geometry → OpenFOAM `buoyantSimpleFoam` → per-node `.npz`, resumable, convergence-gated, mesh-convergence-verified, and license-clean (0 cases from any external source).
- **Honest evaluation harness.** Per-field normalized RMSE, near-wall fidelity, a numerical no-explosion stability proof, and a value-of-data ablation — with machine-readable results.
- **Up to 10,000× faster than CFD.** Sub-second, geometry-independent inference versus minutes-to-hours per CFD solve.

*Kuber is an open **suite**, not a finished benchmark — see the [Roadmap](#roadmap) for what's next.*

![SurfaceGeoTransolver architecture — real geometry input, network pipeline, predicted field](assets/sim/architecture.png)

## Table of contents

- [Key Features](#key-features)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Recipes](#recipes)
- [Dataset & Training](#dataset--training)
- [Performance Benchmarks](#performance-benchmarks)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Supported systems](#supported-systems)
- [Licensing](#licensing)
- [Acknowledgements](#acknowledgements)
- [Citing](#citing)

## Installation

```bash
git clone https://github.com/ShubhJain007/Kuber.git
cd Kuber
python -m venv .venv && source .venv/bin/activate     # or use conda
pip install -r requirements.txt
```

The model core is **GeoTransolver** from NVIDIA PhysicsNeMo (`physicsnemo.experimental.models.geotransolver`) — install per its [instructions](https://github.com/NVIDIA/physicsnemo). **Data generation** additionally needs [OpenFOAM](https://www.openfoam.com/) (v2306+) on your `PATH`; the model, evaluation, and the data sample work without it. See [Supported systems](#supported-systems) for tested versions.

## Quickstart

**Try it in your browser, no install:** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ShubhJain007/Kuber/blob/main/notebooks/quickstart.ipynb) — loads a real CHT case and visualizes it on CPU.

```bash
# 1. Peek at a sample case (no OpenFOAM, no GPU needed)
python -c "import numpy as np, os; d=np.load('data_sample/'+sorted(os.listdir('data_sample'))[0]); \
           print({k:getattr(d[k],'shape',None) for k in d.files})"

# 2. Evaluate a trained checkpoint (geometry mode + conditioning are read from it)
python -m kuber.train_simshift \
    --data <simshift_npz_dir> --splits <splits.json> --difficulty medium \
    --eval_only <model.pt>

# 3. Train the production model (full-geometry / surface input)
python -m kuber.train_simshift \
    --data <npz_dir> --splits <splits.json> --difficulty medium --geom_mode surface
```

## Recipes

Common workflows — each is a single command (see `python -m kuber.train_simshift --help` for all flags).

| Recipe | Command |
|---|---|
| **Train** the production model | `python -m kuber.train_simshift --data <simshift> --splits <splits> --difficulty medium --geom_mode surface` |
| **Evaluate** a checkpoint (in-dist + OOD) | `python -m kuber.train_simshift --eval_only <model.pt> --data <dir> --splits <splits> --difficulty medium` |
| **Value-of-data** pretraining, then fine-tune | `... --geom_mode surface --drop_geom_scalars --init_from pretrain/<ckpt>` |
| **Stability proof** (no-explosion) | `python -m kuber.edge_proof --ckpt <model.pt> --data <dir> --splits <splits> --difficulty medium` |
| **Generate** CHT data (needs OpenFOAM) | `python datagen/run_sweep_bsf.py --cases <cases> --out <corpus> --scripts <bsf_scripts>` |
| **Regenerate** the result figures | `python assets/make_figures.py` |

`--geom_mode {none,sdf,dsdf,surface}` selects the geometry representation (`surface` is the reported production model); `--refine` adds the generative PDE-Refiner head. Architecture card: [`docs/MODEL.md`](docs/MODEL.md).

## Dataset & Training

**The corpus.** A self-generated OpenFOAM conjugate-heat-transfer dataset for electronics cooling —
**0 cases from SIMSHIFT or any licensed/scraped source**. Each case is one solved steady-state field
sampled to a 16,384-node fluid point cloud plus a 2,048-point solid-surface cloud, spanning two device
classes (buoyancy-driven **heatsinks** and forced-convection **cold plates**), four fluids (air, water,
oil, glycol), and both natural and forced convection.

*How it's built:* a parametric (Latin-hypercube) generator → STL → `snappyHexMesh` / `blockMesh` (+ prism
layers) → OpenFOAM **`buoyantSimpleFoam`** (compressible, `kOmegaSST`; the solid modelled as a
heated-wall or heat-flux boundary) → `.npz`. The pipeline is resumable and gated on convergence plus a
physics filter. One non-obvious detail makes the whole thing stable: because the solver is compressible,
pressure must be **absolute** (`p_rgh ≈ 1e5` Pa), not gauge-zero. Coverage, the mesh-convergence study,
and the full format: **[`docs/DATASET.md`](docs/DATASET.md)**.

**Training.** One **SurfaceGeoTransolver** (~14.3 M params) is trained across both device classes with
geometry read *only* from the surface point cloud (a learned local descriptor, not a scalar SDF) under a
single unified physics-conditioning vector. Targets are z-scored on the training split only (no leakage);
optimisation is Adam (`1e-3`) with a **Warmup–Stable–Decay** schedule and best-validation checkpointing;
**no unsupervised domain adaptation**. Preprocessing, conditioning, the schedule, and reproduction:
**[`docs/TRAINING.md`](docs/TRAINING.md)**.

## Performance Benchmarks

All numbers are for **SurfaceGeoTransolver** (full-geometry input), measured and reproducible with the code here; caveats are in [`docs/RESULTS.md`](docs/RESULTS.md). Machine-readable: [`results/simshift_medium.json`](results/simshift_medium.json), [`results/leaderboard.csv`](results/leaderboard.csv), [`results/multigeo.json`](results/multigeo.json).

> **UDA — Unsupervised Domain Adaptation:** training-time techniques that adapt a model to the *unlabeled* target (test) distribution — e.g. aligning source- and target-domain feature statistics — to shrink the out-of-distribution gap. The SIMSHIFT baselines rely on it; **Kuber uses none** and still leads.

**SIMSHIFT heatsink — medium / out-of-distribution split** (train fin counts 5–8 → test 10–12). Baselines include UDA; Kuber uses none. Lower is better.

![SIMSHIFT heatsink leaderboard — temperature RMSE](assets/fig_leaderboard.svg)

| model | UDA | Temp RMSE (K) ↓ | Velocity RMSE (m/s) ↓ | params |
|---|:---:|:---:|:---:|:---:|
| **Kuber — SurfaceGeoTransolver** | ✗ | **12.14** | 0.044 | 14.3 M |
| UPT *(prev. published best)* | ✓ | 12.41 | 0.039 | ~14 M¹ |
| Transolver | ✓ | 13.43 | 0.041 | ~14 M¹ |
| PointNet | ✓ | 17.43 | 0.044 | ~14 M¹ |

¹ The SIMSHIFT paper prints no parameter counts; its configs show all three baselines are comparably sized (~10–15 M). The edge is geometry conditioning + training recipe at the same budget, without the UDA the baselines use. Full per-field metrics: [`docs/RESULTS.md`](docs/RESULTS.md). **Want on the board?** → [How to submit](docs/BENCHMARK.md#submitting-a-result).

**Generalization** — the leading number is zero-shot (test fin counts never appear in training); **numerical stability** — predicted ∇T stays at or below physical everywhere (explosion fraction 0, zero NaN/Inf).

| | |
|---|---|
| ![In-distribution vs out-of-distribution](assets/fig_indist_vs_ood.svg) | ![Stability — edge gradient fidelity](assets/fig_stability.svg) |

**Value of the data engine** — pretraining on the self-generated corpus lowers error further; **speed** — up to 10,000× faster than the CFD it learns from.

| | |
|---|---|
| ![Value of the corpus](assets/fig_value_of_data.svg) | ![Speed — surrogate vs CFD](assets/fig_speed.svg) |

> **Pretraining a small dataset — and why it isn't leakage.** With only a few hundred fine-tuning cases, initialization matters: pretraining on the broader OpenFOAM corpus gives the model a physics prior (temperature–flow coupling, near-wall behavior, boundary-layer structure) that is hard to learn from the small target set alone — which is why the gain is largest at the easy shift and in-distribution, and tapers at the hardest shift the corpus doesn't cover. **No held-out evaluation data is ever seen during pretraining or fine-tuning:** the pretraining corpus is our own OpenFOAM data, disjoint from SIMSHIFT and its splits; the SIMSHIFT target (OOD) split is used only for evaluation; and the multi-geometry held-out per-device cases are excluded from training. The gains reflect transfer of physics, not exposure to the test distribution.

**One model, two device classes.** Held-out per class on our corpus (there is no public cold-plate CHT benchmark; not comparable to the SIMSHIFT numbers above):

| held-out class | Temp RMSE (K) ↓ | mean nRMSE ↓ |
|---|:---:|:---:|
| cold plates | 3.11 | 0.028 |
| heatsinks | 5.13 | 0.145 |
| in-distribution (both) | 1.72 | 0.027 |

![Multi-geometry — one model, heatsinks + cold plates](assets/fig_multigeo.svg)

**Ground truth vs. prediction** — see the heatsink and cold-plate comparisons at the top (Kuber vs CFD, ±2.11 K and ±1.33 K), or [**open the interactive 3D viewer →**](https://shubhjain007.github.io/Kuber/demo.html).

**Dataset.** A self-generated OpenFOAM CHT corpus — 0 cases from SIMSHIFT or any licensed source. A 6-case sample is in [`data_sample/`](data_sample); the contract is in [`docs/DATASET.md`](docs/DATASET.md). Fidelity is verified: prism layers recover the near-wall hot spot to within 0.1 K of a fine mesh at ~2.7× lower cost.

![The Kuber corpus at a glance](assets/fig_corpus.svg)

## Roadmap

Kuber is an open **suite**, not a finished benchmark. Three milestones, each gated on a verifiable outcome rather than an activity.

### 🟢 Milestone 1 — Production-ready on real customer geometry

*Done when: the model clears a design partner's written acceptance criteria on their own parts.*

- [ ] **Generalize beyond parametric families to as-supplied CAD.** The surface encoder already accepts arbitrary meshes; this is a data and validation problem, not an architecture one.
- [ ] **Grow the corpus past its air-dominated composition** — substantially more liquid-cooled cases, working fluids, and regimes.
- [ ] **Establish scaling laws for physical AI.** Extend the [value-of-data](docs/RESULTS.md) result from a two-point ablation into measured curves: how out-of-distribution error scales with corpus size, model parameters, and — the question specific to physics — *geometry and physics diversity* versus raw case count. Published openly, with the data engine as the instrument that makes it measurable.
- [ ] **A graded distribution-shift protocol.** Replace single-axis holdouts with a shift taxonomy evaluated independently along each axis, reported as degradation curves rather than pass/fail:

  | Axis | Weak shift | Strong shift |
  |---|---|---|
  | Geometry | parametric extrapolation (current SIMSHIFT split) | topology change; then human-authored CAD |
  | Regime | Ra/Re beyond the training envelope | natural → forced convection |
  | Fluid | unseen Prandtl number | unseen material class |
  | Boundary condition | wall-temperature → heat-flux | mixed / conjugate interfaces |
  | Scale | characteristic length outside range | order-of-magnitude change |

  Three requirements make this rigorous rather than decorative. **(i) Shift magnitude is measured, not asserted** — report a distributional distance (MMD or Wasserstein in a geometry/physics feature space) between train and test for every split, so "out-of-distribution" is a quantity with a number attached. **(ii) A leakage audit** — nearest-neighbour search over the corpus to confirm no near-duplicates straddle any split. **(iii) Provenance shift as the ceiling test** — evaluate against cases generated by a *different* solver and mesher, and ultimately against experimental thermocouple or IR measurements. Only a provenance-shifted test can separate learned physics from learned numerics; every split drawn from one generator is confounded with that generator's discretization.

- [ ] **Release trained checkpoints** as tagged GitHub Releases, and replace estimated inference latency with measured per-GPU timings.

### 🔵 Milestone 2 — Trustworthy enough to design against

*Done when: an engineer can act on a prediction without re-running CFD to check it.*

- [ ] **Calibrated per-node uncertainty.** Turn the PDE-Refiner denoising ensemble into real error bars, so the surrogate reports where it should be trusted and where to fall back to CFD. This is the gate on adoption — accuracy alone is not sufficient.
- [ ] **Cold-plate topologies beyond straight-channel** — serpentine, pin-fin, parallel micro-channel.
- [ ] **Additional device classes** — heat exchangers, power electronics, battery packs — prioritized by what design partners actually bring us.
- [ ] **Architecture** — hierarchical multi-resolution surface tokens; per-block geometry cross-attention.
- [ ] **Active learning driven by the scaling curves.** Target the data engine at the regions where error falls fastest per case generated.

### 🟣 Milestone 3 — From predictor to design tool

*Done when: a user goes from CAD file to an optimized geometry without leaving the loop.*

- [ ] **Connectors.** STEP / IGES / STL ingest and native OpenFOAM case import, plus export back into standard thermal workflows, exposed as a Python API and CLI.
- [ ] **Agentic geometry optimization.** Propose → predict → score → refine against thermal and pressure-drop objectives, with CFD in the loop only to verify the winner.
- [ ] **Deployment.** Hosted inference API, ONNX / TensorRT export, and a public versioned leaderboard with a sealed test set.

Contributions to any of these are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributing

Contributions that add rigor — new baselines, harder splits, better data — or catch over-claims are all welcome. Submit a leaderboard result, add a baseline, extend the dataset, or report a bug. Ground rules: no licensed/scraped data, full reproducibility (exact command + environment), and honest disclosure of UDA and external pretraining. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Supported systems

- **OS:** Linux (tested). **Python:** 3.10+.
- **Tested versions** (pinned in [`requirements.txt`](requirements.txt)): `torch==2.12.1`, `numpy==2.4.6`, `nvidia-physicsnemo==2.1.1`.
- **Hardware:** a CUDA GPU is recommended for training and required for the multiscale ball-query core; evaluation and the data sample run on CPU (slower) for small cases.
- **Data generation** (`datagen/`, optional): OpenFOAM v2306+ with `buoyantSimpleFoam`, `blockMesh`, `snappyHexMesh` on `PATH`.

## Licensing

Kuber is released under the **[PolyForm Noncommercial License 1.0.0](LICENSE)** — free for research, education, evaluation, and any other noncommercial use. **Commercial use requires a separate license** — contact the Kuber.ai team. This mirrors how open Engineering-AI frameworks are typically licensed: open for the community, with a separate commercial track. The GeoTransolver core is used as a dependency from NVIDIA PhysicsNeMo (Apache-2.0).

## Acknowledgements

Kuber builds on **GeoTransolver / PhysicsNeMo** (NVIDIA), and draws on **Transolver** (Wu et al., 2024), **AB-UPT** (Alkin et al., 2025), **SIMSHIFT** (Setinek et al.), **PDE-Refiner** (Lippe et al., 2023), and **OpenFOAM** (Weller et al., 1998). Full references are in [`docs/MODEL.md`](docs/MODEL.md) and the [paper](paper/kuber.pdf).

## Citing

A full technical report is in [`paper/kuber.pdf`](paper/kuber.pdf) (IEEE conference format; LaTeX source in [`paper/`](paper)).

```bibtex
@techreport{kuber2026,
  title       = {Kuber: Geometry-General Neural Surrogates for Conjugate Heat Transfer},
  author      = {Jain, Shubh},
  institution = {Kuber.ai},
  year        = {2026},
  url         = {https://github.com/ShubhJain007/Kuber}
}
```

<div align="center">
<sub>Built by the Kuber.ai team — an open framework for conjugate–heat-transfer AI.</sub>
</div>
