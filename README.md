# Reduced-Basis Neural Surrogates for Parametric Differential Equations

A Julia project (a *Treball de Recerca* in scientific machine learning) that learns
the solution operator of a parametric elliptic PDE **efficiently** by predicting a
handful of reduced-basis coefficients instead of every grid value.

> **Thesis.** Parametric PDE solutions live on a low-dimensional manifold
> (fast-decaying singular values / Kolmogorov *n*-width). So a *small* network only
> needs to learn the smooth map from parameters `μ` to the coefficients `c` of a
> POD basis, and reconstruct `u(x;μ) ≈ ū(x) + Σᵢ cᵢ(μ) φᵢ(x)`. The efficiency comes
> from the **right basis**, not from the network.

We solve `-∇·(a(x,y;μ)∇u) = f` on the unit square (`u = 0` on the boundary),
compress the solutions with POD/SVD, and compare three surrogates:

| Surrogate   | Learns   | rel L² error | learnable params | PDE residual | inference speed-up |
|-------------|----------|-------------:|-----------------:|-------------:|-------------------:|
| Direct MLP  | `μ ↦ u`  |        ~3.2% |          112,512 |        ~10.1 |               ~22× |
| **POD-MLP** | `μ ↦ c`  |       ~0.62% |            9,162 |       ~0.124 |               ~24× |
| **POD-KAN** | `μ ↦ c`  |       ~0.99% |        **2,640** |       ~0.207 |          **~455×** |

The reduced-basis models are far more accurate, far smaller and far more physically
consistent (tiny PDE residual) than the direct baseline. The POD reconstruction
floor at rank 10 is ~2e-6, i.e. **the basis is never the bottleneck — the network
is.** (Numbers from `scripts/07`; regenerate to reproduce. Inference is timed as
batched throughput over the test set vs. the solver's per-problem time, so the
speed-up reflects batch deployment, not single-query latency.)

## Quick start

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate()'   # install deps (first time)
julia --project=. test/runtests.jl                    # 33 tests: solver, POD, models

# reproducible pipeline (each step caches to data/)
julia --project=. scripts/01_validate_solver.jl       # MMS: 2nd-order convergence
julia --project=. scripts/02_generate_dataset.jl      # Sobol μ → snapshots
julia --project=. scripts/03_compute_pod.jl           # SVD / reduced basis
julia --project=. scripts/04_train_direct_mlp.jl      # baseline  μ ↦ u
julia --project=. scripts/05_train_pod_mlp.jl         # reduced   μ ↦ c
julia --project=. scripts/06_train_pod_kan.jl         # reduced   μ ↦ c  (KAN, P1)
julia --project=. scripts/07_evaluate_models.jl       # comparison table
julia --project=. scripts/08_make_figures.jl          # all report figures → figures/
julia --project=. scripts/10_residual_ablation.jl     # physics-informed loss (P1)

# interactive live-calibration console (needs an OpenGL display)
julia --project=. scripts/09_run_console.jl
```

## The interactive console

`scripts/09_run_console.jl` opens a GLMakie window: drag the `μ₁, μ₂` sliders,
pick a model, move the rank slider, and every panel updates live — the coefficient
field, the true (re-solved) and predicted solutions, the error and PDE-residual
heatmaps, the singular-value spectrum, a parameter-space minimap and timing/error
read-outs. A static preview is `figures/11_dashboard.png`; the in-app **save
screenshot** button writes `figures/console_screenshot.png`.

## Repository layout

```
src/        ReducedBasisSurrogates.jl (core module) + pde_solver, coefficients,
            pod, models, metrics, plotting, console; models_kan.jl is the P1 add-on
scripts/    01–10 numbered, deterministic pipeline steps (+ config.jl)
test/       runtests.jl + five focused test files
report/     report.md (11-section TR) + refs.bib
figures/    exported PNG figures
data/        cached snapshots/POD/models (git-ignored, regenerable)
```

## Method (one paragraph)

A conservative second-order finite-volume discretisation gives a sparse SPD system
`A(μ)u = b` (validated by manufactured solutions, observed order ≈ 2.00). Snapshots
for Sobol-sampled `μ` are stacked and the SVD yields POD modes `Φ` and mean `ū`; by
Eckart–Young the rank-`r` truncation is optimal and the tail energy bounds the
Kolmogorov *n*-width. Because every snapshot vanishes on the boundary, so do the
modes — reconstructions satisfy the Dirichlet condition **exactly**. A small MLP or
KAN learns `μ ↦ c`; the relative PDE residual is the physics-consistency metric, and
an optional residual-regularised loss is available as an ablation.

Equivalently, the solution operator `G : μ ↦ u` is factored as **`G = R ∘ N`**, where
`N : μ ↦ c` is the small learned network and `R : c ↦ ū + Φᵣc` is the *fixed* POD
reconstruction. Unlike DeepONet / FNO (which learn the output representation jointly
with the map), the basis here is fixed a priori from data — maximal structure for
minimal learning. See `scripts/11`–`12` for the rank sweep and the
POD-vs-sine-vs-random basis stress-test.

## Future work

Structure-preserving bases via Finite Element Exterior Calculus (Arnold–Falk–
Winther) — POD is data-optimal but not structure-preserving — pointing toward
structure-preserving neural operators. See `report/report.md` §11.

## Requirements

Julia ≥ 1.10 (developed on 1.12). Key packages: Lux, KolmogorovArnold, Zygote,
Optimisers, CairoMakie/GLMakie, JLD2, QuasiMonteCarlo (see `Project.toml`).
