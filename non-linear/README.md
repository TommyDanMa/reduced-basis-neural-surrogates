# RadiativeSurrogates: reduced-basis neural surrogates with a nonlinear radiation BC

Sub-project of the reduced-basis repo: the heat equation on a 2D radiator section whose
dominant heat rejection is **Stefan–Boltzmann radiation to deep space** (T_space = 3 K),
with parametric emissivity ε and electronics load cycles, a controlled prototype for
AI1-class orbital-compute thermal design (`docs/images/`). **Gridap assembles, SciML
solves**: Gridap.jl builds the FE residual/Jacobian, NonlinearSolve.jl drives the steady
Newton solves, OrdinaryDiffEq.jl (mass-matrix ImplicitEuler/Trapezoid/FBDF) drives the
transients.

**TL;DR**
- **Question:** does "learn the POD coefficients, not the field" survive a strongly
  nonlinear boundary condition, and what does the radiation regime do to the rank?
- **Core result:** for fixed load geometry the *steady* manifold is numerically **rank 3**
  (σ₃/σ₁ ≈ 7e-9); all nonlinearity lives in the coefficient map c(μ), and a whitened
  8.6k-parameter MLP hits 9.2e-4 relative L² (PDE residual 6.5e-3 vs **38.7** for a direct
  field MLP) at ~4,500× single-query / ~10,000× batched throughput over the 49 ms Newton
  FOM. Load-cycle *transients* re-thicken the manifold (σ₈/σ₁ ≈ 3e-5 vs the steady cliff),
  and the (μ, t) ↦ c surrogate, an honest 1.5–1.7% on raw trajectories, reproduces
  full-order **design rankings** on a 10×10 (ε, duty) grid: Spearman 0.999, top-5 overlap
  5/5, identical optimum, QoI error median 2.4 K on a 382–785 K range, 373× end-to-end.
- **Main caveat:** 2D, fixed load geometry, unity view factor, in-distribution queries:
  a controlled prototype, not a general theorem. The rank-3 collapse is a property of the
  fixed geometry, not of radiation physics.

## Quickstart

```bash
julia --project=non-linear -e 'using Pkg; Pkg.instantiate()'
julia --project=non-linear non-linear/test/runtests.jl      # 49 assertions
# then, in order:
julia --project=non-linear non-linear/scripts/01_validate_steady.jl
julia --project=non-linear non-linear/scripts/02_generate_steady_dataset.jl
# ... through 13; and the live console:
julia --project=non-linear non-linear/scripts/14_run_console.jl
```

The console solves the steady FOM live while you drag ε and Q (a 50 ms Newton solve is
interactive), next to the surrogate, the error field, QoI readouts and both σ-spectra; the
transient panel updates the surrogate's peak-T trace instantly, with the true FOM one
button-press away:

![Design console](figures/15_console.png)

## Layout

- `src/`: `physics.jl` (weak forms, u|u|³ radiation term), `fom_steady.jl`
  (Gridap→NonlinearSolve glue, warm starts, residual traces), `fom_transient.jl`
  (mass-matrix ODE form, square-wave loads via `tstops`), `postprocess.jl` (radiated power,
  peak T, energy balance), plus `pod.jl` / `models.jl` / `metrics.jl` ported from the parent
  project (`models_kan.jl` stays out of the core module).
- `scripts/01–14`: numbered, deterministic (StableRNGs + Sobol): validate steady, steady
  dataset/POD/mappers/eval/benchmark, validate transient, transient dataset/space-time
  POD/mapper/eval, Nr study, design study, console.
- `test/`: seven named test files: MMS rates 1.9996/1.9999/2.0000 through the nonlinear BC,
  AD = FD = hand-written Jacobian, exact discrete energy balance, θ-family dt-orders
  0.95–0.98 / 2.08–2.00, nodally exact 1D radiative equilibrium.
- `report/report.md`: the write-up, figures included; `figures/`: all generated figures.

## Headline (steady, held-out; deployed rank r = 2, pinned by the rank sweep)

| model | rel L² | PDE residual ‖R(û)‖/‖F‖ | params | single× | batched× |
| --- | ---: | ---: | ---: | ---: | ---: |
| direct MLP | 5.8e-3 | 3.9e+1 | 131,427 | 576× | 6,711× |
| **POD-MLP** | **9.2e-4** | **6.5e-3** | 8,642 | 4,468× | 9,923× |
| POD-KAN | 2.9e-3 | 2.1e-2 | **880** | 4,541× | 10,899× |

POD floor at r=2: 1.1e-9 (basis ≠ bottleneck). Speed-ups: batched throughput over FOM Newton
solves, incl. reconstruction, excl. plotting/model loading, after warm-up (BenchmarkTools
medians, batch = 102). Coefficients are trained **whitened**; the rank cliff gives c a ~1e9
dynamic range.

Key figures: `figures/11_spacetime_decay.png` (the steady rank-3 cliff vs the rich transient
spectrum, the project's keystone), `figures/12_transient_eval.png` (P_rad tracking through
load switches, surrogate vs FOM), `figures/14_design_study.png` (the (ε, duty) design-ranking
grid), `figures/10_load_cycle_response.png` (the radiator's thermal inertia filtering a
0↔600 W square wave to a ~20 W ripple).

Relationship to the parent project: this executes its §11 "Gridap.jl FEM" future-work item,
keeps its conventions (numbered scripts, MMS acceptance band, honest benchmark captions), and
drops one of its claims honestly: with a radiation BC there is no "exact BC by construction",
so QoI fidelity (radiated power, peak temperature) takes that role.
