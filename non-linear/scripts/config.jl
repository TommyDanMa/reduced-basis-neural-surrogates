# Shared configuration and small I/O helpers for the numbered scripts.
# Every script starts with `include("config.jl")`.

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)

using RadiativeSurrogates
using JLD2
using Random, StableRNGs
using Statistics, LinearAlgebra, Printf

const PROJECT_DIR = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR    = joinpath(PROJECT_DIR, "data")
const FIG_DIR     = joinpath(PROJECT_DIR, "figures")
mkpath(DATA_DIR); mkpath(FIG_DIR)

# ----- experiment configuration -------------------------------------------------
# Physical regime (SI units): loads and emissivities chosen so steady
# radiating-edge temperatures span ≈ 250–520 K over the μ-box.
const CFG = RadiativeConfig()               # 60×30 P1, k = 5 W/mK, T_space = 3 K
const CFG_HOTWALL = RadiativeConfig(bc = :hotwall, k = 1.0)  # config A (draft setup)

const EPS_RANGE    = (0.1, 0.95)            # emissivity ε(μ)
const Q_RANGE      = (100.0, 800.0)         # chip power, W per m depth
const DUTY_RANGE   = (0.2, 0.8)             # transient load duty cycle
const PERIOD_RANGE = (1800.0, 10800.0)      # transient load period, s

const N_SAMPLES = 512                       # steady dataset size
const TEST_FRAC = 0.2

# ----- transient study ------------------------------------------------------------
const N_MU_T    = 96                        # transient parameter samples (ε,Qp,duty,period)
const NCYCLES   = 3                         # load periods per trajectory
const NPHASE    = 20                        # saved states per period (phase-uniform)
const RANKS_T   = (4, 8, 16, 32)            # space-time POD truncations
const DEFAULT_R_T = 8                       # provisional; pinned by the scripts/11 sweep
# The steady manifold with fixed load geometry is numerically rank-3 (σ₃/σ₁ ≈
# 7e-9, reconstruction floor ~1e-16 at r=3): μ=(ε,Q) moves levels/amplitudes of
# essentially fixed shapes, so the nonlinearity lives in c(μ), not the basis.
const RANKS     = (1, 2, 3, 5)              # POD truncations to compare
const DEFAULT_R = 2                         # pinned by the scripts/05 rank sweep
const SEED      = 20260622

# Map physical μ to the network's [-1,1]^d box and back.
standardize(v, lo, hi)   = 2 * (v - lo) / (hi - lo) - 1
unstandardize(s, lo, hi) = lo + (s + 1) * (hi - lo) / 2
standardize_mu(eps_r, Q) =
    (standardize(eps_r, EPS_RANGE...), standardize(Q, Q_RANGE...))
standardize_mu_t(eps_r, Qp, duty, period) =
    (standardize(eps_r, EPS_RANGE...), standardize(Qp, Q_RANGE...),
     standardize(duty, DUTY_RANGE...), standardize(period, PERIOD_RANGE...))

"""
Network input for the space-time mapper at saved state `j` (1-based): the four
standardized parameters, normalized time 2τ−1, and two phase harmonics (the
load switches at a duty-dependent phase, so the map has in-phase kinks — the
harmonics give the net a periodic basis to build them from). Phase-uniform
saving makes these features sample-independent functions of `j`.
"""
function transient_features(μst::NTuple{4,Float64}, j::Integer)
    τ = (j - 1) / (NCYCLES * NPHASE)
    ph = 2pi * ((j - 1) % NPHASE) / NPHASE
    return [μst..., 2τ - 1, sin(ph), cos(ph), sin(2ph), cos(2ph)]
end

# ----- file locations -----------------------------------------------------------
snapshots_path()  = joinpath(DATA_DIR, "snapshots.jld2")
pod_path()        = joinpath(DATA_DIR, "pod.jld2")
eval_path()       = joinpath(DATA_DIR, "eval.jld2")
model_path(name)  = joinpath(DATA_DIR, "model_$(name).jld2")
snapshots_t_path() = joinpath(DATA_DIR, "snapshots_transient.jld2")
pod_t_path()       = joinpath(DATA_DIR, "pod_transient.jld2")
eval_t_path()      = joinpath(DATA_DIR, "eval_transient.jld2")

# ----- Lux model persistence (parent conventions) --------------------------------
"""Save an MLP-family model: architecture spec + parameters/state + training log.
`cscales` are the per-mode coefficient scales (targets are trained whitened
because the σ-cliff gives the raw coefficients a ~1e9 dynamic range)."""
function save_mlp(path; kind, indim, outdim, hidden, depth, ps, st,
                  losses = Float64[], train_time = 0.0, cscales = Float64[])
    jldsave(path; kind, indim, outdim, hidden, depth, ps, st, losses,
            train_time, cscales)
end

"""Reload an MLP-family model, rebuilding its architecture and restoring weights."""
function load_mlp(path; rng = StableRNG(SEED))
    d = load(path)
    model, _, _ = make_mlp(d["indim"], d["outdim"];
                           hidden = d["hidden"], depth = d["depth"], rng = rng)
    return (; model, ps = d["ps"], st = d["st"], kind = d["kind"],
            losses = d["losses"], train_time = d["train_time"],
            indim = d["indim"], outdim = d["outdim"],
            cscales = get(d, "cscales", Float64[]))
end

# ----- dataset / basis loaders ----------------------------------------------------
"Load the raw snapshot dataset dictionary."
load_snapshots() = load(snapshots_path())

"Load the POD basis as a `PODBasis` plus the cached reconstruction curves."
function load_pod()
    d = load(pod_path())
    return (; P = PODBasis(d["mean"], d["modes"], d["svals"]),
            ranks = d["ranks"], recon_train = d["recon_train"],
            recon_test = d["recon_test"], energy = d["energy"])
end
