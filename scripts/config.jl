# Shared configuration and small I/O helpers for the numbered scripts.
# Every script starts with `include("config.jl")`.

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)

using ReducedBasisSurrogates
using JLD2
using Random, StableRNGs
using Statistics, LinearAlgebra, Printf

const PROJECT_DIR = normpath(joinpath(@__DIR__, ".."))
const DATA_DIR    = joinpath(PROJECT_DIR, "data")
const FIG_DIR     = joinpath(PROJECT_DIR, "figures")
mkpath(DATA_DIR); mkpath(FIG_DIR)

# ----- experiment configuration -------------------------------------------------
const GRID_N    = 40            # interior nodes per dimension  (ndof = GRID_N²)
const N_SAMPLES = 500           # number of parameter samples
const TEST_FRAC = 0.2           # held-out fraction
const PARAM_LO  = -1.0          # μ lower bound (keeps a(x,y;μ) ≥ 0.5 > 0)
const PARAM_HI  =  1.0          # μ upper bound
const RANKS     = (5, 10, 20)   # POD truncations to compare
const DEFAULT_R = 5             # rank of the deployed coefficient models (selected by §8.1 sweep)
const SEED      = 20240601

# ----- file locations -----------------------------------------------------------
snapshots_path()  = joinpath(DATA_DIR, "snapshots.jld2")
pod_path()        = joinpath(DATA_DIR, "pod.jld2")
eval_path()       = joinpath(DATA_DIR, "eval.jld2")
model_path(name)  = joinpath(DATA_DIR, "model_$(name).jld2")

# ----- Lux model persistence ----------------------------------------------------
"""Save an MLP-family model: architecture spec + parameters/state + training log."""
function save_mlp(path; kind, indim, outdim, hidden, depth, ps, st,
                  losses = Float64[], train_time = 0.0)
    jldsave(path; kind, indim, outdim, hidden, depth, ps, st, losses, train_time)
end

"""Reload an MLP-family model, rebuilding its architecture and restoring weights."""
function load_mlp(path; rng = StableRNG(SEED))
    d = load(path)
    model, _, _ = make_mlp(d["indim"], d["outdim"];
                           hidden = d["hidden"], depth = d["depth"], rng = rng)
    return (; model, ps = d["ps"], st = d["st"], kind = d["kind"],
            losses = d["losses"], train_time = d["train_time"],
            indim = d["indim"], outdim = d["outdim"])
end

# ----- dataset / basis loaders --------------------------------------------------
"Load the raw snapshot dataset dictionary."
load_snapshots() = load(snapshots_path())

"Load the POD basis as a `PODBasis` plus the cached reconstruction curves."
function load_pod()
    d = load(pod_path())
    return (; P = PODBasis(d["mean"], d["modes"], d["svals"]),
            ranks = d["ranks"], rank_curve = d["rank_curve"],
            recon_train = d["recon_train"], recon_test = d["recon_test"],
            energy = d["energy"])
end
