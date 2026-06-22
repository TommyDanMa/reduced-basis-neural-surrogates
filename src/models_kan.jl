# POD-KAN coefficient mapper (P1).
#
# Kept separate from the core module so the P0 pipeline never depends on
# KolmogorovArnold. Loaded explicitly by scripts that use the KAN (06, and 07/09
# when a trained KAN is present). `train!` and `predict` from the core package work
# unchanged because they are model-agnostic.

using Lux, KolmogorovArnold, Random

"""
    build_pod_kan(r; width=16, grid_size=6, rng) -> (model, ps, st)

Two-layer Kolmogorov–Arnold network `μ ∈ ℝ² ↦ c ∈ ℝ^r` using RBF `KDense` layers.
Parameters/state are converted to Float64 to match the POD pipeline.
"""
function build_pod_kan(r::Integer; width::Integer = 16, grid_size::Integer = 6,
                       rng::AbstractRNG = Random.default_rng())
    model = Chain(KDense(2, width, grid_size), KDense(width, r, grid_size))
    ps, st = Lux.setup(rng, model)
    return model, Lux.f64(ps), Lux.f64(st)
end

"""Reload a trained POD-KAN, rebuilding its architecture and restoring weights."""
function load_kan(path; rng = StableRNG(SEED))
    d = JLD2.load(path)
    model, _, _ = build_pod_kan(d["outdim"]; width = d["width"],
                                grid_size = d["grid_size"], rng = rng)
    return (; model, ps = d["ps"], st = d["st"], kind = "pod_kan",
            outdim = d["outdim"], losses = d["losses"], train_time = d["train_time"])
end
