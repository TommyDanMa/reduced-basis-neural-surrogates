# POD-KAN coefficient mapper — compact baseline mapper, ported from the parent.
#
# Kept separate from the core module so the core pipeline never depends on
# KolmogorovArnold. Loaded explicitly by the scripts that use the KAN. `train!`
# and `predict` from the core package work unchanged (model-agnostic).

using Lux, KolmogorovArnold, Random

"""
    build_pod_kan(r; indim=2, width=16, grid_size=6, rng) -> (model, ps, st)

Two-layer Kolmogorov–Arnold network `μ ∈ ℝ^indim ↦ c ∈ ℝ^r` using RBF `KDense`
layers. Parameters/state are converted to Float64 to match the POD pipeline.
"""
function build_pod_kan(r::Integer; indim::Integer = 2, width::Integer = 16,
                       grid_size::Integer = 6,
                       rng::AbstractRNG = Random.default_rng())
    model = Chain(KDense(indim, width, grid_size), KDense(width, r, grid_size))
    ps, st = Lux.setup(rng, model)
    return model, Lux.f64(ps), Lux.f64(st)
end

"""Reload a trained POD-KAN, rebuilding its architecture and restoring weights."""
function load_kan(path; rng = StableRNG(SEED))
    d = JLD2.load(path)
    model, _, _ = build_pod_kan(d["outdim"]; indim = get(d, "indim", 2),
                                width = d["width"], grid_size = d["grid_size"],
                                rng = rng)
    return (; model, ps = d["ps"], st = d["st"], kind = "pod_kan",
            outdim = d["outdim"], losses = d["losses"], train_time = d["train_time"],
            cscales = get(d, "cscales", Float64[]))
end
