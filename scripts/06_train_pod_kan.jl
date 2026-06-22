# 06 — Reduced surrogate (P1): train the POD-KAN  μ ↦ c.
# Same task as the POD-MLP but with a Kolmogorov–Arnold coefficient mapper, which
# is typically smaller and more interpretable.

include("config.jl")
using Lux, KolmogorovArnold
include(joinpath(PROJECT_DIR, "src", "models_kan.jl"))

d = load_snapshots()
pod = load_pod()
Xtr, Utr = d["mu_train"], d["U_train"]
r = DEFAULT_R
Ctr = project(pod.P, Utr, r)

width, grid_size, epochs = 20, 10, 10000
rng = StableRNG(SEED + 3)
model, ps, st = build_pod_kan(r; width, grid_size, rng)

println("Training POD-KAN:  μ ∈ ℝ² → c ∈ ℝ^$r   ($(param_count(model)) params)")
t = @elapsed (ps, losses) = train!(model, ps, st, Xtr, Ctr; epochs, lr = 1e-3,
                                   logevery = 1000, rng)
jldsave(model_path("pod_kan"); kind = "pod_kan", outdim = r, width, grid_size,
        ps, st, losses, train_time = t)
@printf("done in %.1f s,  final MSE %.3e\n", t, losses[end])
println("saved ", relpath(model_path("pod_kan"), PROJECT_DIR))
