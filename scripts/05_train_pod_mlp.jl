# 05 — Reduced surrogate: train the POD-MLP  μ ↦ c  (predicts r coefficients).
# Reconstruction û = ū + Φᵣc then recovers the field at full resolution.

include("config.jl")

d = load_snapshots()
pod = load_pod()
Xtr, Utr = d["mu_train"], d["U_train"]
r = DEFAULT_R
Ctr = project(pod.P, Utr, r)            # r × M training targets

hidden, depth, epochs = 64, 3, 6000
rng = StableRNG(SEED + 100 + r)   # same seed as the scripts/11 sweep at this rank,
                                  # so the headline equals the §8.1 r-sweep row
model, ps, st = build_pod_mlp(r; hidden, depth, rng)

println("Training POD-MLP:  μ ∈ ℝ² → c ∈ ℝ^$r   ($(param_count(model)) params)")
t = @elapsed (ps, losses) = train!(model, ps, st, Xtr, Ctr; epochs, lr = 1e-3,
                                   logevery = 1000, rng)
save_mlp(model_path("pod_mlp"); kind = "pod_mlp", indim = 2, outdim = r,
         hidden, depth, ps, st, losses, train_time = t)
@printf("done in %.1f s,  final MSE %.3e\n", t, losses[end])
println("saved ", relpath(model_path("pod_mlp"), PROJECT_DIR))
