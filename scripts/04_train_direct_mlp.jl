# 04 — Baseline: train the direct MLP  μ ↦ u  (predicts the whole field).
# This is the "obvious" surrogate the reduced-basis approach is compared against.

include("config.jl")

d = load_snapshots()
Xtr, Utr = d["mu_train"], d["U_train"]
g = make_grid(d["grid_n"])

hidden, depth, epochs = 64, 3, 8000
rng = StableRNG(SEED + 1)
model, ps, st = build_direct_mlp(ndof(g); hidden, depth, rng)

# The large (ndof-dimensional) output needs sustained learning rate, so decay only
# mildly here (POD-MLP, with its tiny output, uses the stronger default schedule).
println("Training DIRECT MLP:  μ ∈ ℝ² → u ∈ ℝ^$(ndof(g))   ($(param_count(model)) params)")
t = @elapsed (ps, losses) = train!(model, ps, st, Xtr, Utr; epochs, lr = 2e-3,
                                   lr_final = 2e-4, logevery = 2000, rng)
save_mlp(model_path("direct"); kind = "direct_mlp", indim = 2, outdim = ndof(g),
         hidden, depth, ps, st, losses, train_time = t)
@printf("done in %.1f s,  final MSE %.3e\n", t, losses[end])
println("saved ", relpath(model_path("direct"), PROJECT_DIR))
