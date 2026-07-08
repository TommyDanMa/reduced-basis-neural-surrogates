# Train the coefficient mappers at every rank in RANKS (POD-MLP and POD-KAN,
# rank-dependent seeds SEED+100+r / SEED+200+r — the deployed models at
# DEFAULT_R *are* the sweep models by construction, so headline numbers can
# never drift from the rank-sweep table), plus the direct μ ↦ u baseline.

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(PROJECT_DIR, "src", "models_kan.jl"))

println("=== 04_train_steady_mappers ===")

d = load_snapshots()
pod = load_pod()
P = pod.P
Xtr = d["X"][:, d["train_idx"]]
Utr = d["U"][:, d["train_idx"]]

# The σ-cliff (rank-3 manifold) gives raw coefficients a ~1e9 dynamic range, so
# each mode is whitened by its training std before the MSE loss; the scales are
# stored with the model and re-applied at prediction time.
for r in RANKS
    Ctr = Matrix(project(P, Utr, r))
    cscales = vec(std(Ctr; dims = 2)) .+ 1e-14 * std(Ctr)
    C̃tr = Ctr ./ cscales
    @printf("\nrank %d: coefficient scales %s\n", r,
            join((@sprintf("%.2e", s) for s in cscales), "  "))

    @printf("POD-MLP  r=%d:  μ ∈ ℝ² → c ∈ ℝ^%d (whitened)\n", r, r)
    rng = StableRNG(SEED + 100 + r)
    model, ps, st = build_pod_mlp(r; rng)
    t = @elapsed ps, losses = train!(model, ps, st, Xtr, C̃tr;
                                     epochs = 6000, logevery = 2000)
    @printf("  done in %.1f s, final MSE %.3e  (%d params)\n",
            t, losses[end], param_count(model))
    save_mlp(model_path("pod_mlp_r$(r)"); kind = "pod_mlp", indim = 2, outdim = r,
             hidden = 64, depth = 3, ps, st, losses, train_time = t, cscales)

    @printf("POD-KAN  r=%d:  μ ∈ ℝ² → c ∈ ℝ^%d (whitened)\n", r, r)
    rng = StableRNG(SEED + 200 + r)
    kan, kps, kst = build_pod_kan(r; width = 20, grid_size = 10, rng)
    t = @elapsed kps, klosses = train!(kan, kps, kst, Xtr, C̃tr;
                                       epochs = 10000, logevery = 2500)
    @printf("  done in %.1f s, final MSE %.3e  (%d params)\n",
            t, klosses[end], param_count(kan))
    jldsave(model_path("pod_kan_r$(r)"); kind = "pod_kan", indim = 2, outdim = r,
            width = 20, grid_size = 10, ps = kps, st = kst,
            losses = klosses, train_time = t, cscales)
end

# Direct baseline: μ ↦ u over all n nodes, trained on (u − ū)/s so the targets
# are O(1) (the fields are hundreds of kelvin).
n = size(Utr, 1)
ū = P.mean
s = std(Utr .- ū)
Yd = (Utr .- ū) ./ s
@printf("\nDirect MLP:  μ ∈ ℝ² → u ∈ ℝ^%d  (scaled targets, s = %.1f K)\n", n, s)
rng = StableRNG(SEED + 300)
dm, dps, dst = build_direct_mlp(n; rng)
t = @elapsed dps, dlosses = train!(dm, dps, dst, Xtr, Yd;
                                   epochs = 3000, logevery = 1000)
@printf("  done in %.1f s, final MSE %.3e  (%d params)\n",
        t, dlosses[end], param_count(dm))
jldsave(model_path("direct_mlp"); kind = "direct_mlp", indim = 2, outdim = n,
        hidden = 64, depth = 3, ps = dps, st = dst, losses = dlosses,
        train_time = t, mean = ū, scale = s)

println("\nsaved models to data/")
