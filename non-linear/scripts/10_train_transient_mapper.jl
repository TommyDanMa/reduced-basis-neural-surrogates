# Space-time coefficient mapper: (μ̃, time features) ∈ ℝ⁷ ↦ c ∈ ℝ^r, trained at
# every rank in RANKS_T with whitened targets (seeds SEED+400+r, so deployed =
# sweep at DEFAULT_R_T by construction, as in the steady pipeline).

include(joinpath(@__DIR__, "config.jl"))

println("=== 10_train_transient_mapper ===")

dt_ = load(snapshots_t_path())
pod = load(pod_t_path())
P = PODBasis(pod["mean"], pod["modes"], pod["svals"])
U3, mus, tr = dt_["U"], dt_["mus"], dt_["train_idx"]
n, nt, _ = size(U3)

Xtr = reduce(hcat, [transient_features(standardize_mu_t(mus[:, k]...), j)
                    for k in tr for j in 1:nt])
Utr = reshape(U3[:, :, tr], n, :)
indim = size(Xtr, 1)
@printf("training set: %d inputs of dim %d\n", size(Xtr, 2), indim)

for r in RANKS_T
    Ctr = Matrix(project(P, Utr, r))
    # Capped whitening: full whitening would upweight σ₈-level modes by ~3e4
    # and the net burns its capacity on unlearnable detail; the 1% floor keeps
    # the objective field-error-driven while still lifting the mid modes.
    s = vec(std(Ctr; dims = 2))
    cscales = max.(s, 0.01 * s[1])
    C̃tr = Ctr ./ cscales

    @printf("\nPOD-MLP-t  r=%d:  (μ̃, t̃, phase) ∈ ℝ^%d → c ∈ ℝ^%d (capped whitening)\n",
            r, indim, r)
    rng = StableRNG(SEED + 400 + r)
    model, ps, st = build_pod_mlp(r; indim, hidden = 128, rng)
    t = @elapsed ps, losses = train!(model, ps, st, Xtr, C̃tr;
                                     epochs = 8000, logevery = 2000)
    @printf("  done in %.1f s, final MSE %.3e  (%d params)\n",
            t, losses[end], param_count(model))
    save_mlp(model_path("pod_mlp_t_r$(r)"); kind = "pod_mlp_t", indim,
             outdim = r, hidden = 128, depth = 3, ps, st, losses,
             train_time = t, cscales)
end

println("\nsaved transient mappers to data/")
