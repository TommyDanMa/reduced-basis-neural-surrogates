# Why does the transient mapper plateau at ~1.5%? Two diagnostics:
#   (a) learning curve: error vs number of training trajectories, same test set;
#       a persisting slope says data-limited, saturation says representation-
#       limited. Report whichever the data shows.
#   (b) parent-style mode-wise view: per-mode signal RMS |cᵢ| vs error RMS
#       |ĉᵢ − cᵢ| for the deployed r=8 mapper.

include(joinpath(@__DIR__, "config.jl"))
using CairoMakie
CairoMakie.activate!()

println("=== 17_mapper_diagnostics ===")

dt_ = load(snapshots_t_path())
pod_t = load(pod_t_path())
Pt = PODBasis(pod_t["mean"], pod_t["modes"], pod_t["svals"])
U3, mus_t = dt_["U"], dt_["mus"]
n, nt, _ = size(U3)
ttr, tte = dt_["train_idx"], dt_["test_idx"]
rt = DEFAULT_R_T

feats(k) = reduce(hcat, [transient_features(standardize_mu_t(mus_t[:, k]...), j)
                         for j in 1:nt])

function eval_rel(model, ps, st, cst)
    rels = Float64[]
    for k in tte
        Û = Matrix(reconstruct(Pt, cst .* predict(model, ps, st, feats(k)), rt))
        push!(rels, norm(Û - U3[:, :, k]) / norm(U3[:, :, k]))
    end
    return mean(rels)
end

# ----- (a) learning curve ----------------------------------------------------------
sizes = (24, 48, length(ttr))
lc = Float64[]
for (i, M) in enumerate(sizes)
    sub = ttr[1:M]
    X = reduce(hcat, [feats(k) for k in sub])
    C = Matrix(project(Pt, reshape(U3[:, :, sub], n, :), rt))
    s_ = vec(std(C; dims = 2))
    cst = max.(s_, 0.01 * s_[1])
    model, ps, st = build_pod_mlp(rt; indim = size(X, 1), hidden = 128,
                                  rng = StableRNG(SEED + 8000 + i))
    ps, _ = train!(model, ps, st, X, C ./ cst; epochs = 8000, verbose = false)
    push!(lc, eval_rel(model, ps, st, cst))
    @printf("  %2d trajectories → test traj rel L² %.3e\n", M, lc[end])
end
slopes = [log2(lc[i] / lc[i+1]) / log2(sizes[i+1] / sizes[i]) for i in 1:length(lc)-1]
@printf("learning-curve slopes (error ~ M^-s): %s\n",
        join((@sprintf("%.2f", s) for s in slopes), "  "))

# ----- (b) mode-wise diagnostic for the deployed mapper ----------------------------
m = load_mlp(model_path("pod_mlp_t_r$(rt)"))
Xte_all = reduce(hcat, [feats(k) for k in tte])
Cte = Matrix(project(Pt, reshape(U3[:, :, tte], n, :), rt))
Ĉte = m.cscales .* predict(m.model, m.ps, m.st, Xte_all)
sig = vec(sqrt.(mean(abs2, Cte; dims = 2)))
err = vec(sqrt.(mean(abs2, Ĉte .- Cte; dims = 2)))
println("mode   RMS |c|     RMS |ĉ−c|   relative")
for i in 1:rt
    @printf("%4d   %.3e   %.3e   %.2f\n", i, sig[i], err[i], err[i] / sig[i])
end

jldsave(joinpath(DATA_DIR, "mapper_diagnostics.jld2");
        sizes = collect(sizes), lc, slopes, sig, err)

fig = Figure(size = (900, 380))
ax1 = Axis(fig[1, 1]; xscale = log2, yscale = log10,
           xlabel = "training trajectories M", ylabel = "test traj rel. L²",
           xticks = (collect(sizes), string.(collect(sizes))),
           title = "Learning curve (transient mapper, r = $rt)")
scatterlines!(ax1, collect(sizes), lc)
lines!(ax1, collect(sizes), lc[1] .* (collect(sizes) ./ sizes[1]) .^ (-0.5);
       linestyle = :dash, color = :gray, label = "M^-1/2 reference")
axislegend(ax1; position = :rt)
ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = "mode index i", ylabel = "RMS",
           title = "Per-mode signal vs mapper error (deployed r = $rt)")
scatterlines!(ax2, 1:rt, max.(sig, 1e-17); label = "signal |cᵢ|")
scatterlines!(ax2, 1:rt, max.(err, 1e-17); label = "error |ĉᵢ − cᵢ|")
axislegend(ax2; position = :rt)
save(joinpath(FIG_DIR, "18_mode_diagnostics.png"), fig)

println("saved data/mapper_diagnostics.jld2 and figures/18_mode_diagnostics.png")
