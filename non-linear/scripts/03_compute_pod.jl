# POD of the steady snapshots: σ-decay, energy fraction, reconstruction floors.
# The σ-decay is the keystone figure — it *is* the low-rank claim for this
# nonlinear problem.

include(joinpath(@__DIR__, "config.jl"))
using CairoMakie
CairoMakie.activate!()

println("=== 03_compute_pod ===")

d = load_snapshots()
Utr = d["U"][:, d["train_idx"]]
Ute = d["U"][:, d["test_idx"]]

P = fit_pod(Utr)
ranks_curve = collect(1:min(40, nmodes(P)))
recon_train = reconstruction_errors(P, Utr, ranks_curve)
recon_test = reconstruction_errors(P, Ute, ranks_curve)
energy = energy_fraction(P)

jldsave(pod_path(); mean = P.mean, modes = P.modes, svals = P.svals,
        ranks = ranks_curve, recon_train, recon_test, energy)

println("rank   σ_r/σ_1        recon(test)    energy captured")
for r in RANKS
    @printf("%4d   %.3e      %.3e      1 - %.2e\n",
            r, P.svals[r] / P.svals[1], recon_test[r], 1 - energy[r])
end

fig = Figure(size = (900, 380))
ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = "mode index r",
           ylabel = "σᵣ / σ₁", title = "POD singular-value decay (steady, nonlinear BC)")
scatterlines!(ax1, ranks_curve, P.svals[ranks_curve] ./ P.svals[1])
ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = "rank r",
           ylabel = "mean relative L² error",
           title = "Reconstruction floor vs rank")
scatterlines!(ax2, ranks_curve, max.(recon_train, 1e-16); label = "train")
scatterlines!(ax2, ranks_curve, max.(recon_test, 1e-16); label = "test")
vlines!(ax2, [DEFAULT_R]; linestyle = :dash, color = :gray)
axislegend(ax2; position = :rt)
save(joinpath(FIG_DIR, "06_pod_decay.png"), fig)

println("saved ", pod_path(), " and figures/06_pod_decay.png")
