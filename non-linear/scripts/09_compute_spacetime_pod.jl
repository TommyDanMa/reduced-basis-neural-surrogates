# Space-time POD: the SVD of all saved transient states of the training
# trajectories. The comparison against the steady σ-decay is the point — fixed
# load geometry makes the *steady* manifold rank-3, while switching transients
# excite genuinely higher-dimensional diffusion dynamics.

include(joinpath(@__DIR__, "config.jl"))
using CairoMakie
CairoMakie.activate!()

println("=== 09_compute_spacetime_pod ===")

d = load_snapshots()                        # steady (for the decay contrast)
P_s = fit_pod(d["U"][:, d["train_idx"]])

dt_ = load(snapshots_t_path())
U3 = dt_["U"]
n, nt, _ = size(U3)
Utr = reshape(U3[:, :, dt_["train_idx"]], n, :)
Ute = reshape(U3[:, :, dt_["test_idx"]], n, :)
@printf("space-time snapshot matrix: %d × %d\n", size(Utr)...)

P = fit_pod(Utr)
ranks_curve = collect(1:min(60, nmodes(P)))
recon_train = reconstruction_errors(P, Utr, ranks_curve)
recon_test = reconstruction_errors(P, Ute, ranks_curve)
energy = energy_fraction(P)

jldsave(pod_t_path(); mean = P.mean, modes = P.modes[:, 1:min(80, nmodes(P))],
        svals = P.svals, ranks = ranks_curve, recon_train, recon_test, energy)

println("rank   σ_r/σ_1        recon(test)")
for r in RANKS_T
    @printf("%4d   %.3e      %.3e\n", r, P.svals[r] / P.svals[1], recon_test[r])
end

ns = min(40, length(P_s.svals))
fig = Figure(size = (900, 380))
ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = "mode index r", ylabel = "σᵣ / σ₁",
           title = "Steady vs space-time σ-decay")
scatterlines!(ax1, 1:ns, max.(P_s.svals[1:ns] ./ P_s.svals[1], 1e-17);
              label = "steady (μ = ε, Q)")
scatterlines!(ax1, ranks_curve, max.(P.svals[ranks_curve] ./ P.svals[1], 1e-17);
              label = "space-time (μ = ε, Qp, duty, period)")
axislegend(ax1; position = :rt)
ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = "rank r",
           ylabel = "mean rel. L² reconstruction error",
           title = "Space-time reconstruction floor")
scatterlines!(ax2, ranks_curve, max.(recon_train, 1e-17); label = "train")
scatterlines!(ax2, ranks_curve, max.(recon_test, 1e-17); label = "test")
vlines!(ax2, [DEFAULT_R_T]; linestyle = :dash, color = :gray)
axislegend(ax2; position = :rt)
save(joinpath(FIG_DIR, "11_spacetime_decay.png"), fig)

println("saved ", pod_t_path(), " and figures/11_spacetime_decay.png")
