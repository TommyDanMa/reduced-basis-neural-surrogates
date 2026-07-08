# Is the steady rank-3 collapse a discretization artifact? Replicate the σ-decay
# on a 2× finer grid (120×60, P1) and with quadratic elements (60×30, P2). POD
# runs on raw free-DOF vectors, so no structured-grid extraction is needed.

include(joinpath(@__DIR__, "config.jl"))
using QuasiMonteCarlo
using CairoMakie
CairoMakie.activate!()

println("=== 16_grid_independence ===")

NS = 256
lb = [EPS_RANGE[1], Q_RANGE[1]]
ub = [EPS_RANGE[2], Q_RANGE[2]]
mus = QuasiMonteCarlo.sample(NS, lb, ub, SobolSample())

"σ-decay of the steady manifold for a given config (POD on raw DOF vectors)."
function decay_for(cfg; label)
    fom = build_steady_fom(cfg)
    U = Matrix{Float64}(undef, fom.n, NS)
    t = @elapsed for k in 1:NS
        s = solve_steady(fom, (mus[1, k], mus[2, k]); abstol = 1e-9)
        @assert s.converged
        U[:, k] = s.x
    end
    P = fit_pod(U)
    @printf("%-14s n=%6d  %.0f s   σ₂/σ₁ %.3e   σ₃/σ₁ %.3e   σ₅/σ₁ %.3e\n",
            label, fom.n, t, P.svals[2] / P.svals[1], P.svals[3] / P.svals[1],
            P.svals[5] / P.svals[1])
    return P.svals ./ P.svals[1]
end

base = load_pod()
d_base = base.P.svals ./ base.P.svals[1]
@printf("%-14s (from pod.jld2)      σ₂/σ₁ %.3e   σ₃/σ₁ %.3e   σ₅/σ₁ %.3e\n",
        "60×30 P1", d_base[2], d_base[3], d_base[5])

d_fine = decay_for(RadiativeConfig(nx = 120, ny = 60); label = "120×60 P1")
d_p2 = decay_for(RadiativeConfig(order = 2, quad_bnd = 10); label = "60×30 P2")

jldsave(joinpath(DATA_DIR, "grid_independence.jld2");
        d_base = d_base[1:20], d_fine = d_fine[1:20], d_p2 = d_p2[1:20])

fig = Figure(size = (520, 400))
ax = Axis(fig[1, 1]; yscale = log10, xlabel = "mode index r", ylabel = "σᵣ/σ₁",
          title = "Steady σ-decay is discretization-independent")
scatterlines!(ax, 1:12, max.(d_base[1:12], 1e-17); label = "60×30 P1 (baseline)")
scatterlines!(ax, 1:12, max.(d_fine[1:12], 1e-17); label = "120×60 P1")
scatterlines!(ax, 1:12, max.(d_p2[1:12], 1e-17); label = "60×30 P2")
axislegend(ax; position = :rt)
save(joinpath(FIG_DIR, "17_grid_independence.png"), fig)

println("saved data/grid_independence.jld2 and figures/17_grid_independence.png")
