# Pre-registered rank-prediction experiments (see notes/rank-note.md §3, committed
# before this script was run):
#   P1  second heater patch, μ = (ε, Q₁, Q₂):    predict centered rank 3, cliff at σ₄
#   P2  conductivity,        μ = (ε, Q, k):      predict NO new O(σ₂) mode (Q/k collapse),
#                                                σ₄/σ₁ < 1e-5
#   P3  box shrinking (ε-axis and Q-axis):       predict log-log slope in [1,2] on the
#                                                dominant-curvature axis
# The script prints measurements; the note's §4 reports them against the predictions,
# hits and misses alike.

include(joinpath(@__DIR__, "config.jl"))
using QuasiMonteCarlo
using CairoMakie
CairoMakie.activate!()

println("=== 19_rank_prediction ===")

fom = build_steady_fom(CFG)
const PATCH2 = (0.60, 0.75, 0.15, 0.35)          # disjoint from cfg.patch, grid-aligned
area2 = (PATCH2[2] - PATCH2[1]) * (PATCH2[4] - PATCH2[3])
source2(Q2) = x -> (PATCH2[1] <= x[1] <= PATCH2[2] &&
                    PATCH2[3] <= x[2] <= PATCH2[4]) ? Q2 / area2 : 0.0

"POD σ-spectrum (normalized) of NS Sobol solves with per-sample solver kwargs."
function spectrum(NS, lb, ub; solvekw)
    mus = QuasiMonteCarlo.sample(NS, lb, ub, SobolSample())
    U = Matrix{Float64}(undef, fom.n, NS)
    for i in 1:NS
        s = solve_steady(fom; solvekw(mus[:, i])..., abstol = 1e-9)
        @assert s.converged
        U[:, i] = s.x
    end
    P = fit_pod(U)
    return P.svals ./ P.svals[1]
end

# ----- P1: second heater --------------------------------------------------------------
d_p1 = spectrum(384, [EPS_RANGE[1], Q_RANGE[1], Q_RANGE[1]],
                [EPS_RANGE[2], Q_RANGE[2], Q_RANGE[2]];
                solvekw = m -> (eps_r = m[1], Q = m[2], f = source2(m[3])))
@printf("P1 (ε,Q1,Q2):  σ2 %.2e  σ3 %.2e  σ4 %.2e  σ5 %.2e\n", d_p1[2:5]...)

# ----- P2: conductivity ---------------------------------------------------------------
d_p2 = spectrum(384, [EPS_RANGE[1], Q_RANGE[1], 2.0],
                [EPS_RANGE[2], Q_RANGE[2], 10.0];
                solvekw = m -> (eps_r = m[1], Q = m[2], k = m[3]))
@printf("P2 (ε,Q,k):    σ2 %.2e  σ3 %.2e  σ4 %.2e  σ5 %.2e\n", d_p2[2:5]...)

# ----- P3: box shrinking ----------------------------------------------------------------
εc, Qc = 0.525, 450.0
εh_full, Qh_full = 0.425, 350.0
fracs = (1.0, 0.5, 0.25)
sig3_eps = Float64[]
sig3_Q = Float64[]
for fr in fracs
    dε = spectrum(256, [εc - fr * εh_full, Q_RANGE[1]],
                  [εc + fr * εh_full, Q_RANGE[2]];
                  solvekw = m -> (eps_r = m[1], Q = m[2]))
    push!(sig3_eps, dε[3])
    dQ = spectrum(256, [EPS_RANGE[1], Qc - fr * Qh_full],
                  [EPS_RANGE[2], Qc + fr * Qh_full];
                  solvekw = m -> (eps_r = m[1], Q = m[2]))
    push!(sig3_Q, dQ[3])
end
slope(v) = [log2(v[i] / v[i+1]) for i in 1:length(v)-1]
@printf("P3 ε-shrink:  σ3/σ1 %s   slopes %s\n",
        join((@sprintf("%.2e", s) for s in sig3_eps), "  "),
        join((@sprintf("%.2f", s) for s in slope(sig3_eps)), "  "))
@printf("P3 Q-shrink:  σ3/σ1 %s   slopes %s\n",
        join((@sprintf("%.2e", s) for s in sig3_Q), "  "),
        join((@sprintf("%.2f", s) for s in slope(sig3_Q)), "  "))

base = load_pod()
d_base = base.P.svals ./ base.P.svals[1]

jldsave(joinpath(DATA_DIR, "rank_prediction.jld2");
        d_base = d_base[1:12], d_p1 = d_p1[1:12], d_p2 = d_p2[1:12],
        fracs = collect(fracs), sig3_eps, sig3_Q)

fig = Figure(size = (940, 400))
ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = "mode index r", ylabel = "σᵣ/σ₁",
           title = "Adding a parameter: source adds a mode, conductivity does not")
scatterlines!(ax1, 1:8, max.(d_base[1:8], 1e-17); label = "μ = (ε, Q)  [d = 2]")
scatterlines!(ax1, 1:8, max.(d_p1[1:8], 1e-17); label = "μ = (ε, Q₁, Q₂)  [P1]")
scatterlines!(ax1, 1:8, max.(d_p2[1:8], 1e-17); label = "μ = (ε, Q, k)  [P2]")
axislegend(ax1; position = :rt)
ax2 = Axis(fig[1, 2]; xscale = log2, yscale = log10, xlabel = "box fraction",
           ylabel = "σ₃/σ₁", title = "Remainder vs box size [P3]")
scatterlines!(ax2, collect(fracs), sig3_eps; label = "shrink ε-box")
scatterlines!(ax2, collect(fracs), sig3_Q; label = "shrink Q-box")
lines!(ax2, collect(fracs), sig3_eps[1] .* collect(fracs) .^ 2;
       linestyle = :dash, color = :gray, label = "slope 2")
axislegend(ax2; position = :rb)
save(joinpath(FIG_DIR, "20_rank_prediction.png"), fig)

println("saved data/rank_prediction.jld2 and figures/20_rank_prediction.png")
