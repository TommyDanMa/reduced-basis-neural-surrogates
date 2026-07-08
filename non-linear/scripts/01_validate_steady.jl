# Validation of the steady nonlinear-radiation solver. Produces:
#   figures/01_mms_convergence.png      MMS L² convergence (P1 → order 2)
#   figures/02_analytic_equilibrium.png closed-form 1D radiative equilibrium
#   figures/03_steady_fields.png        config A + config B temperature fields
#   figures/04_newton_trace.png         cold vs warm Newton residual histories
#   figures/05_regime_sanity.png        peak T / edge T / balance / steps over μ-box

include(joinpath(@__DIR__, "config.jl"))
using CairoMakie
CairoMakie.activate!()

println("=== 01_validate_steady ===")

# ----- 1. MMS convergence --------------------------------------------------------
eps_mms = 0.85
errs, hs = Float64[], Float64[]
for (nx, ny) in ((20, 10), (40, 20), (80, 40), (160, 80))
    cfg = RadiativeConfig(; nx, ny)
    fom = build_steady_fom(cfg)
    s = solve_steady(fom; eps_r = eps_mms, Q = 0.0,
                     f = mms_bulk(cfg), g = mms_bc_data(cfg, eps_mms),
                     u0 = fill(300.0, fom.n), abstol = 1e-9)
    @assert s.converged
    push!(errs, l2_error(fom, s.uh, mms_solution))
    push!(hs, cfg.Lx / nx)
end
rates = [log2(errs[i] / errs[i+1]) for i in 1:length(errs)-1]
@printf("MMS L² errors: %s\n", join((@sprintf("%.3e", e) for e in errs), "  "))
@printf("observed rates: %s   (accept band [1.7, 2.2])\n",
        join((@sprintf("%.4f", r) for r in rates), "  "))

fig = Figure(size = (520, 400))
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
          xlabel = "h", ylabel = "‖u_h − u*‖_L²",
          title = "MMS convergence, nonlinear radiation BC (P1)")
scatterlines!(ax, hs, errs; label = "FE error")
lines!(ax, hs, errs[1] .* (hs ./ hs[1]) .^ 2;
       linestyle = :dash, color = :gray, label = "slope 2")
axislegend(ax; position = :rb)
save(joinpath(FIG_DIR, "01_mms_convergence.png"), fig)

# ----- 2. analytic 1D radiative equilibrium --------------------------------------
fom = build_steady_fom(CFG)
q0, eps_1d = 500.0, 0.6
u1 = (q0 * CFG.Lx / (eps_1d * CFG.sigma) + CFG.T_space^4)^(1 / 4)
s1d = solve_steady(fom; eps_r = eps_1d, Q = 0.0, f = x -> q0,
                   u0 = fill(u1, fom.n), abstol = 1e-9)
A1d = to_grid(fom, nodal_values(fom, s1d.uh))
xs, ys = node_axes(fom)
u_ex = [u1 + q0 * (CFG.Lx^2 - x^2) / (2 * CFG.k) for x in xs]
@printf("1D equilibrium: max |u_h − u_exact| = %.2e K,  u(Lx) = %.3f (exact %.3f)\n",
        maximum(abs.(A1d[:, 1] .- u_ex)), A1d[end, 1], u1)

fig = Figure(size = (520, 400))
ax = Axis(fig[1, 1]; xlabel = "x (m)", ylabel = "T (K)",
          title = "Uniform load: FE vs closed form (ε=$(eps_1d), q₀=$(q0) W/m³)")
lines!(ax, collect(xs), u_ex; color = :black, label = "exact parabola")
scatter!(ax, collect(xs)[1:3:end], A1d[1:3:end, 1];
         marker = :circle, markersize = 7, color = :tomato, label = "FE nodes")
hlines!(ax, [u1]; linestyle = :dot, color = :gray)
text!(ax, 0.02, u1 + 1.5; text = "radiative equilibrium u₁", fontsize = 11)
axislegend(ax; position = :lb)
save(joinpath(FIG_DIR, "02_analytic_equilibrium.png"), fig)

# ----- 3. steady temperature fields, configs A and B ------------------------------
fom_a = build_steady_fom(CFG_HOTWALL)
sa = solve_steady(fom_a; eps_r = 0.85, Q = 0.0)
Aa = to_grid(fom_a, nodal_values(fom_a, sa.uh))
μc = (0.5, 450.0)
sb = solve_steady(fom, μc)
Ab = to_grid(fom, nodal_values(fom, sb.uh))
@printf("config A: peak %.1f K  |  config B(ε=%.2f,Q=%.0f): peak %.1f K, edge eq ≈ %.1f K\n",
        maximum(Aa), μc..., maximum(Ab), equilibrium_estimate(CFG, μc...))

fig = Figure(size = (1000, 340))
ax1 = Axis(fig[1, 1]; xlabel = "x", ylabel = "y", aspect = DataAspect(),
           title = "Config A: hot wall 400 K → radiating edge")
hm1 = heatmap!(ax1, xs, ys, Aa; colormap = :inferno)
Colorbar(fig[1, 2], hm1; label = "T (K)")
ax2 = Axis(fig[1, 3]; xlabel = "x", ylabel = "y", aspect = DataAspect(),
           title = "Config B: chip patch (ε=$(μc[1]), Q=$(μc[2]) W/m)")
hm2 = heatmap!(ax2, xs, ys, Ab; colormap = :inferno)
x1, x2, y1, y2 = CFG.patch
lines!(ax2, [x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1];
       color = :cyan, linewidth = 1.5)
Colorbar(fig[1, 4], hm2; label = "T (K)")
save(joinpath(FIG_DIR, "03_steady_fields.png"), fig)

# ----- 4. Newton residual traces ---------------------------------------------------
cold = solve_steady(fom, μc; abstol = 1e-10)
warm = solve_steady(fom, (0.52, 460.0); u0 = cold.x, abstol = 1e-10)
@printf("Newton: cold %d steps (%d res evals), warm %d steps (%d res evals)\n",
        cold.nsteps, length(cold.hist), warm.nsteps, length(warm.hist))

fig = Figure(size = (520, 400))
ax = Axis(fig[1, 1]; yscale = log10, xlabel = "residual evaluation",
          ylabel = "‖R‖₂", title = "Newton convergence (NonlinearSolve)")
scatterlines!(ax, 1:length(cold.hist), cold.hist;
              label = "cold start (equilibrium init)")
scatterlines!(ax, 1:length(warm.hist), warm.hist;
              label = "warm start (neighbouring μ)")
axislegend(ax; position = :rt)
save(joinpath(FIG_DIR, "04_newton_trace.png"), fig)

# ----- 5. regime + balance sweep over the μ-box ------------------------------------
nε, nQ = 9, 9
eps_vals = collect(range(EPS_RANGE...; length = nε))
Q_vals = collect(range(Q_RANGE...; length = nQ))
peakT = zeros(nε, nQ); edgeT = zeros(nε, nQ)
balrel = zeros(nε, nQ); steps = zeros(Int, nε, nQ)
for (j, Q) in enumerate(Q_vals)
    prev = nothing
    for (i, ε) in enumerate(eps_vals)
        s = solve_steady(fom; eps_r = ε, Q = Q, u0 = prev, abstol = 1e-9)
        @assert s.converged "diverged at ε=$ε Q=$Q"
        peakT[i, j] = peak_temperature(fom, s.uh)
        edgeT[i, j] = mean(edge_profile(fom, s.uh)[2])
        balrel[i, j] = energy_balance(fom, s.uh; eps_r = ε, Q = Q).rel
        steps[i, j] = s.nsteps
        prev = s.x
    end
end
@printf("regime: peak T ∈ [%.0f, %.0f] K, edge T ∈ [%.0f, %.0f] K\n",
        minimum(peakT), maximum(peakT), minimum(edgeT), maximum(edgeT))
@printf("balance: worst rel = %.2e   Newton steps: max %d (warm-started sweep)\n",
        maximum(balrel), maximum(steps))

fig = Figure(size = (900, 640))
for (pos, M, ttl, cm) in ((  (1, 1), peakT, "peak T (K)", :inferno),
                          ((1, 3), edgeT, "mean radiating-edge T (K)", :inferno),
                          ((2, 1), log10.(max.(balrel, 1e-16)),
                           "log₁₀ energy-balance rel. error", :viridis),
                          ((2, 3), Float64.(steps), "Newton steps", :viridis))
    axp = Axis(fig[pos...]; xlabel = "ε", ylabel = "Q (W/m)", title = ttl)
    hmp = heatmap!(axp, eps_vals, Q_vals, M; colormap = cm)
    Colorbar(fig[pos[1], pos[2] + 1], hmp)
end
save(joinpath(FIG_DIR, "05_regime_sanity.png"), fig)

println("wrote figures 01–05 to figures/")
