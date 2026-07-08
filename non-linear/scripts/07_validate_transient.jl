# Transient validation on the production grid. Produces:
#   figures/09_transient_convergence.png  dt-orders of ImplicitEuler/Trapezoid
#   figures/10_load_cycle_response.png    square-wave cycle: temps, powers, energy
# and prints the time-constant sanity numbers (τ_diff, τ_rad vs load period).

include(joinpath(@__DIR__, "config.jl"))
using Gridap
using NonlinearSolve
using CairoMakie
CairoMakie.activate!()

println("=== 07_validate_transient ===")

fom = build_steady_fom(CFG)
eps_r = 0.5

# ----- time-constant sanity -------------------------------------------------------
T_eq = equilibrium_estimate(CFG, eps_r, 450.0)
τ_diff = CFG.rho_c * CFG.Lx^2 / CFG.k
τ_rad = CFG.rho_c * (CFG.Lx * CFG.Ly / CFG.Ly) /
        (4 * eps_r * CFG.sigma * T_eq^3)
@printf("τ_diff = ρcL²/k = %.2e s   τ_rad = ρc(V/A)/(4εσT³) = %.2e s\n",
        τ_diff, τ_rad)

# ----- dt-convergence (smooth load, tight FBDF reference) --------------------------
load_s = t -> 300.0 + 150.0 * sin(2pi * t / 2000.0)
u0 = solve_steady(fom; eps_r, Q = 300.0, abstol = 1e-10).x
T = 2000.0
uref = solve_transient(fom; eps_r, load = load_s, u0, tspan = (0.0, T),
                       abstol = 1e-11, reltol = 1e-11, saveat = [T]).u[end]
dts = (400.0, 200.0, 100.0, 50.0)
dt_err(alg, dt) = relative_l2_error(
    solve_transient(fom; eps_r, load = load_s, u0, tspan = (0.0, T),
                    alg, dt, saveat = [T]).u[end], uref)
e_be = [dt_err(ImplicitEuler(), dt) for dt in dts]
e_tr = [dt_err(Trapezoid(), dt) for dt in dts]
r_be = [log2(e_be[i] / e_be[i+1]) for i in 1:length(dts)-1]
r_tr = [log2(e_tr[i] / e_tr[i+1]) for i in 1:length(dts)-1]
@printf("ImplicitEuler rates: %s\n", join((@sprintf("%.3f", r) for r in r_be), "  "))
@printf("Trapezoid    rates: %s\n", join((@sprintf("%.3f", r) for r in r_tr), "  "))

fig = Figure(size = (520, 400))
ax = Axis(fig[1, 1]; xscale = log10, yscale = log10, xlabel = "Δt (s)",
          ylabel = "rel. L² error at t = $(Int(T)) s",
          title = "θ-family dt-convergence (vs FBDF 1e-11)")
scatterlines!(ax, collect(dts), e_be; label = "ImplicitEuler (BE)")
scatterlines!(ax, collect(dts), e_tr; label = "Trapezoid (CN)")
lines!(ax, collect(dts), e_be[1] .* (collect(dts) ./ dts[1]);
       linestyle = :dash, color = :gray, label = "slope 1")
lines!(ax, collect(dts), e_tr[1] .* (collect(dts) ./ dts[1]) .^ 2;
       linestyle = :dot, color = :gray, label = "slope 2")
axislegend(ax; position = :rb)
save(joinpath(FIG_DIR, "09_transient_convergence.png"), fig)

# ----- square-wave load cycle (orbital-ish) ----------------------------------------
Qp, period, duty = 600.0, 5400.0, 0.6
ncycles = 3
Qmean = Qp * duty
u0c = solve_steady(fom; eps_r, Q = Qmean, abstol = 1e-10).x   # spin-up shortcut
T_end = ncycles * period
ts = load_switch_times(period, duty, 0.0, T_end)
load_q = square_wave(Qp, period, duty)
tsave = collect(0.0:60.0:T_end)
tsolve = @elapsed sol = solve_transient(fom; eps_r, load = load_q, u0 = u0c,
                                        tspan = (0.0, T_end), tstops = ts,
                                        saveat = tsave, abstol = 1e-8,
                                        reltol = 1e-8)
@assert NonlinearSolve.SciMLBase.successful_retcode(sol)
@printf("cycle run: %d saved states, FBDF, %.1f s wall\n", length(sol.t), tsolve)

peakT_t = [maximum(u) for u in sol.u]
edgeT_t = [mean(to_grid(fom, u)[end, :]) for u in sol.u]
Prad_t = [radiated_power(fom, FEFunction(fom.Un, u), eps_r) for u in sol.u]
Pin_t = load_q.(sol.t)
@printf("peak T swing over cycle: %.1f → %.1f K (Δ %.1f K)\n",
        minimum(peakT_t), maximum(peakT_t), maximum(peakT_t) - minimum(peakT_t))

fig = Figure(size = (900, 560))
ax1 = Axis(fig[1, 1]; ylabel = "T (K)",
           title = @sprintf("Square-wave load: Qp=%.0f W/m, period=%.0f s, duty=%.1f, ε=%.1f",
                            Qp, period, duty, eps_r))
lines!(ax1, sol.t ./ 3600, peakT_t; label = "peak T (chip)")
lines!(ax1, sol.t ./ 3600, edgeT_t; label = "mean radiating-edge T")
axislegend(ax1; position = :rt)
ax2 = Axis(fig[2, 1]; xlabel = "t (h)", ylabel = "P (W/m)")
stairs!(ax2, sol.t ./ 3600, Pin_t; label = "P_in (load)", color = :gray)
lines!(ax2, sol.t ./ 3600, Prad_t; label = "P_rad (radiated)", color = :tomato)
axislegend(ax2; position = :rt)
save(joinpath(FIG_DIR, "10_load_cycle_response.png"), fig)

println("wrote figures 09–10")
