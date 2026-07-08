# Design-study capstone: pick an emissivity coating ε and a load duty cycle to
# keep the worst-case chip temperature down. The FOM answers with one transient
# solve per candidate; the surrogate answers with matrix algebra. The claim
# defended here is not raw speed but *ranking fidelity*: the surrogate must
# order the candidates the way the FOM does.

include(joinpath(@__DIR__, "config.jl"))
using NonlinearSolve

println("=== 13_design_study ===")

fom = build_steady_fom(CFG)
pod = load(pod_t_path())
P = PODBasis(pod["mean"], pod["modes"], pod["svals"])
m = load_mlp(model_path("pod_mlp_t_r$(DEFAULT_R_T)"))

Qp, period = 650.0, 5400.0
nε, nd = 10, 10
eps_grid = collect(range(0.15, 0.9; length = nε))
duty_grid = collect(range(0.25, 0.75; length = nd))
nt = NCYCLES * NPHASE + 1
lastcycle = (2 * NPHASE + 1):nt          # steady-cycling window

"Worst-case chip temperature over the final load cycle (the design QoI)."
qoi(traj) = maximum(maximum(view(traj, :, j)) for j in lastcycle)

# ----- FOM sweep -------------------------------------------------------------------
fomQ = zeros(nε, nd)
t_fom = @elapsed for (i, ε) in enumerate(eps_grid), (j, dy) in enumerate(duty_grid)
    T_end = NCYCLES * period
    u0 = solve_steady(fom; eps_r = ε, Q = dy * Qp, abstol = 1e-9).x
    sol = solve_transient(fom; eps_r = ε, load = square_wave(Qp, period, dy),
                          u0, tspan = (0.0, T_end),
                          tstops = load_switch_times(period, dy, 0.0, T_end),
                          saveat = collect(range(0.0, T_end; length = nt)),
                          abstol = 1e-7, reltol = 1e-7)
    @assert NonlinearSolve.SciMLBase.successful_retcode(sol)
    fomQ[i, j] = qoi(Array(sol))
end

# ----- surrogate sweep -------------------------------------------------------------
surQ = zeros(nε, nd)
t_sur = @elapsed for (i, ε) in enumerate(eps_grid), (j, dy) in enumerate(duty_grid)
    μst = standardize_mu_t(ε, Qp, dy, period)
    X = reduce(hcat, [transient_features(μst, jj) for jj in 1:nt])
    Ĉ = m.cscales .* predict(m.model, m.ps, m.st, X)
    surQ[i, j] = qoi(Matrix(reconstruct(P, Ĉ, DEFAULT_R_T)))
end

# ----- ranking fidelity ------------------------------------------------------------
rankvec(v) = invperm(sortperm(v))
rf, rs = rankvec(vec(fomQ)), rankvec(vec(surQ))
n_ = length(rf)
spearman = 1 - 6 * sum(abs2, rf .- rs) / (n_ * (n_^2 - 1))
top5f = sortperm(vec(fomQ))[1:5]
top5s = sortperm(vec(surQ))[1:5]
top5 = length(intersect(top5f, top5s))
best_f = argmin(fomQ); best_s = argmin(surQ)

@printf("\n%d candidates:  FOM sweep %.1f s   surrogate sweep %.3f s  (%.0f× )\n",
        n_, t_fom, t_sur, t_fom / t_sur)
@printf("QoI error: median %.2f K, max %.2f K (QoI range %.0f–%.0f K)\n",
        median(abs.(surQ .- fomQ)), maximum(abs.(surQ .- fomQ)),
        minimum(fomQ), maximum(fomQ))
@printf("Spearman rank correlation: %.4f   top-5 overlap: %d/5\n", spearman, top5)
@printf("best design: FOM (ε=%.2f, duty=%.2f) vs surrogate (ε=%.2f, duty=%.2f)\n",
        eps_grid[best_f[1]], duty_grid[best_f[2]],
        eps_grid[best_s[1]], duty_grid[best_s[2]])

jldsave(joinpath(DATA_DIR, "design_study.jld2");
        eps_grid, duty_grid, fomQ, surQ, t_fom, t_sur, spearman, top5,
        Qp, period)

using CairoMakie
CairoMakie.activate!()
fig = Figure(size = (960, 380))
lims = (minimum([fomQ; surQ]), maximum([fomQ; surQ]))
ax1 = Axis(fig[1, 1]; xlabel = "ε", ylabel = "duty",
           title = @sprintf("FOM: worst-case chip T (K), %.1f s", t_fom))
hm1 = heatmap!(ax1, eps_grid, duty_grid, fomQ; colorrange = lims, colormap = :inferno)
scatter!(ax1, [eps_grid[best_f[1]]], [duty_grid[best_f[2]]];
         marker = :star5, color = :cyan, markersize = 18)
ax2 = Axis(fig[1, 2]; xlabel = "ε", ylabel = "duty",
           title = @sprintf("Surrogate: same QoI, %.3f s (Spearman %.3f)",
                            t_sur, spearman))
hm2 = heatmap!(ax2, eps_grid, duty_grid, surQ; colorrange = lims, colormap = :inferno)
scatter!(ax2, [eps_grid[best_s[1]]], [duty_grid[best_s[2]]];
         marker = :star5, color = :cyan, markersize = 18)
Colorbar(fig[1, 3], hm2; label = "worst-case chip T (K)")
save(joinpath(FIG_DIR, "14_design_study.png"), fig)

println("saved data/design_study.jld2 and figures/14_design_study.png")
