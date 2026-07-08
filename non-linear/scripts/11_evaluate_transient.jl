# Held-out evaluation of the space-time surrogate: trajectory-wise relative L²,
# radiated-power tracking through load switches, and peak-temperature-over-
# cycle errors — the quantities a thermal design study actually consumes.

include(joinpath(@__DIR__, "config.jl"))
using Gridap
using CairoMakie
CairoMakie.activate!()

println("=== 11_evaluate_transient ===")

fom = build_steady_fom(CFG)
dt_ = load(snapshots_t_path())
pod = load(pod_t_path())
P = PODBasis(pod["mean"], pod["modes"], pod["svals"])
U3, mus, te = dt_["U"], dt_["mus"], dt_["test_idx"]
n, nt, _ = size(U3)
ntest = length(te)

"Predict the full trajectory matrix (n × nt) for test sample k at rank r."
function predict_traj(m, k, r)
    Xk = reduce(hcat, [transient_features(standardize_mu_t(mus[:, k]...), j)
                       for j in 1:nt])
    Ĉ = m.cscales .* predict(m.model, m.ps, m.st, Xk)
    return Matrix(reconstruct(P, Ĉ, r))
end

traj_rel(Û, Uk) = norm(Û - Uk) / norm(Uk)

sweep_rel = Float64[]
for r in RANKS_T
    m = load_mlp(model_path("pod_mlp_t_r$(r)"))
    rels = [traj_rel(predict_traj(m, k, r), U3[:, :, k]) for k in te]
    push!(sweep_rel, mean(rels))
end
floor_rel = [pod["recon_test"][r] for r in RANKS_T]

println("rank sweep (space-time mapper):")
for (i, r) in enumerate(RANKS_T)
    @printf("  r=%2d  rel L² %.3e   (POD floor %.3e)\n",
            r, sweep_rel[i], floor_rel[i])
end

# ----- deployed-rank diagnostics ---------------------------------------------------
r = DEFAULT_R_T
m = load_mlp(model_path("pod_mlp_t_r$(r)"))
err_t = zeros(nt)                     # mean over test set, per saved state
dTpk = Float64[]                      # peak-T error over each trajectory
dPq = Float64[]                       # radiated-power tracking error (rel)
worst_ref = Ref((te[1], -1.0))
for k in te
    Û = predict_traj(m, k, r)
    Uk = U3[:, :, k]
    err_t .+= [relative_l2_error(Û[:, j], Uk[:, j]) for j in 1:nt] ./ ntest
    pk̂ = vec(maximum(Û; dims = 1)); pk = vec(maximum(Uk; dims = 1))
    push!(dTpk, maximum(abs.(pk̂ .- pk)))
    ε = mus[1, k]
    P̂r = [radiated_power(fom, FEFunction(fom.Un, Û[:, j]), ε) for j in 1:nt]
    Pr = [radiated_power(fom, FEFunction(fom.Un, Uk[:, j]), ε) for j in 1:nt]
    push!(dPq, maximum(abs.(P̂r .- Pr) ./ max.(abs.(Pr), 1.0)))
    rel = traj_rel(Û, Uk)
    rel > worst_ref[][2] && (worst_ref[] = (k, rel))
end
worst = worst_ref[][1]
@printf("\ndeployed r=%d:  mean traj rel L² %.3e\n", r, sweep_rel[findfirst(==(r), collect(RANKS_T))])
@printf("peak-T error: median %.2f K, max %.2f K   P_rad tracking: median %.2e, max %.2e\n",
        median(dTpk), maximum(dTpk), median(dPq), maximum(dPq))

jldsave(eval_t_path(); ranks = collect(RANKS_T), sweep_rel, floor_rel,
        err_t, dTpk, dPq, deployed_r = r)

# ----- figures ---------------------------------------------------------------------
εw, Qpw, dutyw, periodw = mus[:, worst]
Ûw = predict_traj(m, worst, r)
Uw = U3[:, :, worst]
tsw = collect(range(0.0, NCYCLES * periodw; length = nt)) ./ 3600
pk̂ = vec(maximum(Ûw; dims = 1)); pk = vec(maximum(Uw; dims = 1))
P̂r = [radiated_power(fom, FEFunction(fom.Un, Ûw[:, j]), εw) for j in 1:nt]
Pr = [radiated_power(fom, FEFunction(fom.Un, Uw[:, j]), εw) for j in 1:nt]

fig = Figure(size = (960, 620))
ax1 = Axis(fig[1, 1]; xlabel = "saved state index", ylabel = "mean rel. L²",
           yscale = log10, title = "Error vs time (test mean, r = $r)")
scatterlines!(ax1, 1:nt, max.(err_t, 1e-12))
vlines!(ax1, 1:NPHASE:nt; linestyle = :dot, color = (:gray, 0.5))
ax2 = Axis(fig[1, 2]; xlabel = "t (h)", ylabel = "peak T (K)",
           title = @sprintf("Worst test traj (ε=%.2f, Qp=%.0f, duty=%.2f, per=%.0fs)",
                            εw, Qpw, dutyw, periodw))
lines!(ax2, tsw, pk; label = "FOM", color = :black)
lines!(ax2, tsw, pk̂; label = "surrogate", color = :tomato, linestyle = :dash)
axislegend(ax2; position = :rt)
ax3 = Axis(fig[2, 1:2]; xlabel = "t (h)", ylabel = "P_rad (W/m)",
           title = "Radiated-power tracking through load switches (worst traj)")
lines!(ax3, tsw, Pr; label = "FOM", color = :black)
lines!(ax3, tsw, P̂r; label = "surrogate", color = :tomato, linestyle = :dash)
axislegend(ax3; position = :rt)
save(joinpath(FIG_DIR, "12_transient_eval.png"), fig)

println("saved ", eval_t_path(), " and figures/12_transient_eval.png")
