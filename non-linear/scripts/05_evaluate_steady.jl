# Held-out evaluation of the steady surrogates: relative L², coefficient error,
# nonlinear PDE residual ‖R(û)‖/‖F‖, and the design-tool QoIs (radiated power,
# peak temperature). Includes the rank sweep and the POD floor.

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(PROJECT_DIR, "src", "models_kan.jl"))
using Gridap
using CairoMakie
CairoMakie.activate!()

println("=== 05_evaluate_steady ===")

fom = build_steady_fom(CFG)
d = load_snapshots()
pod = load_pod()
P = pod.P
te = d["test_idx"]
Xte, Ute = d["X"][:, te], d["U"][:, te]
mus_te = [(d["mus"][1, k], d["mus"][2, k]) for k in te]
ntest = length(te)

"Evaluate a reconstructed field matrix against the truth."
function field_metrics(Û)
    rel = mean_relative_l2(Û, Ute)
    resid = mean_relative_residual(fom, mus_te, Û)
    dP = mean(qoi_errors(fom, Û[:, k], Ute[:, k]; eps_r = mus_te[k][1]).dP_rel
              for k in 1:ntest)
    dT = maximum(abs.(vec(maximum(Û; dims = 1)) .- vec(maximum(Ute; dims = 1))))
    return (; rel, resid, dP, dT)
end

# Physics floors: the FOM solutions themselves and the pure POD projection.
fom_resid = mean_relative_residual(fom, mus_te[1:10], Ute[:, 1:10])
proj = reconstruct(P, project(P, Ute, DEFAULT_R), DEFAULT_R)
floor_m = field_metrics(Matrix(proj))

results = Dict{String,Any}()
sweep = Dict("ranks" => collect(RANKS),
             "mlp_rel" => Float64[], "mlp_resid" => Float64[],
             "kan_rel" => Float64[], "kan_resid" => Float64[])

for r in RANKS
    m = load_mlp(model_path("pod_mlp_r$(r)"))
    Ĉ = m.cscales .* predict(m.model, m.ps, m.st, Xte)
    Û = Matrix(reconstruct(P, Ĉ, r))
    mm = field_metrics(Û)
    push!(sweep["mlp_rel"], mm.rel); push!(sweep["mlp_resid"], mm.resid)
    r == DEFAULT_R && (results["pod_mlp"] = (; mm...,
        coef = coefficient_error(Ĉ, Matrix(project(P, Ute, r))),
        params = param_count(m.model), train_time = m.train_time))

    kk = load_kan(model_path("pod_kan_r$(r)"))
    Ĉk = kk.cscales .* predict(kk.model, kk.ps, kk.st, Xte)
    Ûk = Matrix(reconstruct(P, Ĉk, r))
    mk = field_metrics(Ûk)
    push!(sweep["kan_rel"], mk.rel); push!(sweep["kan_resid"], mk.resid)
    r == DEFAULT_R && (results["pod_kan"] = (; mk...,
        coef = coefficient_error(Ĉk, Matrix(project(P, Ute, r))),
        params = param_count(kk.model), train_time = kk.train_time))
end

dd = load(model_path("direct_mlp"))
dm, _, _ = make_mlp(dd["indim"], dd["outdim"];
                    hidden = dd["hidden"], depth = dd["depth"], rng = StableRNG(SEED))
Ûd = dd["mean"] .+ dd["scale"] .* predict(dm, dd["ps"], dd["st"], Xte)
md_ = field_metrics(Matrix(Ûd))
results["direct_mlp"] = (; md_..., coef = NaN,
                         params = param_count(dm), train_time = dd["train_time"])

@printf("\nEvaluation on %d held-out parameters (FOM: median %.0f ms/solve, resid floor %.1e)\n",
        ntest, 1e3 * median(d["times"]), fom_resid)
@printf("POD floor at r=%d: rel L² = %.3e, resid = %.3e\n\n",
        DEFAULT_R, floor_m.rel, floor_m.resid)
@printf("%-12s %-10s %-10s %-10s %-9s %-9s %s\n",
        "model", "rel L²", "resid", "coef err", "ΔP/P", "ΔTpeak K", "params")
for name in ("direct_mlp", "pod_mlp", "pod_kan")
    r = results[name]
    @printf("%-12s %-10.3e %-10.3e %-10.3e %-9.2e %-9.3f %d\n",
            name, r.rel, r.resid, r.coef, r.dP, r.dT, r.params)
end
println("\nrank sweep (rel L² / resid):")
for (i, r) in enumerate(RANKS)
    @printf("  r=%2d  MLP %.3e / %.3e   KAN %.3e / %.3e\n", r,
            sweep["mlp_rel"][i], sweep["mlp_resid"][i],
            sweep["kan_rel"][i], sweep["kan_resid"][i])
end

jldsave(eval_path(); results = Dict(k => Dict(pairs(v)) for (k, v) in results),
        sweep, fom_resid, floor_rel = floor_m.rel, floor_resid = floor_m.resid,
        fom_ms = 1e3 * median(d["times"]))

# ----- figures -------------------------------------------------------------------
rc = collect(RANKS)
fig = Figure(size = (900, 380))
ax1 = Axis(fig[1, 1]; xscale = log2, yscale = log10, xlabel = "rank r",
           ylabel = "mean rel. L²", title = "Accuracy vs rank",
           xticks = (rc, string.(rc)))
scatterlines!(ax1, rc, sweep["mlp_rel"]; label = "POD-MLP")
scatterlines!(ax1, rc, sweep["kan_rel"]; label = "POD-KAN")
lines!(ax1, pod.ranks[rc], pod.recon_test[rc];
       linestyle = :dash, color = :gray, label = "POD floor")
axislegend(ax1; position = :rt)
ax2 = Axis(fig[1, 2]; xscale = log2, yscale = log10, xlabel = "rank r",
           ylabel = "mean rel. residual ‖R(û)‖/‖F‖",
           title = "Physics consistency vs rank", xticks = (rc, string.(rc)))
scatterlines!(ax2, rc, sweep["mlp_resid"]; label = "POD-MLP")
scatterlines!(ax2, rc, sweep["kan_resid"]; label = "POD-KAN")
hlines!(ax2, [fom_resid]; linestyle = :dot, color = :black, label = "FOM floor")
axislegend(ax2; position = :rb)
save(joinpath(FIG_DIR, "07_rank_sweep.png"), fig)

# worst-case test sample for the deployed POD-MLP
m = load_mlp(model_path("pod_mlp_r$(DEFAULT_R)"))
Ĉ = m.cscales .* predict(m.model, m.ps, m.st, Xte)
Û = Matrix(reconstruct(P, Ĉ, DEFAULT_R))
errs = [relative_l2_error(Û[:, k], Ute[:, k]) for k in 1:ntest]
kw = argmax(errs)
xs, ys = node_axes(fom)
res = steady_form(fom; eps_r = mus_te[kw][1], Q = mus_te[kw][2])
aop = Gridap.FESpaces.get_algebraic_operator(FEOperator(res, fom.U, fom.V))
rfield = abs.(Gridap.Algebra.residual(aop, Û[:, kw]))

fig = Figure(size = (1100, 320))
for (i, (M, ttl, cm)) in enumerate(
    ((to_grid(fom, Ute[:, kw]), "FOM truth (K)", :inferno),
     (to_grid(fom, Û[:, kw]), "POD-MLP r=$(DEFAULT_R) (K)", :inferno),
     (to_grid(fom, abs.(Û[:, kw] .- Ute[:, kw])), "|error| (K)", :viridis),
     (to_grid(fom, rfield), "|R(û)| nodal (W)", :viridis)))
    axp = Axis(fig[1, 2i - 1]; aspect = DataAspect(), title = ttl)
    hidedecorations!(axp)
    hmp = heatmap!(axp, xs, ys, M; colormap = cm)
    Colorbar(fig[1, 2i], hmp)
end
Label(fig[0, :], @sprintf("Worst test case: ε=%.2f, Q=%.0f W/m (rel L² %.2e)",
                          mus_te[kw][1], mus_te[kw][2], errs[kw]))
save(joinpath(FIG_DIR, "08_prediction_worstcase.png"), fig)

println("saved ", eval_path(), " and figures 07–08")
