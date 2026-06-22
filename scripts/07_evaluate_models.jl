# 07 — Evaluate every trained surrogate on the held-out test set.
# Reports accuracy, physics consistency (PDE residual), size and speed, and caches
# the numbers for the figure script.

include("config.jl")

d = load_snapshots()
pod = load_pod()
g = make_grid(d["grid_n"])
Xte, Ute = d["mu_test"], d["U_test"]
μte = [collect(c) for c in eachcol(Xte)]
Mte = size(Xte, 2)

# Full-order solve time baseline (warm up once, then time).
solve_parametric(g, μte[1])
tfom = @elapsed for μ in μte; solve_parametric(g, μ); end
fom_ms = 1000 * tfom / Mte

# Load whatever models are present. POD-KAN needs its own loader + KolmogorovArnold,
# pulled in only if a trained KAN exists (so the rest never depends on it).
loaded = Tuple{String,NamedTuple}[]
isfile(model_path("direct"))  && push!(loaded, ("direct MLP", load_mlp(model_path("direct"))))
isfile(model_path("pod_mlp")) && push!(loaded, ("POD-MLP", load_mlp(model_path("pod_mlp"))))
if isfile(model_path("pod_kan"))
    include(joinpath(PROJECT_DIR, "src", "models_kan.jl"))
    push!(loaded, ("POD-KAN", load_kan(model_path("pod_kan"))))
end

names = String[]; rel_l2 = Float64[]; rel_res = Float64[]; coeff = Float64[]
params = Int[]; traint = Float64[]; pred_ms = Float64[]

for (label, m) in loaded
    r = m.kind == "direct_mlp" ? 0 : m.outdim
    # full inference path → (field, coefficients-or-nothing)
    function infer()
        if m.kind == "direct_mlp"
            return predict(m.model, m.ps, m.st, Xte), nothing
        else
            C = predict(m.model, m.ps, m.st, Xte)
            return reconstruct(pod.P, C, r), C
        end
    end
    infer()                                                            # warmup (compiles)
    local Uhat, Chat
    tp = @elapsed ((Uhat, Chat) = infer())
    ce = Chat === nothing ? NaN : coefficient_error(Chat, project(pod.P, Ute, r))
    push!(names, label)
    push!(rel_l2, mean_relative_l2(Uhat, Ute))
    push!(rel_res, mean(relative_residual(g, μte[k], view(Uhat, :, k)) for k in 1:Mte))
    push!(coeff, ce)
    push!(params, param_count(m.model))
    push!(traint, m.train_time)
    push!(pred_ms, 1000 * tp / Mte)
end

# ----- report -------------------------------------------------------------------
println("\nEvaluation on $Mte held-out parameters  (FOM solve: $(@sprintf("%.2f", fom_ms)) ms/sample)\n")
hdr = (rpad("model", 13), rpad("rel L²", 11), rpad("rel resid", 11), rpad("coef err", 11),
       rpad("params", 9), rpad("train s", 9), rpad("pred ms", 9), "speedup")
println(join(hdr))
for i in eachindex(names)
    spd = fom_ms / pred_ms[i]
    println(rpad(names[i], 13),
            rpad(@sprintf("%.3e", rel_l2[i]), 11),
            rpad(@sprintf("%.3e", rel_res[i]), 11),
            rpad(isnan(coeff[i]) ? "—" : @sprintf("%.3e", coeff[i]), 11),
            rpad(string(params[i]), 9),
            rpad(@sprintf("%.1f", traint[i]), 9),
            rpad(@sprintf("%.3f", pred_ms[i]), 9),
            @sprintf("%.0f×", spd))
end
@printf("\nPOD floor at r=%d (best possible): rel L² = %.3e\n", DEFAULT_R, pod.recon_test[DEFAULT_R])

jldsave(eval_path(); names, rel_l2, rel_res, coeff, params, train_time = traint,
        pred_ms, fom_ms, default_r = DEFAULT_R, pod_floor = pod.recon_test[DEFAULT_R])
println("saved ", relpath(eval_path(), PROJECT_DIR))
