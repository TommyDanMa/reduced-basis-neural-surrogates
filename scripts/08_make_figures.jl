# 08 — Render every figure for the report (CairoMakie).
# Run after 03 (POD) and 07 (evaluation); also reads a trained POD-MLP for the
# qualitative panels.

include("config.jl")
using CairoMakie
CairoMakie.activate!()
include(joinpath(PROJECT_DIR, "src", "plotting.jl"))

d   = load_snapshots()
pod = load_pod()
g   = make_grid(d["grid_n"])
xs  = full_coords(g)

μ = [0.7, -0.5]                                  # representative parameter
A  = [diffusion(x, y, μ) for x in xs, y in xs]
Utrue = embed_full(g, solve_parametric(g, μ))

save(joinpath(FIG_DIR, "02_coefficient_field.png"),
     field_figure(xs, xs, A; title = "diffusion a(x,y;μ),  μ = $(μ)"))
save(joinpath(FIG_DIR, "03_ground_truth.png"),
     field_figure(xs, xs, Utrue; title = "FOM solution u(x,y;μ)"))

# keystone + reduced-basis quality
save(joinpath(FIG_DIR, "04_sigma_decay.png"),
     sigma_decay_figure(pod.P.svals; ranks = Tuple(pod.ranks)))
save(joinpath(FIG_DIR, "05_recon_error_vs_rank.png"),
     recon_error_figure(pod.rank_curve, pod.recon_test))

# qualitative POD-MLP prediction + residual
mp = load_mlp(model_path("pod_mlp"))
r  = mp.outdim
uhat = reconstruct(pod.P, predict(mp.model, mp.ps, mp.st, μ), r)
Uhat = embed_full(g, uhat)
save(joinpath(FIG_DIR, "07_prediction_error.png"),
     comparison_triptych(xs, xs, Utrue, Uhat; title = "POD-MLP prediction (μ = $(μ), r = $r)"))

resfield = embed_full(g, residual(g, (x, y) -> diffusion(x, y, μ), forcing, uhat))
cr = maximum(abs, resfield)
save(joinpath(FIG_DIR, "08_residual_heatmap.png"),
     field_figure(xs, xs, resfield; title = "PDE residual  A û − b", colormap = :balance,
                  colorrange = (-cr, cr)))

# quantitative comparison from the evaluation cache
e = load(eval_path())
names = e["names"]
save(joinpath(FIG_DIR, "06_model_error.png"),
     bar_figure(names, e["rel_l2"]; ylabel = "mean relative L² error",
                title = "Surrogate accuracy (test set)", fmt = x -> @sprintf("%.1e", x)))
save(joinpath(FIG_DIR, "09_runtime.png"),
     bar_figure(vcat("FOM solve", names .* "\npredict"), vcat(e["fom_ms"], e["pred_ms"]);
                ylabel = "time per sample (ms)", title = "Runtime: solver vs surrogates",
                fmt = x -> @sprintf("%.2f", x)))
save(joinpath(FIG_DIR, "10_paramcount.png"),
     bar_figure(names, e["params"]; ylabel = "learnable parameters",
                title = "Model size", fmt = x -> string(Int(x)), color = :indianred))

# console screenshot stand-in
save(joinpath(FIG_DIR, "11_dashboard.png"),
     dashboard_figure(xs, xs, A, Utrue, Uhat, pod.P.svals; r = r, μ = μ))

# operator factorisation G = R ∘ N (parameters → coefficients → field)
mode_imgs = [embed_full(g, pod.P.modes[:, k]) for k in 1:5]
save(joinpath(FIG_DIR, "12_operator_diagram.png"),
     operator_diagram(mode_imgs, xs; r = r, N = ndof(g)))

println("wrote figures to ", relpath(FIG_DIR, PROJECT_DIR), "/")
foreach(f -> println("  ", f), sort(readdir(FIG_DIR)))
