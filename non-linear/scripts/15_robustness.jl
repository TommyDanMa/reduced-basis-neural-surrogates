# Robustness of the headline numbers.
#   Part 1 — seed ensembles: retrain the deployed mappers from 5 fresh seeds and
#   report the spread (the deployed models keep their rank-keyed seeds; the
#   ensemble is reported, not deployed).
#   Part 2 — out-of-distribution probe: evaluate the deployed steady surrogate on
#   a ring outside the training box, so the "in-distribution only" caveat becomes
#   a measured statement.

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(PROJECT_DIR, "src", "models_kan.jl"))
using CairoMakie
CairoMakie.activate!()

println("=== 15_robustness ===")

NSEEDS = 5

# ----- part 1a: steady seed ensembles ----------------------------------------------
d = load_snapshots()
pod = load_pod()
P = pod.P
tr, te = d["train_idx"], d["test_idx"]
Xtr, Xte = d["X"][:, tr], d["X"][:, te]
Utr, Ute = d["U"][:, tr], d["U"][:, te]

r = DEFAULT_R
Ctr = Matrix(project(P, Utr, r))
cscales = vec(std(Ctr; dims = 2)) .+ 1e-14 * std(Ctr)
C̃tr = Ctr ./ cscales

steady_rel(Ĉ) = mean_relative_l2(Matrix(reconstruct(P, Ĉ, r)), Ute)

mlp_rels = Float64[]
kan_rels = Float64[]
for s in 1:NSEEDS
    model, ps, st = build_pod_mlp(r; rng = StableRNG(SEED + 5000 + s))
    ps, _ = train!(model, ps, st, Xtr, C̃tr; epochs = 6000, verbose = false)
    push!(mlp_rels, steady_rel(cscales .* predict(model, ps, st, Xte)))

    kan, kps, kst = build_pod_kan(r; width = 20, grid_size = 10,
                                  rng = StableRNG(SEED + 6000 + s))
    kps, _ = train!(kan, kps, kst, Xtr, C̃tr; epochs = 10000, verbose = false)
    push!(kan_rels, steady_rel(cscales .* predict(kan, kps, kst, Xte)))
    @printf("  steady seed %d/%d: MLP %.3e  KAN %.3e\n",
            s, NSEEDS, mlp_rels[end], kan_rels[end])
end

# ----- part 1b: transient seed ensemble --------------------------------------------
dt_ = load(snapshots_t_path())
pod_t = load(pod_t_path())
Pt = PODBasis(pod_t["mean"], pod_t["modes"], pod_t["svals"])
U3, mus_t = dt_["U"], dt_["mus"]
n, nt, _ = size(U3)
ttr, tte = dt_["train_idx"], dt_["test_idx"]

Xt = reduce(hcat, [transient_features(standardize_mu_t(mus_t[:, k]...), j)
                   for k in ttr for j in 1:nt])
Ut = reshape(U3[:, :, ttr], n, :)
rt = DEFAULT_R_T
Ct = Matrix(project(Pt, Ut, rt))
st_ = vec(std(Ct; dims = 2))
cst = max.(st_, 0.01 * st_[1])
C̃t = Ct ./ cst

function transient_rel(model, ps, st)
    rels = Float64[]
    for k in tte
        Xk = reduce(hcat, [transient_features(standardize_mu_t(mus_t[:, k]...), j)
                           for j in 1:nt])
        Û = Matrix(reconstruct(Pt, cst .* predict(model, ps, st, Xk), rt))
        push!(rels, norm(Û - U3[:, :, k]) / norm(U3[:, :, k]))
    end
    return mean(rels)
end

t_rels = Float64[]
for s in 1:NSEEDS
    model, ps, st = build_pod_mlp(rt; indim = size(Xt, 1), hidden = 128,
                                  rng = StableRNG(SEED + 7000 + s))
    ps, _ = train!(model, ps, st, Xt, C̃t; epochs = 8000, verbose = false)
    push!(t_rels, transient_rel(model, ps, st))
    @printf("  transient seed %d/%d: %.3e\n", s, NSEEDS, t_rels[end])
end

ev = load(eval_path())
dep_mlp = ev["results"]["pod_mlp"][:rel]
dep_kan = ev["results"]["pod_kan"][:rel]
evt = load(eval_t_path())
dep_t = evt["sweep_rel"][findfirst(==(rt), evt["ranks"])]

@printf("\nseed spreads (deployed value first):\n")
@printf("  POD-MLP  r=%d: %.3e   ensemble [%.3e, %.3e]\n",
        r, dep_mlp, minimum(mlp_rels), maximum(mlp_rels))
@printf("  POD-KAN  r=%d: %.3e   ensemble [%.3e, %.3e]\n",
        r, dep_kan, minimum(kan_rels), maximum(kan_rels))
@printf("  POD-MLP-t r=%d: %.3e   ensemble [%.3e, %.3e]\n",
        rt, dep_t, minimum(t_rels), maximum(t_rels))

# ----- part 2: out-of-distribution probe -------------------------------------------
fom = build_steady_fom(CFG)
m = load_mlp(model_path("pod_mlp_r$(r)"))
B1 = ((0.02, 1.0), (50.0, 1000.0))            # physically sensible outer box

"Normalized exceedance outside the training box (0 on the boundary/inside)."
function box_dist(eps_r, Q)
    dε = max(EPS_RANGE[1] - eps_r, eps_r - EPS_RANGE[2], 0.0) /
         (EPS_RANGE[2] - EPS_RANGE[1])
    dQ = max(Q_RANGE[1] - Q, Q - Q_RANGE[2], 0.0) / (Q_RANGE[2] - Q_RANGE[1])
    return max(dε, dQ)
end

rng = StableRNG(SEED + 42)
ood_pts = Tuple{Float64,Float64}[]
while length(ood_pts) < 60
    ε = B1[1][1] + rand(rng) * (B1[1][2] - B1[1][1])
    Q = B1[2][1] + rand(rng) * (B1[2][2] - B1[2][1])
    in_training_box(ε, Q) || push!(ood_pts, (ε, Q))
end

function surrogate_rel_at(ε, Q)
    s = solve_steady(fom; eps_r = ε, Q, abstol = 1e-9)
    @assert s.converged
    x = [standardize(ε, EPS_RANGE...), standardize(Q, Q_RANGE...)]
    û = reconstruct(P, m.cscales .* predict(m.model, m.ps, m.st, x), r)
    return relative_l2_error(collect(û), nodal_values(fom, s.uh))
end

ood_d = [box_dist(p...) for p in ood_pts]
ood_rel = [surrogate_rel_at(p...) for p in ood_pts]
in_rel = [surrogate_rel_at(d["mus"][1, k], d["mus"][2, k]) for k in te[1:20]]
@printf("\nOOD probe (60 points, exceedance up to %.2f):\n", maximum(ood_d))
@printf("  in-box reference: median %.2e\n", median(in_rel))
@printf("  OOD: median %.2e, worst %.2e (at exceedance %.2f)\n",
        median(ood_rel), maximum(ood_rel), ood_d[argmax(ood_rel)])

jldsave(joinpath(DATA_DIR, "robustness.jld2");
        mlp_rels, kan_rels, t_rels, dep_mlp, dep_kan, dep_t,
        ood_d, ood_rel, in_rel)

# ----- figure ----------------------------------------------------------------------
fig = Figure(size = (900, 380))
ax1 = Axis(fig[1, 1]; yscale = log10, ylabel = "test rel. L²",
           xticks = (1:3, ["POD-MLP r=$r", "POD-KAN r=$r", "POD-MLP-t r=$rt"]),
           title = "Seed robustness (5 fresh seeds + deployed)")
for (i, (rels, dep)) in enumerate(((mlp_rels, dep_mlp), (kan_rels, dep_kan),
                                   (t_rels, dep_t)))
    scatter!(ax1, fill(i, length(rels)), rels; color = (:steelblue, 0.7))
    scatter!(ax1, [i], [dep]; color = :tomato, marker = :star5, markersize = 16)
end
ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = "normalized exceedance outside μ-box",
           ylabel = "rel. L²", title = "Out-of-distribution degradation (steady)")
scatter!(ax2, zeros(length(in_rel)), in_rel; color = (:gray, 0.6), label = "in-box")
scatter!(ax2, ood_d, ood_rel; color = :tomato, label = "outside")
axislegend(ax2; position = :lt)
save(joinpath(FIG_DIR, "16_robustness.png"), fig)

println("saved data/robustness.jld2 and figures/16_robustness.png")
