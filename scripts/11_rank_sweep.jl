# 11 — Rank sweep: POD-MLP and POD-KAN accuracy/residual vs retained rank.
# (The POD-MLP vs +residual-loss ablation lives in scripts/10_residual_ablation.jl.)

include("config.jl")
using Lux, KolmogorovArnold
include(joinpath(PROJECT_DIR, "src", "models_kan.jl"))

d = load_snapshots()
pod = load_pod()
g = make_grid(d["grid_n"])
P = pod.P
Xtr, Utr = d["mu_train"], d["U_train"]
Xte, Ute = d["mu_test"], d["U_test"]
μte = [collect(c) for c in eachcol(Xte)]

function evaluate(model, ps, st, r)
    Û = reconstruct(P, predict(model, ps, st, Xte), r)
    l2 = mean_relative_l2(Û, Ute)
    res = mean(relative_residual(g, μte[k], view(Û, :, k)) for k in eachindex(μte))
    return l2, res
end

ranks = (3, 5, 10, 20)
mlp_l2 = Float64[]; mlp_res = Float64[]; kan_l2 = Float64[]; kan_res = Float64[]

for r in ranks
    Ctr = project(P, Utr, r)
    m, ps, st = build_pod_mlp(r; hidden = 64, depth = 3, rng = StableRNG(SEED + 100 + r))
    ps, _ = train!(m, ps, st, Xtr, Ctr; epochs = 6000, lr = 1e-3, verbose = false,
                   rng = StableRNG(SEED + 100 + r))
    a, b = evaluate(m, ps, st, r); push!(mlp_l2, a); push!(mlp_res, b)

    m, ps, st = build_pod_kan(r; width = 20, grid_size = 10, rng = StableRNG(SEED + 200 + r))
    ps, _ = train!(m, ps, st, Xtr, Ctr; epochs = 10000, lr = 1e-3, verbose = false,
                   rng = StableRNG(SEED + 200 + r))
    a, b = evaluate(m, ps, st, r); push!(kan_l2, a); push!(kan_res, b)
    @printf("rank %2d done\n", r)
end

# ---- report --------------------------------------------------------------------
println("\n| Rank | POD-MLP rel L² | POD-MLP residual | POD-KAN rel L² | POD-KAN residual |")
println("| ---: | -------------: | ---------------: | -------------: | ---------------: |")
for (i, r) in enumerate(ranks)
    @printf("| %4d | %14.3e | %16.3e | %14.3e | %16.3e |\n",
            r, mlp_l2[i], mlp_res[i], kan_l2[i], kan_res[i])
end

jldsave(joinpath(DATA_DIR, "sweep.jld2"); ranks = collect(ranks),
        mlp_l2, mlp_res, kan_l2, kan_res)

using CairoMakie
CairoMakie.activate!()
include(joinpath(PROJECT_DIR, "src", "plotting.jl"))
save(joinpath(FIG_DIR, "13_rank_sweep.png"),
     rank_sweep_figure(ranks, mlp_l2, kan_l2, mlp_res, kan_res))
println("\nsaved data/sweep.jld2 and figures/13_rank_sweep.png")
