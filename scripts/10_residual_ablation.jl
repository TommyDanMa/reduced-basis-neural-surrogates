# 10 — (P1, optional) Physics-informed residual-loss ablation.
# Trains the POD-MLP with the data loss plus a λ_res-weighted PDE-residual term and
# reports how accuracy and the residual metric trade off. Uses a training subset so
# it runs quickly. The residual term is *optional*; the main models use data loss.

include("config.jl")

d = load_snapshots()
pod = load_pod()
g = make_grid(d["grid_n"])
Xte, Ute = d["mu_test"], d["U_test"]
μte = [collect(c) for c in eachcol(Xte)]
r = 10          # fixed: this ablation is reported at r=10, independent of DEFAULT_R

nsub = 150
Xs = d["mu_train"][:, 1:nsub]
Cs = project(pod.P, d["U_train"][:, 1:nsub], r)
μs = [collect(c) for c in eachcol(Xs)]
resloss = make_residual_loss(g, pod.P, μs, r)

function train_variant(λ; epochs = 2500)
    rng = StableRNG(SEED + 7)
    model, ps, st = build_pod_mlp(r; hidden = 64, depth = 3, rng)
    extra = λ > 0 ? resloss : nothing
    ps, _ = train!(model, ps, st, Xs, Cs; epochs, lr = 1e-3, extra, λ, verbose = false)
    Û = reconstruct(pod.P, predict(model, ps, st, Xte), r)
    l2 = mean_relative_l2(Û, Ute)
    res = mean(relative_residual(g, μte[k], view(Û, :, k)) for k in eachindex(μte))
    return l2, res
end

println("Residual-regularised loss ablation  (POD-MLP, r=$r, $nsub training samples)\n")
println(rpad("λ_res", 10), rpad("rel L² error", 16), "rel PDE residual")
for λ in (0.0, 1e-3, 1e-2, 1e-1)
    l2, res = train_variant(λ)
    println(rpad(@sprintf("%.0e", λ), 10), rpad(@sprintf("%.3e", l2), 16), @sprintf("%.3e", res))
end
println("\n(Data-only λ=0 is the default; the residual term trades a little L² accuracy")
println(" for better physics consistency — it is an optional extension, not core.)")
