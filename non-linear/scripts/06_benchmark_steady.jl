# Rigorous timing with BenchmarkTools, separating single-query latency from
# batched throughput. FOM = full Newton solve (Gridap assembly + NonlinearSolve)
# from the equilibrium init at abstol 1e-9 — the same solve that built the
# dataset. Speed-ups are batched throughput over FOM solves, including POD
# reconstruction (and unscaling for the direct model), excluding plotting and
# model loading, after warm-up.

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(PROJECT_DIR, "src", "models_kan.jl"))
using BenchmarkTools

println("=== 06_benchmark_steady ===")

fom = build_steady_fom(CFG)
d = load_snapshots()
pod = load_pod()
P = pod.P
te = d["test_idx"]
Xte = d["X"][:, te]
x1 = Xte[:, 1]
μ1 = (d["mus"][1, te[1]], d["mus"][2, te[1]])
B = size(Xte, 2)

m = load_mlp(model_path("pod_mlp_r$(DEFAULT_R)"))
kk = load_kan(model_path("pod_kan_r$(DEFAULT_R)"))
dd = load(model_path("direct_mlp"))
dm, _, _ = make_mlp(dd["indim"], dd["outdim"];
                    hidden = dd["hidden"], depth = dd["depth"], rng = StableRNG(SEED))

pod_mlp_one(x) = reconstruct(P, m.cscales .* predict(m.model, m.ps, m.st, x), DEFAULT_R)
pod_mlp_batch(X) = reconstruct(P, m.cscales .* predict(m.model, m.ps, m.st, X), DEFAULT_R)
pod_kan_one(x) = reconstruct(P, kk.cscales .* predict(kk.model, kk.ps, kk.st, x), DEFAULT_R)
pod_kan_batch(X) = reconstruct(P, kk.cscales .* predict(kk.model, kk.ps, kk.st, X), DEFAULT_R)
direct_one(x) = dd["mean"] .+ dd["scale"] .* predict(dm, dd["ps"], dd["st"], x)
direct_batch(X) = dd["mean"] .+ dd["scale"] .* predict(dm, dd["ps"], dd["st"], X)

# warm-up every timed path
solve_steady(fom, μ1); pod_mlp_one(x1); pod_kan_one(x1); direct_one(x1)
pod_mlp_batch(Xte); pod_kan_batch(Xte); direct_batch(Xte)

bf = @benchmark solve_steady($fom, $μ1) samples = 20 seconds = 20
t_fom = median(bf).time / 1e6          # ms

rows = []
for (name, fone, fbatch) in (("direct MLP", direct_one, direct_batch),
                             ("POD-MLP", pod_mlp_one, pod_mlp_batch),
                             ("POD-KAN", pod_kan_one, pod_kan_batch))
    b1 = @benchmark $fone($x1) samples = 200 seconds = 10
    bb = @benchmark $fbatch($Xte) samples = 200 seconds = 10
    t1 = median(b1).time / 1e6                    # ms
    tb = median(bb).time / 1e6 / B                # ms per query
    push!(rows, (; name, t1, tb, allocs = b1.allocs,
                 mem = b1.memory / 1024, s1 = t_fom / t1, sb = t_fom / tb))
end

@printf("\nBenchmark — median of BenchmarkTools samples; batch size = %d.\n", B)
println("Speed-ups = batched throughput over FOM Newton solves, incl. POD")
println("reconstruction (and unscaling for direct), excl. plotting / model")
println("loading, after warmup.\n")
@printf("%-11s %-12s %-15s %-8s %-9s %-8s %s\n",
        "method", "single (ms)", "batched (ms/q)", "single×", "batched×",
        "allocs", "mem (KiB)")
@printf("%-11s %-12.3f %-15s %-8d %-9s %-8d %.1f\n",
        "FOM solve", t_fom, "—", 1, "—", bf.allocs, bf.memory / 1024)
for r in rows
    @printf("%-11s %-12.4f %-15.5f %-8.0f %-9.0f %-8d %.1f\n",
            r.name, r.t1, r.tb, r.s1, r.sb, r.allocs, r.mem)
end

jldsave(joinpath(DATA_DIR, "benchmark.jld2");
        t_fom_ms = t_fom, batch = B,
        names = [r.name for r in rows], t1 = [r.t1 for r in rows],
        tb = [r.tb for r in rows], s1 = [r.s1 for r in rows],
        sb = [r.sb for r in rows], allocs = [Int(r.allocs) for r in rows])
println("\nsaved data/benchmark.jld2")
