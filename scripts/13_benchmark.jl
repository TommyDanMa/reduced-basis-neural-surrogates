# 13 — Rigorous timing with BenchmarkTools: single-query latency vs batched
# throughput, with allocations. The earlier "speed-up" was batched throughput;
# this separates it from per-query latency so neither number is oversold.

include("config.jl")
using BenchmarkTools
using Lux, KolmogorovArnold
include(joinpath(PROJECT_DIR, "src", "models_kan.jl"))

d = load_snapshots()
pod = load_pod()
g = make_grid(d["grid_n"])
P = pod.P
Xte = d["mu_test"]
Mte = size(Xte, 2)
x1 = Xte[:, 1]
μ1 = collect(x1)

direct = load_mlp(model_path("direct"))
mlp = load_mlp(model_path("pod_mlp"))
kan = load_kan(model_path("pod_kan"))

# inference paths (single query and batched), reconstruction included for POD models
fom_single(μ) = solve_parametric(g, μ)
field_single(m, x) = predict(m.model, m.ps, m.st, x)
field_batch(m, X) = predict(m.model, m.ps, m.st, X)
pod_single(m, x) = reconstruct(P, predict(m.model, m.ps, m.st, x), m.outdim)
pod_batch(m, X) = reconstruct(P, predict(m.model, m.ps, m.st, X), m.outdim)

# warmups (compile)
fom_single(μ1); field_single(direct, x1); field_batch(direct, Xte)
pod_single(mlp, x1); pod_batch(mlp, Xte); pod_single(kan, x1); pod_batch(kan, Xte)

ms(b)   = median(b).time / 1e6
kib(b)  = median(b).memory / 1024
naloc(b) = median(b).allocs

b_fom   = @benchmark fom_single($μ1)
b_dir_s = @benchmark field_single($direct, $x1)
b_dir_b = @benchmark field_batch($direct, $Xte)
b_mlp_s = @benchmark pod_single($mlp, $x1)
b_mlp_b = @benchmark pod_batch($mlp, $Xte)
b_kan_s = @benchmark pod_single($kan, $x1)
b_kan_b = @benchmark pod_batch($kan, $Xte)

fom_ms = ms(b_fom)
rows = [("FOM solve", ms(b_fom),  NaN,             naloc(b_fom),  kib(b_fom)),
        ("direct MLP", ms(b_dir_s), ms(b_dir_b)/Mte, naloc(b_dir_s), kib(b_dir_s)),
        ("POD-MLP",    ms(b_mlp_s), ms(b_mlp_b)/Mte, naloc(b_mlp_s), kib(b_mlp_s)),
        ("POD-KAN",    ms(b_kan_s), ms(b_kan_b)/Mte, naloc(b_kan_s), kib(b_kan_s))]

println("\nBenchmark - median of BenchmarkTools samples; batch size = $Mte.")
println("Speed-ups = batched throughput over FOM solves, incl. POD reconstruction,")
println("excl. plotting / model loading, after warmup.\n")
println(rpad("method", 12), rpad("single (ms)", 13), rpad("batched (ms/q)", 16),
        rpad("single×", 9), rpad("batched×", 10), rpad("allocs", 9), "mem (KiB)")
for (name, s, bch, al, mem) in rows
    sx  = name == "FOM solve" ? "1" : @sprintf("%.0f", fom_ms / s)
    bx  = isnan(bch) ? "—" : @sprintf("%.0f", fom_ms / bch)
    println(rpad(name, 12), rpad(@sprintf("%.3f", s), 13),
            rpad(isnan(bch) ? "—" : @sprintf("%.4f", bch), 16),
            rpad(sx, 9), rpad(bx, 10), rpad(string(Int(al)), 9), @sprintf("%.1f", mem))
end

jldsave(joinpath(DATA_DIR, "benchmark.jld2"); batch_size = Mte, fom_ms,
        names = [r[1] for r in rows], single_ms = [r[2] for r in rows],
        batched_ms = [r[3] for r in rows], allocs = [r[4] for r in rows])
println("\nsaved data/benchmark.jld2")
