# Statistics hygiene: bootstrap CIs, paired comparisons, ablations, seed table.
# Everything here is analysis of cached artifacts plus two cheap retrains.

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(PROJECT_DIR, "src", "models_kan.jl"))

println("=== 20_statistics ===")
rng = StableRNG(SEED + 9000)
NBOOT = 10_000
qs(v) = (median(v), quantile(v, 0.025), quantile(v, 0.975))

# ----- bootstrap CI on Spearman (both operating points) ---------------------------
ds = load(joinpath(DATA_DIR, "design_study.jld2"))
rankvec(v) = invperm(sortperm(v))
function spearman(a, b)
    ra, rb = rankvec(a), rankvec(b)
    n = length(a)
    return 1 - 6 * sum(abs2, ra .- rb) / (n * (n^2 - 1))
end
sp_cis = []
for op in ds["ops"]
    f, s = vec(op[:fomQ]), vec(op[:surQ])
    n = length(f)
    boots = [begin
                 idx = rand(rng, 1:n, n)
                 spearman(f[idx], s[idx])
             end for _ in 1:NBOOT]
    m, lo, hi = qs(boots)
    push!(sp_cis, (; Qp = op[:Qp], period = op[:period], m, lo, hi))
    @printf("Spearman CI (Qp=%.0f, per=%.0f): median %.4f  [%.4f, %.4f]  (N=%d, B=%d)\n",
            op[:Qp], op[:period], m, lo, hi, n, NBOOT)
end

# ----- paired per-sample errors + paired bootstrap ---------------------------------
d = load_snapshots()
pod = load_pod()
P = pod.P
te = d["test_idx"]
Xte, Ute = d["X"][:, te], d["U"][:, te]
ntest = length(te)

m = load_mlp(model_path("pod_mlp_r$(DEFAULT_R)"))
kk = load_kan(model_path("pod_kan_r$(DEFAULT_R)"))
dd = load(model_path("direct_mlp"))
dm, _, _ = make_mlp(dd["indim"], dd["outdim"];
                    hidden = dd["hidden"], depth = dd["depth"], rng = StableRNG(SEED))

persample(Û) = [relative_l2_error(Û[:, i], Ute[:, i]) for i in 1:ntest]
e_mlp = persample(Matrix(reconstruct(P, m.cscales .* predict(m.model, m.ps, m.st, Xte), DEFAULT_R)))
e_kan = persample(Matrix(reconstruct(P, kk.cscales .* predict(kk.model, kk.ps, kk.st, Xte), DEFAULT_R)))
e_dir = persample(Matrix(dd["mean"] .+ dd["scale"] .* predict(dm, dd["ps"], dd["st"], Xte)))

function paired_ci(a, b)   # mean(a − b) with paired bootstrap
    diffs = a .- b
    boots = [mean(diffs[rand(rng, 1:ntest, ntest)]) for _ in 1:NBOOT]
    return qs(boots)
end
for (name, a, b) in (("KAN − MLP", e_kan, e_mlp), ("direct − MLP", e_dir, e_mlp))
    mid, lo, hi = paired_ci(a, b)
    @printf("paired Δ rel L² (%s): median %.2e  [%.2e, %.2e]  (N=%d paired samples)\n",
            name, mid, lo, hi, ntest)
end

# ----- ablations --------------------------------------------------------------------
tr = d["train_idx"]
Xtr, Utr = d["X"][:, tr], d["U"][:, tr]
r = DEFAULT_R
Ctr = Matrix(project(P, Utr, r))
cscales = vec(std(Ctr; dims = 2)) .+ 1e-14 * std(Ctr)

# (a) whitening off: same seed as the deployed model, raw-coefficient targets
model, ps, st = build_pod_mlp(r; rng = StableRNG(SEED + 100 + r))
ps, _ = train!(model, ps, st, Xtr, Ctr; epochs = 6000, verbose = false)
e_raw = mean_relative_l2(Matrix(reconstruct(P, predict(model, ps, st, Xte), r)), Ute)

# (b) random orthonormal basis control (same mean, same rank, same training recipe)
Φr = Matrix(qr(randn(StableRNG(31), size(Utr, 1), r)).Q)
Prand = PODBasis(P.mean, Φr, ones(r))
Crand = Matrix(project(Prand, Utr, r))
crs = vec(std(Crand; dims = 2)) .+ 1e-14 * std(Crand)
mr, pr, sr = build_pod_mlp(r; rng = StableRNG(SEED + 100 + r))
pr, _ = train!(mr, pr, sr, Xtr, Crand ./ crs; epochs = 6000, verbose = false)
e_rand = mean_relative_l2(Matrix(reconstruct(Prand, crs .* predict(mr, pr, sr, Xte), r)), Ute)
floor_rand = mean_relative_l2(Matrix(reconstruct(Prand, Matrix(project(Prand, Ute, r)), r)), Ute)

ev = load(eval_path())
@printf("\nablations at r=%d (deployed POD-MLP: %.2e):\n", r, ev["results"]["pod_mlp"][:rel])
@printf("  whitening OFF (same seed):      %.2e\n", e_raw)
@printf("  random orthonormal basis:       %.2e  (its projection floor: %.2e)\n",
        e_rand, floor_rand)
println("  rank ablation: see the §8.1 sweep table (cached in eval.jld2)")

# ----- seed table --------------------------------------------------------------------
seed_table = [
    ("global SEED", SEED),
    ("02 train/test split", "StableRNG(SEED)"),
    ("04/15 steady MLP rank r", "StableRNG(SEED + 100 + r)"),
    ("04/15 steady KAN rank r", "StableRNG(SEED + 200 + r)"),
    ("04 direct MLP", "StableRNG(SEED + 300)"),
    ("08 transient split", "StableRNG(SEED + 1)"),
    ("10 transient MLP rank r", "StableRNG(SEED + 400 + r)"),
    ("15 seed ensembles", "StableRNG(SEED + 5000/6000/7000 + s), s = 1..5"),
    ("15 OOD sampling", "StableRNG(SEED + 42)"),
    ("17 learning curve", "StableRNG(SEED + 8000 + i)"),
    ("20 bootstraps", "StableRNG(SEED + 9000)"),
    ("20 random basis", "StableRNG(31)"),
]
println("\nseed table (report appendix):")
for (k, v) in seed_table
    println("  ", rpad(k, 28), v)
end

jldsave(joinpath(DATA_DIR, "statistics.jld2");
        sp_cis = [Dict(pairs(x)) for x in sp_cis],
        e_mlp, e_kan, e_dir, e_raw, e_rand, floor_rand, nboot = NBOOT,
        seed_table = ["$(k): $(v)" for (k, v) in seed_table])
println("\nsaved data/statistics.jld2")
