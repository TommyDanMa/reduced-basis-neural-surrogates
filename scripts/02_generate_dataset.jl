# 02 — Generate the snapshot dataset.
# Sample parameters μ with a Sobol sequence, solve the full-order model for each,
# and store a train/test split.

include("config.jl")
using QuasiMonteCarlo

g = make_grid(GRID_N)
rng = StableRNG(SEED)

μmat = QuasiMonteCarlo.sample(N_SAMPLES, [PARAM_LO, PARAM_LO], [PARAM_HI, PARAM_HI],
                              SobolSample())              # 2 × N_SAMPLES
μs = [collect(c) for c in eachcol(μmat)]

println("Solving $N_SAMPLES full-order problems on a $GRID_N×$GRID_N grid (ndof = $(ndof(g)))…")
local U
t = @elapsed (U = reduce(hcat, [solve_parametric(g, μ) for μ in μs]))
@printf("  done in %.2f s  (%.2f ms / solve)\n", t, 1000t / N_SAMPLES)

ntest = round(Int, TEST_FRAC * N_SAMPLES)
perm = randperm(rng, N_SAMPLES)
test_idx, train_idx = perm[1:ntest], perm[ntest+1:end]

mu_train = reduce(hcat, μs[train_idx])
mu_test  = reduce(hcat, μs[test_idx])
U_train, U_test = U[:, train_idx], U[:, test_idx]

jldsave(snapshots_path(); grid_n = GRID_N, mu_train, mu_test, U_train, U_test, solve_time = t)
@printf("saved %s  (train = %d, test = %d)\n",
        relpath(snapshots_path(), PROJECT_DIR), size(mu_train, 2), size(mu_test, 2))
