# 12 — Is POD the "right" basis? Compare POD against a sine (Laplacian
# eigenfunction) basis and a random orthonormal basis, at equal rank.
# All three share the same data mean ū, so the comparison isolates basis quality.

include("config.jl")
using CairoMakie
CairoMakie.activate!()
include(joinpath(PROJECT_DIR, "src", "plotting.jl"))

d = load_snapshots()
pod = load_pod()
g = make_grid(d["grid_n"])
P = pod.P
Xtr, Utr = d["mu_train"], d["U_train"]
Xte, Ute = d["mu_test"], d["U_test"]
n = ndof(g)
maxr = 30

# --- sine (Laplacian eigenfunction) basis: φ_{ij} ∝ sin(iπx)sin(jπy) -------------
xs = interior_coords(g)
pairs = sort(vec([(i, j) for i in 1:maxr, j in 1:maxr]); by = p -> p[1]^2 + p[2]^2)[1:maxr]
sine_cols = [vec([sinpi(i * x) * sinpi(j * y) for x in xs, y in xs]) for (i, j) in pairs]
B_sine = reduce(hcat, [c ./ norm(c) for c in sine_cols])           # orthonormal on the grid

# --- random orthonormal basis ----------------------------------------------------
B_rand = Matrix(qr(randn(StableRNG(SEED), n, maxr)).Q)[:, 1:maxr]

# --- reconstruction error vs rank (test snapshots, common mean ū) ----------------
function recon_err_curve(B, U, ū, ranks)
    Uc = U .- ū
    nrm = [norm(view(U, :, k)) for k in axes(U, 2)]
    return [mean(let R = Uc - view(B, :, 1:r) * (view(B, :, 1:r)' * Uc)
                     norm(view(R, :, k)) / nrm[k]
                 end for k in axes(U, 2)) for r in ranks]
end

ranks = collect(1:maxr)
pod_errs  = recon_err_curve(P.modes, Ute, P.mean, ranks)
sine_errs = recon_err_curve(B_sine, Ute, P.mean, ranks)
rand_errs = recon_err_curve(B_rand, Ute, P.mean, ranks)

# --- surrogate accuracy at r = DEFAULT_R for each basis ---------------------------
function surrogate_l2(B)
    r = DEFAULT_R
    basis = PODBasis(P.mean, Matrix(B[:, 1:r]), ones(r))
    Ctr = project(basis, Utr, r)
    m, ps, st = build_pod_mlp(r; rng = StableRNG(SEED + 400))
    ps, _ = train!(m, ps, st, Xtr, Ctr; epochs = 5000, lr = 1e-3, verbose = false,
                   rng = StableRNG(SEED + 400))
    return mean_relative_l2(reconstruct(basis, predict(m, ps, st, Xte), r), Ute)
end
pod_surr, sine_surr, rand_surr = surrogate_l2(P.modes), surrogate_l2(B_sine), surrogate_l2(B_rand)

# --- report ---------------------------------------------------------------------
ri = findfirst(==(DEFAULT_R), ranks)
println("\nReconstruction error at r = $DEFAULT_R (test):")
@printf("  POD = %.3e   sine = %.3e   random = %.3e\n", pod_errs[ri], sine_errs[ri], rand_errs[ri])
println("\n| Basis (r=$DEFAULT_R)        | recon error | surrogate rel L² |")
println("| --------------------- | ----------: | ---------------: |")
@printf("| POD (data-optimal)    | %.3e | %.3e |\n", pod_errs[ri], pod_surr)
@printf("| sine (Laplacian)      | %.3e | %.3e |\n", sine_errs[ri], sine_surr)
@printf("| random orthonormal    | %.3e | %.3e |\n", rand_errs[ri], rand_surr)

jldsave(joinpath(DATA_DIR, "basis.jld2"); ranks, pod_errs, sine_errs, rand_errs,
        default_r = DEFAULT_R, pod_surr, sine_surr, rand_surr)
save(joinpath(FIG_DIR, "14_basis_comparison.png"),
     basis_comparison_figure(ranks, pod_errs, sine_errs, rand_errs))
println("\nsaved data/basis.jld2 and figures/14_basis_comparison.png")
