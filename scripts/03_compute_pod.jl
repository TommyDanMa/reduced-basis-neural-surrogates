# 03 — Reduced basis by POD/SVD.
# Compute the POD basis of the training snapshots, report the singular-value decay
# and reconstruction error, and cache everything for the models and figures.

include("config.jl")

d = load_snapshots()
U_train, U_test = d["U_train"], d["U_test"]

P = fit_pod(U_train)
rank_curve = collect(1:min(50, nmodes(P)))
recon_train = reconstruction_errors(P, U_train, rank_curve)
recon_test  = reconstruction_errors(P, U_test, rank_curve)
ef = energy_fraction(P)

println("POD of $(size(U_train, 2)) snapshots  (ndof = $(size(U_train, 1)))")
println(rpad("rank", 6), rpad("σ_r/σ_1", 13), rpad("energy %", 11), "recon err (test)")
for r in RANKS
    println(rpad(r, 6), rpad(@sprintf("%.3e", P.svals[r] / P.svals[1]), 13),
            rpad(@sprintf("%.4f", 100 * ef[r]), 11), @sprintf("%.3e", recon_test[r]))
end

jldsave(pod_path(); mean = P.mean, modes = P.modes, svals = P.svals,
        ranks = collect(RANKS), rank_curve = rank_curve,
        recon_train = recon_train, recon_test = recon_test, energy = ef)
println("saved ", relpath(pod_path(), PROJECT_DIR))
