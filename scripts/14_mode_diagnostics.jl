# 14 — Mode-wise diagnostics that explain the non-monotonic rank result.
# Train a POD-MLP with many outputs and look, per mode i, at the coefficient signal
# |c_i|, the prediction error |ĉ_i - c_i|, their ratio, and the residual leverage
# ‖Aφ_i‖. High modes have both high relative error and high leverage.

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

RMAX = 20
Ctr = project(P, Utr, RMAX)
Cte = project(P, Ute, RMAX)
m, ps, st = build_pod_mlp(RMAX; hidden = 64, depth = 3, rng = StableRNG(SEED + 500))
ps, _ = train!(m, ps, st, Xtr, Ctr; epochs = 6000, lr = 1e-3, verbose = false,
               rng = StableRNG(SEED + 500))
Ĉ = predict(m, ps, st, Xte)

ε = 1e-12
sig    = [mean(abs, view(Cte, i, :)) for i in 1:RMAX]              # |c_i|
err    = [mean(abs, view(Ĉ, i, :) .- view(Cte, i, :)) for i in 1:RMAX]  # |ĉ_i - c_i|
relerr = err ./ (sig .+ ε)                                        # signal-to-noise

# residual leverage ‖Aφ_i‖ at a representative central parameter (a ≈ 1), normalised
A0 = assemble_matrix(g, (x, y) -> diffusion(x, y, [0.0, 0.0]))
lev = [norm(A0 * view(P.modes, :, i)) for i in 1:RMAX]
lev ./= lev[1]

println("mode |   |c_i|     |Δc_i|    rel.err   ‖Aφ_i‖/‖Aφ_1‖")
for i in 1:RMAX
    @printf("%4d | %.3e %.3e %.3e %.3e\n", i, sig[i], err[i], relerr[i], lev[i])
end

jldsave(joinpath(DATA_DIR, "modes.jld2"); idx = collect(1:RMAX), sig, err, relerr, leverage = lev)
save(joinpath(FIG_DIR, "15_mode_diagnostics.png"),
     mode_diagnostics_figure(1:RMAX, sig, err, relerr, lev))
println("saved data/modes.jld2 and figures/15_mode_diagnostics.png")
