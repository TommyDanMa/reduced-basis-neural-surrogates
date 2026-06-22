# 01 — Solver validation by the Method of Manufactured Solutions.
# Confirms the full-order solver is second-order accurate before it is used as the
# ground-truth oracle for the surrogates.

include("config.jl")
using CairoMakie
CairoMakie.activate!()
include(joinpath(PROJECT_DIR, "src", "plotting.jl"))

uex(x, y)   = sinpi(x) * sinpi(y)
acoef(x, y) = 1 + 0.25 * sinpi(x) * cospi(y)
ffun(x, y)  = mms_forcing(uex, acoef, x, y)

ns = (16, 32, 64, 128)
hs = Float64[]; errs = Float64[]

println("Method of Manufactured Solutions  (u* = sin πx sin πy,  a = 1 + ¼ sin πx cos πy)")
println(rpad("n", 6), rpad("h", 14), rpad("L² error", 16), "rate")
for n in ns
    g = make_grid(n)
    uh = solve_pde(g, acoef, ffun)
    ue = [uex(i * g.h, j * g.h) for j in 1:g.n for i in 1:g.n]
    e = l2_norm(g, uh .- ue)
    rate = isempty(errs) ? NaN : log(errs[end] / e) / log(hs[end] / g.h)
    println(rpad(n, 6), rpad(@sprintf("%.4e", g.h), 14), rpad(@sprintf("%.6e", e), 16),
            isnan(rate) ? "—" : @sprintf("%.3f", rate))
    push!(hs, g.h); push!(errs, e)
end

rates = [log(errs[k] / errs[k+1]) / log(hs[k] / hs[k+1]) for k in 1:length(errs)-1]
@printf("\nmean observed order: %.3f   (acceptance band [1.7, 2.2])\n", mean(rates))
@assert all(1.7 .≤ rates .≤ 2.2) "Convergence order outside the acceptance band!"
println("PASS: solver is second-order accurate.")

fig = convergence_figure(hs, errs)
save(joinpath(FIG_DIR, "01_convergence.png"), fig)
println("saved figures/01_convergence.png")
