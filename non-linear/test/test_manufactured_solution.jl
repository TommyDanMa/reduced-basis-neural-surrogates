# Manufactured solution exercising the *nonlinear* radiation boundary path:
# u* = 300 + 50 cos(πx)cos(2πy) has zero normal derivative on every edge, so the
# radiation BC picks up inhomogeneous data g = εσ(u*⁴ − T_s⁴) on Γ_r and the
# insulated edges stay natural. P1 elements must show L² order 2 — accepted
# band [1.7, 2.2], matching the parent project's convention.

@testset "manufactured solution (MMS)" begin
    eps_r = 0.85
    errs = Float64[]
    hs = Float64[]
    for (nx, ny) in ((20, 10), (40, 20), (80, 40), (160, 80))
        cfg = RadiativeConfig(; nx, ny)
        fom = build_steady_fom(cfg)
        s = solve_steady(fom; eps_r, Q = 0.0,
                         f = mms_bulk(cfg), g = mms_bc_data(cfg, eps_r),
                         u0 = fill(300.0, fom.n), abstol = 1e-9)
        @test s.converged
        push!(errs, l2_error(fom, s.uh, mms_solution))
        push!(hs, cfg.Lx / nx)
    end
    @test all(diff(errs) .< 0)                    # strictly decreasing
    rates = [log2(errs[i] / errs[i+1]) for i in 1:length(errs)-1]
    @info "MMS L² errors" errs rates
    @test 1.7 <= rates[end] <= 2.2
end
