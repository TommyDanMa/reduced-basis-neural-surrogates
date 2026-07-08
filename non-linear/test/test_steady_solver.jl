# Solver plumbing: boundary-tag geometry, the linear (ε = 0) limit against a
# direct sparse solve, cross-algorithm agreement, and warm starting.

@testset "steady solver" begin
    cfg_a = RadiativeConfig(bc = :hotwall, k = 1.0, nx = 30, ny = 15)
    fom_a = build_steady_fom(cfg_a)          # constructor asserts Γ_r geometry

    # ε = 0, no source: pure conduction with a uniform Dirichlet wall → u ≡ T_hot.
    r0 = solve_steady(fom_a; eps_r = 0.0, Q = 0.0, u0 = fill(390.0, fom_a.n))
    @test r0.converged
    @test maximum(abs.(r0.x .- cfg_a.T_hot)) < 1e-8

    # ε = 0 with a source: the residual is affine, so one Newton step from any
    # state must match the direct sparse solve x = x0 − J⁻¹R(x0).
    ra = solve_steady(fom_a; eps_r = 0.0, Q = 300.0)
    x0 = fill(380.0, fom_a.n)
    J = Gridap.Algebra.allocate_jacobian(ra.aop, x0)
    Gridap.Algebra.jacobian!(J, ra.aop, x0)
    resv = Gridap.Algebra.residual(ra.aop, x0)
    x_direct = x0 .- J \ resv
    @test ra.converged
    @test norm(ra.x - x_direct) / norm(x_direct) < 1e-9

    # Config B nonlinear: two independent algorithms must agree.
    fom_b = build_steady_fom(RadiativeConfig(nx = 30, ny = 15))
    μ = (0.5, 450.0)
    s_newton = solve_steady(fom_b, μ)
    s_trust = solve_steady(fom_b, μ; alg = TrustRegion())
    @test s_newton.converged && s_trust.converged
    @test norm(s_newton.x - s_trust.x) / norm(s_newton.x) < 1e-7

    # Warm starting from a neighbouring parameter must not cost more steps.
    s_warm = solve_steady(fom_b, (0.52, 460.0); u0 = s_newton.x)
    @test s_warm.converged
    @test s_warm.nsteps <= s_newton.nsteps

    # k-override: interior linearity means (Q, k) enter the particular solution
    # only through Q/k, and the radiative level is k-free, so doubling both Q
    # and k must reproduce the doubled-load shape scaled consistently; here we
    # just pin that the override is wired: k = 2*cfg.k halves the conduction
    # temperature rise above the (k-free) edge equilibrium level.
    s_k1 = solve_steady(fom_b; eps_r = 0.5, Q = 450.0)
    s_k2 = solve_steady(fom_b; eps_r = 0.5, Q = 450.0, k = 2 * fom_b.cfg.k)
    rise1 = maximum(s_k1.x) - minimum(s_k1.x)
    rise2 = maximum(s_k2.x) - minimum(s_k2.x)
    @test s_k2.converged
    @test isapprox(rise2, rise1 / 2; rtol = 0.05)
end
