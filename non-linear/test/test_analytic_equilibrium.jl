# Closed-form check: uniform bulk load q0 with insulated top/bottom makes the
# problem 1D in x:  u(x) = u₁ + q0(Lx² − x²)/(2k),  u₁ = (q0·Lx/(εσ) + T_s⁴)^¼.
# This pins the radiative equilibrium level *and* the conductive profile.

@testset "analytic radiative equilibrium" begin
    cfg = RadiativeConfig()
    fom = build_steady_fom(cfg)
    q0, eps_r = 500.0, 0.6
    u1 = (q0 * cfg.Lx / (eps_r * cfg.sigma) + cfg.T_space^4)^(1 / 4)
    u_exact(x) = u1 + q0 * (cfg.Lx^2 - x[1]^2) / (2 * cfg.k)

    s = solve_steady(fom; eps_r, Q = 0.0, f = x -> q0,
                     u0 = fill(u1, fom.n), abstol = 1e-9)
    @test s.converged

    A = to_grid(fom, nodal_values(fom, s.uh))
    xs, _ = node_axes(fom)

    # y-invariance of the discrete solution (data and BCs are y-independent).
    @test maximum(abs.(A .- A[:, 1])) < 1e-6

    # Nodal profile matches the exact parabola to O(h²).
    prof_err = maximum(abs.(A[:, 1] .- [u_exact((x,)) for x in xs]))
    @test prof_err < 0.02
    # Radiating-edge temperature hits the analytic equilibrium level.
    @test abs(A[end, 1] - u1) < 0.02
end
