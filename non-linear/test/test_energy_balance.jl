# Global energy conservation. Config B (no Dirichlet DOFs) satisfies the
# discrete balance P_in = P_rad exactly at solver tolerance because v ≡ 1 is in
# the test space. Config A recovers the wall influx from the discrete reaction.

@testset "energy balance" begin
    fom = build_steady_fom(RadiativeConfig())

    # Patch source integrates to exactly Q (patch edges align with cell faces).
    @test abs(source_power(fom, 640.0) - 640.0) / 640.0 < 1e-12

    prev = nothing
    for (eps_r, Q) in ((0.1, 100.0), (0.1, 800.0), (0.95, 100.0),
                       (0.95, 800.0), (0.5, 450.0))
        s = solve_steady(fom; eps_r, Q, u0 = prev, abstol = 1e-9)
        @test s.converged
        bal = energy_balance(fom, s.uh; eps_r, Q)
        @test bal.rel < 1e-8
        prev = s.x
    end

    # Config A: reaction-based wall influx equals the radiated power.
    fom_a = build_steady_fom(RadiativeConfig(bc = :hotwall, k = 1.0))
    sa = solve_steady(fom_a; eps_r = 0.85, Q = 0.0, abstol = 1e-9)
    @test sa.converged
    P_in = dirichlet_influx(fom_a, sa.uh; eps_r = 0.85)
    P_rad = radiated_power(fom_a, sa.uh, 0.85)
    @test P_rad > 0
    @test abs(P_in - P_rad) / P_rad < 1e-8
end
