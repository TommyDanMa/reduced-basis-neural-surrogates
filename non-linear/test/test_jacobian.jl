# "Jacobian under control": Gridap's AD Jacobian is checked against
# (a) central finite differences of the assembled residual, and
# (b) the hand-written analytic Jacobian  J = A_cond + B_rad(u), where
#     A_cond is the constant conduction matrix and B_rad carries 4εσ|u|³ on Γ_r.

@testset "jacobian" begin
    cfg = RadiativeConfig(nx = 24, ny = 12)
    fom = build_steady_fom(cfg)
    eps_r, Q = 0.6, 400.0

    res = steady_form(fom; eps_r, Q)
    op = FEOperator(res, fom.U, fom.V)
    aop = Gridap.FESpaces.get_algebraic_operator(op)

    rng = StableRNG(7)
    x = 320.0 .+ 40.0 .* (rand(rng, fom.n) .- 0.5)
    J = Gridap.Algebra.allocate_jacobian(aop, x)
    Gridap.Algebra.jacobian!(J, aop, x)

    # (a) AD vs central finite differences along random directions.
    δ = 0.03
    for _ in 1:3
        v = randn(rng, fom.n)
        v ./= norm(v)
        rp = Gridap.Algebra.residual(aop, x .+ δ .* v)
        rm = Gridap.Algebra.residual(aop, x .- δ .* v)
        fd = (rp .- rm) ./ (2δ)
        @test norm(J * v - fd) / norm(J * v) < 1e-6
    end

    # (b) AD vs hand-written Jacobian, assembled with the same measures.
    uh = FEFunction(fom.U, copy(x))
    k, σ = cfg.k, cfg.sigma
    a_cond(du, v) = ∫(k * (∇(v) ⋅ ∇(du))) * fom.dΩ
    b_rad(du, v) = ∫(v * du * ((4 * eps_r * σ) * (abs3 ∘ uh))) * fom.dΓ
    A = assemble_matrix(a_cond, fom.U, fom.V)
    B = assemble_matrix(b_rad, fom.U, fom.V)
    @test norm(J - A - B) < 1e-9 * norm(J)
end
