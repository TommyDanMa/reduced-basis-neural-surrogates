# Homogeneous Dirichlet BC enforcement and qualitative Poisson behaviour.

@testset "boundary conditions" begin
    g = make_grid(21)                      # odd ⇒ a unique centre node
    u = solve_parametric(g, [0.3, 0.5])

    @testset "embedded solution vanishes on the boundary" begin
        U = embed_full(g, u)
        @test all(U[1, :] .== 0) && all(U[end, :] .== 0)
        @test all(U[:, 1] .== 0) && all(U[:, end] .== 0)
    end

    @testset "POD reconstruction also satisfies the BC exactly" begin
        S = reduce(hcat, [solve_parametric(g, [m1, m2]) for m1 in -1:0.5:1, m2 in -1:0.5:1])
        P = fit_pod(S)
        û = reconstruct(P, project(P, S[:, 1], 5), 5)
        Û = embed_full(g, û)
        @test all(Û[1, :] .== 0) && all(Û[end, :] .== 0)
        @test all(Û[:, 1] .== 0) && all(Û[:, end] .== 0)
    end

    @testset "constant a, constant f ⇒ symmetric positive bump" begin
        u0 = solve_pde(g, (x, y) -> 1.0, (x, y) -> 1.0)
        U0 = reshape_interior(g, u0)
        @test all(>(0), U0)                                  # interior strictly positive
        @test U0 ≈ reverse(U0; dims = 1)                     # symmetric in x
        @test U0 ≈ reverse(U0; dims = 2)                     # symmetric in y
        c = (g.n + 1) ÷ 2
        @test argmax(U0) == CartesianIndex(c, c)             # maximum at the centre
    end
end
