# Structural properties of the assembled operator A(μ).

@testset "solver matrix" begin
    g = make_grid(10)
    A = assemble_matrix(g, (x, y) -> diffusion(x, y, [0.4, -0.6]))

    @testset "shape" begin
        @test size(A) == (g.n^2, g.n^2)
    end

    @testset "symmetry" begin
        @test A == permutedims(A)          # exactly symmetric by construction
        @test issymmetric(A)
    end

    @testset "positive definite when a > 0" begin
        @test isposdef(Symmetric(Matrix(A)))
    end

    @testset "constant a reproduces the 5-point Laplacian" begin
        gc = make_grid(8)
        Ac = assemble_matrix(gc, (x, y) -> 1.0)
        h2 = gc.h^2
        @test all(≈(4 / h2), diag(Ac))                       # every diagonal = 4/h²
        # a horizontal-neighbour coupling equals -1/h²
        k = lin(gc, 3, 3)
        @test Ac[k, lin(gc, 4, 3)] ≈ -1 / h2
        @test Ac[k, lin(gc, 3, 4)] ≈ -1 / h2
    end

    @testset "direct solution has tiny residual" begin
        b = assemble_rhs(g, forcing)
        u = solve_pde(g, (x, y) -> diffusion(x, y, [0.4, -0.6]), forcing)
        @test norm(A * u - b) / norm(b) < 1e-8
    end
end
