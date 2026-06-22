# POD basis properties: orthonormality, error decay, and exact full-rank recovery.

@testset "POD reconstruction" begin
    g = make_grid(24)
    μs = [[m1, m2] for m1 in -1:0.4:1, m2 in -1:0.4:1]
    S = reduce(hcat, [solve_parametric(g, μ) for μ in μs])
    P = fit_pod(S)

    @testset "modes are orthonormal" begin
        r = 8
        Φ = P.modes[:, 1:r]
        @test norm(Φ' * Φ - I(r)) < 1e-10
    end

    @testset "reconstruction error decreases with rank" begin
        ranks = (1, 2, 4, 8, 16)
        errs = reconstruction_errors(P, S, ranks)
        @test all(errs[k] ≥ errs[k+1] - 1e-12 for k in 1:length(errs)-1)
        @test errs[end] < errs[1]
    end

    @testset "full-rank reconstruction recovers snapshots" begin
        r = min(size(S, 2), nmodes(P))
        Ŝ = reconstruct(P, project(P, S, r), r)
        @test mean_relative_l2(Ŝ, S) < 1e-9
    end

    @testset "singular values are sorted and non-negative" begin
        @test all(P.svals .≥ 0)
        @test issorted(P.svals; rev = true)
    end
end
