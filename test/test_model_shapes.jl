# Surrogate model I/O contracts: shapes, determinism, parameter accounting.

using StableRNGs

@testset "model shapes" begin
    rng = StableRNG(123)
    M = 7
    X = randn(rng, 2, M)            # M parameter samples (μ ∈ ℝ²)

    @testset "POD-MLP maps μ → c" begin
        r = 10
        model, ps, st = build_pod_mlp(r; hidden = 32, depth = 3, rng = rng)
        C = predict(model, ps, st, X)
        @test size(C) == (r, M)
    end

    @testset "direct MLP maps μ → field" begin
        g = make_grid(12)
        model, ps, st = build_direct_mlp(ndof(g); hidden = 32, rng = rng)
        U = predict(model, ps, st, X)
        @test size(U) == (ndof(g), M)
    end

    @testset "inference is deterministic in eval mode" begin
        model, ps, st = build_pod_mlp(5; rng = rng)
        @test predict(model, ps, st, X) == predict(model, ps, st, X)
    end

    @testset "predicted coefficients reconstruct a full-size field" begin
        g = make_grid(16)
        S = reduce(hcat, [solve_parametric(g, [m1, m2]) for m1 in -1:0.5:1, m2 in -1:0.5:1])
        P = fit_pod(S)
        r = 6
        model, ps, st = build_pod_mlp(r; rng = rng)
        ĉ = predict(model, ps, st, [0.2, -0.4])
        û = reconstruct(P, ĉ, r)
        @test length(û) == ndof(g)
    end

    @testset "parameter counts: direct ≫ reduced" begin
        g = make_grid(20)
        direct, _, _ = build_direct_mlp(ndof(g); hidden = 64, rng = rng)
        reduced, _, _ = build_pod_mlp(10; hidden = 64, rng = rng)
        @test param_count(direct) > param_count(reduced)
        @test param_count(reduced) > 0
    end
end
