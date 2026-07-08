# Reduced-basis and mapper plumbing, ported from the parent project:
# orthonormal modes, exact full-rank round trip, monotone reconstruction
# floors, deterministic inference, and a trainability smoke test.

@testset "pod + models" begin
    rng = StableRNG(11)
    S = randn(rng, 50, 20)
    P = fit_pod(S)
    @test size(P.modes) == (50, 20)
    @test norm(P.modes' * P.modes - I) < 1e-10
    full = reconstruct(P, project(P, S, 20), 20)
    @test norm(full - S) / norm(S) < 1e-10
    errs = reconstruction_errors(P, S, (2, 5, 10, 20))
    @test all(diff(errs) .<= 1e-12)

    model, ps, st = build_pod_mlp(3; rng = StableRNG(1))
    X = rand(StableRNG(2), 2, 8) .* 2 .- 1
    Y1 = predict(model, ps, st, X)
    @test size(Y1) == (3, 8)
    @test Y1 == predict(model, ps, st, X)          # deterministic eval mode
    @test length(predict(model, ps, st, X[:, 1])) == 3

    C = vcat(sin.(pi .* X[1:1, :]), X[2:2, :] .^ 2, X[1:1, :] .* X[2:2, :])
    _, losses = train!(model, ps, st, X, C; epochs = 200, verbose = false)
    @test losses[end] < losses[1]
end
