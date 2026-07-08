# Regression tripwire for the deployed surrogates. Skips cleanly when data/ is
# absent (fresh clone, CI); with artifacts present it pins the headline quality
# with 3× margin so silently degraded models can never ship.

@testset "deployed models (regression)" begin
    datadir = normpath(joinpath(@__DIR__, "..", "data"))
    needed = joinpath.(datadir, ["pod.jld2", "model_pod_mlp_r2.jld2",
                                 "pod_transient.jld2", "model_pod_mlp_t_r8.jld2"])
    if !all(isfile, needed)
        @info "deployed-model regression skipped (data/ artifacts not present)"
        @test true
    else
        include(joinpath(@__DIR__, "..", "scripts", "config.jl"))
        fom = build_steady_fom(CFG)
        pod = load_pod()
        m = load_mlp(model_path("pod_mlp_r$(DEFAULT_R)"))

        for (ε, Q) in ((0.3, 300.0), (0.6, 600.0), (0.85, 150.0),
                       (0.2, 700.0), (0.9, 450.0))
            s = solve_steady(fom; eps_r = ε, Q, abstol = 1e-9)
            @test s.converged
            x = [standardize(ε, EPS_RANGE...), standardize(Q, Q_RANGE...)]
            û = collect(reconstruct(pod.P,
                    m.cscales .* predict(m.model, m.ps, m.st, x), DEFAULT_R))
            @test relative_l2_error(û, nodal_values(fom, s.uh)) < 3e-3
            @test steady_residual_norms(fom, û; eps_r = ε, Q).rel < 3e-2
        end

        pod_t = load(pod_t_path())
        Pt = PODBasis(pod_t["mean"], pod_t["modes"], pod_t["svals"])
        mt = load_mlp(model_path("pod_mlp_t_r$(DEFAULT_R_T)"))
        μst = standardize_mu_t(0.5, 500.0, 0.5, 5400.0)
        nt = NCYCLES * NPHASE + 1
        X = reduce(hcat, [transient_features(μst, j) for j in 1:nt])
        Û = Matrix(reconstruct(Pt,
                mt.cscales .* predict(mt.model, mt.ps, mt.st, X), DEFAULT_R_T))
        @test all(isfinite, Û)
        @test all(200 .< Û .< 1100)
    end
end
