# Method of Manufactured Solutions: proves the full-order solver is second order,
# so the ML models learn from a trustworthy oracle.

@testset "manufactured solution convergence" begin
    uex(x, y)   = sinpi(x) * sinpi(y)
    acoef(x, y) = 1 + 0.25 * sinpi(x) * cospi(y)
    ffun(x, y)  = mms_forcing(uex, acoef, x, y)

    ns = (16, 32, 64, 128)
    errs = Float64[]; hs = Float64[]
    for n in ns
        g = make_grid(n)
        uh = solve_pde(g, acoef, ffun)
        ue = [uex(i * g.h, j * g.h) for j in 1:g.n for i in 1:g.n]
        push!(errs, l2_norm(g, uh .- ue))
        push!(hs, g.h)
    end

    rates = [log(errs[k] / errs[k+1]) / log(hs[k] / hs[k+1]) for k in 1:length(errs)-1]

    @testset "error decreases monotonically" begin
        @test all(errs[k] > errs[k+1] for k in 1:length(errs)-1)
    end
    @testset "observed order ∈ [1.7, 2.2]" begin
        for r in rates
            @test 1.7 ≤ r ≤ 2.2
        end
    end
    @testset "finest grid is accurate" begin
        @test errs[end] < 1e-3
    end
end
