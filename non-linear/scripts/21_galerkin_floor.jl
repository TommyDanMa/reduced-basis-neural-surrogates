# Exact reduced-Galerkin floor: the honest lower bound for ANY surrogate living in
# the POD subspace. At n = 1891 no hyper-reduction is needed offline: Newton on the
# r-dimensional projected system  g(c) = Φᵣᵀ R(ū + Φᵣ c) = 0  with Jacobian
# Φᵣᵀ J Φᵣ. DEIM (Chaturantabut & Sorensen 2010) is what would make this fast
# online; here it is a diagnostic, not a competitor. Output: the error-
# decomposition table  L²-projection floor → Galerkin floor → coefficient-map error.

include(joinpath(@__DIR__, "config.jl"))
using Gridap

println("=== 21_galerkin_floor ===")

fom = build_steady_fom(CFG)
d = load_snapshots()
pod = load_pod()
P = pod.P
te = d["test_idx"]
Ute = d["U"][:, te]
mus_te = [(d["mus"][1, k], d["mus"][2, k]) for k in te]
ntest = length(te)

"Solve the exact reduced-Galerkin system at rank r for parameters μ."
function galerkin_solve(r, μ)
    ε, Q = μ
    res = steady_form(fom; eps_r = ε, Q)
    aop = Gridap.FESpaces.get_algebraic_operator(FEOperator(res, fom.U, fom.V))
    Φ = Matrix(modes_r(P, r))
    ū = P.mean
    u0 = fill(equilibrium_estimate(fom.cfg, ε, Q), fom.n)
    c = Φ' * (u0 .- ū)
    J = Gridap.Algebra.allocate_jacobian(aop, u0)
    local x
    for it in 1:40
        x = ū .+ Φ * c
        g = Φ' * Gridap.Algebra.residual(aop, x)
        norm(g) < 1e-10 && return (x, true)
        Gridap.Algebra.jacobian!(J, aop, x)
        c -= (Φ' * (J * Φ)) \ g
    end
    x = ū .+ Φ * c
    return (x, norm(Φ' * Gridap.Algebra.residual(aop, x)) < 1e-8)
end

ranks = collect(RANKS)
gal_rel = Float64[]
gal_resid = Float64[]
for r in ranks
    rels = Float64[]
    resids = Float64[]
    nfail = 0
    for (i, μ) in enumerate(mus_te)
        x, ok = galerkin_solve(r, μ)
        ok || (nfail += 1; continue)
        push!(rels, relative_l2_error(x, Ute[:, i]))
        push!(resids, steady_residual_norms(fom, x; eps_r = μ[1], Q = μ[2]).rel)
    end
    push!(gal_rel, mean(rels))
    push!(gal_resid, mean(resids))
    @printf("r=%d: Galerkin rel L² %.3e   full residual %.3e   (%d/%d converged)\n",
            r, gal_rel[end], gal_resid[end], ntest - nfail, ntest)
end

# ----- the decomposition table -------------------------------------------------------
ev = load(eval_path())
sw = ev["sweep"]
proj_floor = [pod.recon_test[r] for r in ranks]
println("\nerror decomposition (mean over $(ntest) held-out parameters):")
@printf("%-5s %-16s %-16s %-16s %-14s\n", "rank",
        "L² proj floor", "Galerkin floor", "POD-MLP error", "map share")
for (i, r) in enumerate(ranks)
    map_err = sw["mlp_rel"][i]
    share = 1 - gal_rel[i] / map_err
    @printf("%-5d %-16.2e %-16.2e %-16.2e %.4f\n",
            r, proj_floor[i], gal_rel[i], map_err, share)
end

jldsave(joinpath(DATA_DIR, "galerkin_floor.jld2");
        ranks, proj_floor, gal_rel, gal_resid,
        mlp_rel = collect(sw["mlp_rel"]))
println("\nsaved data/galerkin_floor.jld2")
