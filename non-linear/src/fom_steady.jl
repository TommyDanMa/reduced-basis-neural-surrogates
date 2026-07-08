# Full-order steady model. Division of labour: Gridap assembles the residual and
# Jacobian of the nonlinear FE system; NonlinearSolve.jl drives the Newton
# iteration (sparse Jacobian prototype, warm starts across parameter sweeps).

struct SteadyFOM
    cfg::RadiativeConfig
    model::Any
    Ω::Any
    dΩ::Any          # bulk measure
    dΩe::Any         # higher-degree measure for error integrals
    Γr::Any          # radiating boundary ("right")
    dΓ::Any
    V::Any           # test space (Dirichlet on "left" iff cfg.bc == :hotwall)
    U::Any           # trial space
    Vn::Any          # unconstrained nodal space (postprocessing / snapshots)
    Un::Any
    n::Int           # free DOFs of U (solver state dimension)
    n_nodes::Int     # DOFs of Un = (nx+1)(ny+1) (snapshot state dimension)
    gi::Vector{Int}  # nodal DOF → grid index i (x-direction)
    gj::Vector{Int}  # nodal DOF → grid index j (y-direction)
end

function build_steady_fom(cfg::RadiativeConfig)
    model = CartesianDiscreteModel((0.0, cfg.Lx, 0.0, cfg.Ly), (cfg.nx, cfg.ny))

    # 2D Cartesian face entities: corners 1:(0,0) 2:(Lx,0) 3:(0,Ly) 4:(Lx,Ly);
    # edges 5:bottom 6:top 7:left 8:right. Verified below by measuring Γ_r.
    labels = get_face_labeling(model)
    add_tag_from_tags!(labels, "left", [1, 3, 7])
    add_tag_from_tags!(labels, "right", [2, 4, 8])

    Ω = Triangulation(model)
    dΩ = Measure(Ω, cfg.quad_bulk)
    dΩe = Measure(Ω, 2 * cfg.quad_bulk)
    Γr = BoundaryTriangulation(model; tags = "right")
    dΓ = Measure(Γr, cfg.quad_bnd)

    reffe = ReferenceFE(lagrangian, Float64, cfg.order)
    if cfg.bc === :hotwall
        V = TestFESpace(model, reffe; conformity = :H1, dirichlet_tags = ["left"])
        U = TrialFESpace(V, cfg.T_hot)
    else
        V = TestFESpace(model, reffe; conformity = :H1)
        U = TrialFESpace(V)
    end
    Vn = TestFESpace(model, reffe; conformity = :H1)
    Un = TrialFESpace(Vn)

    # Fail fast if the face-tag convention ever changes: Γ_r must be the x = Lx
    # edge, of length Ly.
    len = sum(∫(CellField(1.0, Γr)) * dΓ)
    xmean = sum(∫(CellField(x -> x[1], Γr)) * dΓ) / len
    @assert isapprox(len, cfg.Ly; rtol = 1e-10) "Γ_r length $(len) ≠ Ly"
    @assert isapprox(xmean, cfg.Lx; rtol = 1e-10) "Γ_r is not the x = Lx edge"

    # Nodal DOF → structured-grid index map, recovered by interpolating the
    # coordinate functions (robust to whatever DOF ordering Gridap uses).
    # P1 only: higher orders place DOFs off the vertex grid, so grid extraction
    # (`to_grid`, `edge_profile`) is unavailable there; POD on raw DOF vectors
    # still works.
    if cfg.order == 1
        xv = get_free_dof_values(interpolate_everywhere(x -> x[1], Un))
        yv = get_free_dof_values(interpolate_everywhere(x -> x[2], Un))
        hx, hy = cfg.Lx / cfg.nx, cfg.Ly / cfg.ny
        gi = round.(Int, xv ./ hx) .+ 1
        gj = round.(Int, yv ./ hy) .+ 1
        @assert maximum(abs.(xv .- (gi .- 1) .* hx)) < 1e-9 "nodal x-coords off grid"
        @assert maximum(abs.(yv .- (gj .- 1) .* hy)) < 1e-9 "nodal y-coords off grid"
    else
        gi = Int[]
        gj = Int[]
    end

    n = num_free_dofs(U)
    n_nodes = num_free_dofs(Un)
    return SteadyFOM(cfg, model, Ω, dΩ, dΩe, Γr, dΓ, V, U, Vn, Un,
                     n, n_nodes, gi, gj)
end

"""
    steady_form(fom; eps_r, Q = 0, f = nothing, g = nothing)

Weak residual `(u, v) ↦ R(u; v)`:

    ∫ k ∇v·∇u dΩ − ∫ v (q_patch + f) dΩ + ∮ v εσ(u|u|³ − T_s⁴) dΓ_r − ∮ v g dΓ_r

`Q` scales the chip-patch source; `f` (bulk) and `g` (radiating-edge data) are
analytic hooks for manufactured solutions.
"""
function steady_form(fom::SteadyFOM; eps_r::Real, Q::Real = 0.0,
                     f = nothing, g = nothing, k = nothing)
    cfg = fom.cfg
    k = k === nothing ? cfg.k : Float64(k)   # per-solve conductivity override
    σ, Ts4 = cfg.sigma, cfg.T_space^4
    qpatch = source_fn(cfg, Q)
    fq = f === nothing ? qpatch : (x -> qpatch(x) + f(x))
    gb = g === nothing ? (x -> 0.0) : g
    dΩ, dΓ = fom.dΩ, fom.dΓ
    return (u, v) ->
        ∫(k * (∇(v) ⋅ ∇(u))) * dΩ - ∫(v * fq) * dΩ +
        ∫(v * ((eps_r * σ) * ((rad4 ∘ u) - Ts4))) * dΓ - ∫(v * gb) * dΓ
end

"""
    solve_steady(fom; eps_r, Q = 0, ...) -> (; uh, x, hist, retcode, nsteps, ...)

Assemble the FE operator with Gridap (Jacobian via Gridap's AD) and solve with
NonlinearSolve. `u0` warm-starts Newton (vector of free DOFs); `hist` records
‖R‖₂ at every residual evaluation. `alg = nothing` picks `NewtonRaphson()`.
"""
function solve_steady(fom::SteadyFOM; eps_r::Real, Q::Real = 0.0,
                      f = nothing, g = nothing, k = nothing, u0 = nothing,
                      alg = nothing, abstol::Real = 1e-8, maxiters::Int = 50)
    res = steady_form(fom; eps_r, Q, f, g, k)
    op = FEOperator(res, fom.U, fom.V)
    aop = Gridap.FESpaces.get_algebraic_operator(op)

    x0 = u0 === nothing ?
         fill(equilibrium_estimate(fom.cfg, eps_r, Q), fom.n) :
         collect(Float64, u0)

    J0 = Gridap.Algebra.allocate_jacobian(aop, x0)
    Gridap.Algebra.jacobian!(J0, aop, x0)

    hist = Float64[]
    f! = (r, x, p) -> begin
        Gridap.Algebra.residual!(r, aop, x)
        push!(hist, norm(r))
        return nothing
    end
    j! = (J, x, p) -> (Gridap.Algebra.jacobian!(J, aop, x); nothing)

    nlf = NonlinearFunction(f!; jac = j!, jac_prototype = J0)
    prob = NonlinearProblem(nlf, x0)
    algo = alg === nothing ? NewtonRaphson() : alg
    sol = NonlinearSolve.solve(prob, algo; abstol = Float64(abstol), maxiters)

    uh = FEFunction(fom.U, sol.u)
    return (; uh, x = sol.u, hist, retcode = sol.retcode,
            converged = NonlinearSolve.SciMLBase.successful_retcode(sol),
            nsteps = sol.stats.nsteps, aop)
end

"Convenience: μ = (ε, Q)."
solve_steady(fom::SteadyFOM, μ; kw...) =
    solve_steady(fom; eps_r = μ[1], Q = μ[2], kw...)
