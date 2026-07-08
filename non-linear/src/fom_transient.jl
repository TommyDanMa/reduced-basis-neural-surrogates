# Transient full-order model. Semidiscrete (method-of-lines) form
#
#     M u̇ = −R_space(u) + Q(t)·F_unit,
#
# where M is the consistent ∫ρc u v mass matrix, R_space the steady spatial
# residual (conduction + radiation) at zero load, and the chip source enters
# linearly through the unit load vector. OrdinaryDiffEq's mass-matrix
# integrators do the stepping: ImplicitEuler/Trapezoid are the θ-family the
# scope doc asked for; FBDF is the adaptive stiff upgrade. Load switches are
# passed as `tstops` so square waves are hit exactly.

"Consistent mass matrix ∫ ρc u v dΩ (config B: unconstrained space)."
function mass_matrix(fom::SteadyFOM)
    ρc = fom.cfg.rho_c
    return assemble_matrix((du, v) -> ∫(ρc * (du * v)) * fom.dΩ, fom.U, fom.V)
end

"Square-wave load: `Q_peak` for the first `duty` fraction of each period."
square_wave(Q_peak::Real, period::Real, duty::Real) =
    t -> mod(t, period) < duty * period ? float(Q_peak) : 0.0

"Switch times of `square_wave` inside `(t0, t1)`, for use as `tstops`."
function load_switch_times(period::Real, duty::Real, t0::Real, t1::Real)
    ts = Float64[]
    for k in floor(Int, t0 / period):ceil(Int, t1 / period)
        for s in (k * period, (k + duty) * period)
            t0 < s < t1 && push!(ts, s)
        end
    end
    return sort(ts)
end

"Thermal energy ∫ ρc u dΩ of a nodal field (J per m depth)."
function thermal_energy(fom::SteadyFOM, uvec::AbstractVector)
    uh = FEFunction(fom.Un, collect(Float64, uvec))
    return sum(∫(fom.cfg.rho_c * uh) * fom.dΩ)
end

"""
    solve_transient(fom; eps_r, load, u0, tspan, alg = FBDF(), ...) -> ODESolution

Integrate the semidiscrete system. `load` is `t ↦ Q(t)` (W per m depth); `u0`
the initial nodal vector; pass `dt` (disables adaptivity) for fixed-step
θ-method runs, `tstops` for load discontinuities, `saveat` for the snapshot
grid. The Jacobian passed to the integrator is the Gridap-assembled −∂R/∂u.
"""
function solve_transient(fom::SteadyFOM; eps_r::Real, load, u0::AbstractVector,
                         tspan::Tuple, alg = nothing, saveat = Float64[],
                         dt = nothing, tstops = Float64[],
                         abstol::Real = 1e-6, reltol::Real = 1e-6)
    @assert fom.cfg.bc === :source "transient FOM assumes config B"
    res0 = steady_form(fom; eps_r, Q = 0.0)
    op = FEOperator(res0, fom.U, fom.V)
    aop = Gridap.FESpaces.get_algebraic_operator(op)
    Funit = load_vector(fom, 1.0)
    M = mass_matrix(fom)
    x0 = collect(Float64, u0)
    J0 = Gridap.Algebra.allocate_jacobian(aop, x0)

    rhs! = (du, u, p, t) -> begin
        Gridap.Algebra.residual!(du, aop, u)
        qt = load(t)
        @. du = -du + qt * Funit
        return nothing
    end
    jac! = (J, u, p, t) -> begin
        Gridap.Algebra.jacobian!(J, aop, u)
        rmul!(J, -1.0)
        return nothing
    end

    f = ODEFunction(rhs!; mass_matrix = M, jac = jac!, jac_prototype = J0)
    prob = ODEProblem(f, x0, tspan)
    algo = alg === nothing ? FBDF() : alg
    if dt === nothing
        return OrdinaryDiffEq.solve(prob, algo; saveat, tstops, abstol, reltol)
    end
    return OrdinaryDiffEq.solve(prob, algo; dt, adaptive = false,
                                saveat, tstops)
end
