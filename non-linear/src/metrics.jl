# Error, residual and QoI metrics. The relative L² / coefficient errors port
# from the parent project; the PDE-residual metric is the *nonlinear* analogue:
# instead of ‖A(μ)û − b‖/‖b‖ we evaluate the assembled nonlinear residual of the
# reconstruction and normalise by the load vector, ‖R(û)‖₂ / ‖F_load‖₂.

"""Relative L² error `‖û − u‖ / ‖u‖` between two solution vectors."""
relative_l2_error(û::AbstractVector, u::AbstractVector) = norm(û .- u) / norm(u)

"""Mean relative L² error over the columns (samples) of two matrices."""
function mean_relative_l2(Û::AbstractMatrix, U::AbstractMatrix)
    return mean(relative_l2_error(view(Û, :, k), view(U, :, k)) for k in axes(U, 2))
end

"""Relative coefficient error `‖ĉ − c‖ / ‖c‖`."""
coefficient_error(ĉ::AbstractVecOrMat, c::AbstractVecOrMat) = norm(ĉ .- c) / norm(c)

"""Assembled load vector `F_v = ∫ v q dΩ` for the chip-patch source at power `Q`."""
function load_vector(fom::SteadyFOM, Q::Real)
    qf = CellField(source_fn(fom.cfg, Q), fom.Ω)
    return assemble_vector(v -> ∫(v * qf) * fom.dΩ, fom.V)
end

"""
    steady_residual_norms(fom, û; eps_r, Q) -> (; rnorm, fnorm, rel)

Nonlinear PDE residual of an arbitrary nodal field `û` (e.g. a surrogate
reconstruction), evaluated with the same Gridap assembly used by the solver and
normalised by the load vector: `rel = ‖R(û)‖₂ / ‖F_load‖₂`. Requires config B
(the nodal vector then *is* the free-DOF vector).
"""
function steady_residual_norms(fom::SteadyFOM, û::AbstractVector;
                               eps_r::Real, Q::Real)
    @assert fom.cfg.bc === :source "residual metric assumes config B DOF layout"
    res = steady_form(fom; eps_r, Q)
    op = FEOperator(res, fom.U, fom.V)
    aop = Gridap.FESpaces.get_algebraic_operator(op)
    r = Gridap.Algebra.residual(aop, collect(Float64, û))
    fnorm = norm(load_vector(fom, Q))
    return (; rnorm = norm(r), fnorm, rel = norm(r) / fnorm)
end

"""Mean relative nonlinear residual over predicted fields `Û` (n × M) at
parameters `μs` (vector of (ε, Q) tuples)."""
function mean_relative_residual(fom::SteadyFOM, μs, Û::AbstractMatrix)
    return mean(steady_residual_norms(fom, view(Û, :, k);
                                      eps_r = μs[k][1], Q = μs[k][2]).rel
                for k in eachindex(μs))
end

"""
    qoi_errors(fom, û, u; eps_r) -> (; dP_rel, dTpeak)

Design-tool QoI errors of a reconstruction against the truth: relative radiated
power error and absolute peak-temperature error (K).
"""
function qoi_errors(fom::SteadyFOM, û::AbstractVector, u::AbstractVector;
                    eps_r::Real)
    ûh = FEFunction(fom.Un, collect(Float64, û))
    uh = FEFunction(fom.Un, collect(Float64, u))
    P̂ = radiated_power(fom, ûh, eps_r)
    P = radiated_power(fom, uh, eps_r)
    return (; dP_rel = abs(P̂ - P) / abs(P), dTpeak = abs(maximum(û) - maximum(u)))
end
