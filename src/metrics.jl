# Error, residual and accounting metrics shared by the evaluation script,
# the tests and the interactive console.
#
# Note on norms: on a uniform grid the discrete L² norm equals `h · ‖·‖₂` in 2-D,
# so the constant cancels in every *relative* quantity and a plain Euclidean norm
# ratio already gives the relative L² error.

"""Discrete L² norm on the grid (2-D): `h · ‖v‖₂`."""
l2_norm(g::Grid, v::AbstractVector) = g.h * norm(v)

"""Relative L² error `‖û − u‖ / ‖u‖` between two solution vectors."""
relative_l2_error(û::AbstractVector, u::AbstractVector) = norm(û .- u) / norm(u)

"""Mean relative L² error over the columns (samples) of two matrices."""
function mean_relative_l2(Û::AbstractMatrix, U::AbstractMatrix)
    return mean(relative_l2_error(view(Û, :, k), view(U, :, k)) for k in axes(U, 2))
end

"""
    residual(g, afun, ffun, û) -> Vector

Discrete PDE residual `A·û − b` for an arbitrary (e.g. surrogate-predicted) field.
A small residual means the field nearly satisfies the discretized PDE.
"""
function residual(g::Grid, afun, ffun, û::AbstractVector)
    A = assemble_matrix(g, afun)
    b = assemble_rhs(g, ffun)
    return A * û .- b
end

"""
    relative_residual(g, μ, û; ffun=forcing) -> Float64

Relative PDE residual `‖A(μ)·û − b‖ / ‖b‖` using the default [`diffusion`](@ref).
This is the core physics-consistency metric reported for every model.
"""
function relative_residual(g::Grid, μ, û::AbstractVector; ffun = forcing)
    A = assemble_matrix(g, (x, y) -> diffusion(x, y, μ))
    b = assemble_rhs(g, ffun)
    return norm(A * û .- b) / norm(b)
end

"""Mean relative PDE residual over a set of parameters `μs` (vector of vectors)
and matching predicted fields `Û` (n × M)."""
function mean_relative_residual(g::Grid, μs, Û::AbstractMatrix; ffun = forcing)
    return mean(relative_residual(g, μs[k], view(Û, :, k); ffun) for k in eachindex(μs))
end

"""Relative coefficient error `‖ĉ − c‖ / ‖c‖`."""
coefficient_error(ĉ::AbstractVecOrMat, c::AbstractVecOrMat) = norm(ĉ .- c) / norm(c)
