# Proper Orthogonal Decomposition (POD) of a snapshot matrix via the SVD.
# Ported from the parent (linear) project unchanged in substance.
#
# On the uniform grid the L²-inner product is a constant times the Euclidean one,
# so the plain SVD of the snapshot matrix already gives the L²-optimal POD basis
# (mass-matrix-weighted POD stays future work). Note one honest difference from
# the parent project: with a radiation BC there is no "exact boundary condition
# by construction" — the BC is part of the residual, and QoI fidelity (radiated
# power, peak temperature) takes over that role.

"""
    PODBasis

Reduced basis obtained from the POD/SVD of a snapshot matrix.

- `mean`  — column mean `ū` (length `n`); subtracted before the SVD.
- `modes` — orthonormal POD modes `Φ` (`n × m`), columns = left singular vectors.
- `svals` — singular values `σ` (length `m`), non-increasing.
"""
struct PODBasis
    mean::Vector{Float64}
    modes::Matrix{Float64}
    svals::Vector{Float64}
end

"""
    fit_pod(S; center=true) -> PODBasis

Compute the POD basis of snapshot matrix `S` (`n × M`, one solution per column).
Subtracts the column mean (when `center`) and takes the thin SVD.
"""
function fit_pod(S::AbstractMatrix; center::Bool = true)
    μ̄ = center ? vec(mean(S; dims = 2)) : zeros(eltype(S), size(S, 1))
    Sc = center ? S .- μ̄ : S
    F = svd(Sc)
    return PODBasis(Vector{Float64}(μ̄), Matrix{Float64}(F.U), Vector{Float64}(F.S))
end

"""Number of available POD modes."""
nmodes(P::PODBasis) = length(P.svals)

"""View of the first `r` POD modes (an `n × r` matrix)."""
modes_r(P::PODBasis, r::Integer) = view(P.modes, :, 1:r)

"""
    project(P, U, r) -> C

Project snapshots onto the first `r` modes: `c = Φᵣᵀ (u − ū)`. Accepts a single
vector (returns a length-`r` vector) or a matrix (returns `r × M`).
"""
project(P::PODBasis, u::AbstractVector, r::Integer) = modes_r(P, r)' * (u .- P.mean)
project(P::PODBasis, U::AbstractMatrix, r::Integer) = modes_r(P, r)' * (U .- P.mean)

"""
    reconstruct(P, C, r) -> Û

Reconstruct fields from reduced coefficients: `û = ū + Φᵣ c`.
"""
reconstruct(P::PODBasis, c::AbstractVector, r::Integer) = modes_r(P, r) * c .+ P.mean
reconstruct(P::PODBasis, C::AbstractMatrix, r::Integer) = modes_r(P, r) * C .+ P.mean

"""
    reconstruction_errors(P, U, ranks) -> Vector

Mean relative L² error of projecting `U` onto `r` modes and reconstructing, for
each `r` in `ranks`. The "best case" floor any surrogate using this basis can reach.
"""
function reconstruction_errors(P::PODBasis, U::AbstractMatrix, ranks)
    return [mean_relative_l2(reconstruct(P, project(P, U, r), r), U) for r in ranks]
end

"""Cumulative captured-energy fraction vs rank: `Σ_{i≤r} σ_i² / Σ_i σ_i²`."""
function energy_fraction(P::PODBasis)
    e = cumsum(P.svals .^ 2)
    return e ./ e[end]
end
