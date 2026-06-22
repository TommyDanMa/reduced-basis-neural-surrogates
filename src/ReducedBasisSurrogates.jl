"""
    ReducedBasisSurrogates

Reduced-basis neural surrogates for a parametric elliptic PDE.

Pipeline: solve the full-order model `-∇·(a(μ)∇u) = f` → collect snapshots →
POD/SVD compression → learn the reduced coefficient map `μ ↦ c` with a small MLP
(or KAN) → reconstruct `û = ū + Φᵣc`. See `scripts/` for the end-to-end workflow
and `src/console.jl` for the interactive explorer.
"""
module ReducedBasisSurrogates

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using ForwardDiff
using Lux
using Optimisers
using Zygote

include("coefficients.jl")
include("pde_solver.jl")
include("metrics.jl")
include("pod.jl")
include("models.jl")

# grid + PDE data
export Grid, make_grid, ndof, lin, node_coord, interior_coords, full_coords,
       diffusion, forcing, mms_forcing
# full-order solver
export assemble_matrix, assemble_rhs, solve_pde, solve_parametric, reshape_interior, embed_full
# reduced basis
export PODBasis, fit_pod, nmodes, project, reconstruct, reconstruction_errors, energy_fraction
# metrics
export l2_norm, relative_l2_error, mean_relative_l2, residual, relative_residual,
       mean_relative_residual, coefficient_error
# models
export make_mlp, build_pod_mlp, build_direct_mlp, param_count, train!, predict, make_residual_loss

end # module
