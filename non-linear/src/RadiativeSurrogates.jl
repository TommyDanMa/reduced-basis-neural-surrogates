"""
    RadiativeSurrogates

Reduced-basis neural surrogates for the heat equation with a nonlinear
Stefan–Boltzmann radiation boundary condition (deep-space sink, T_space = 3 K).

Division of labour: Gridap assembles FE residuals/Jacobians; the SciML stack
solves them (NonlinearSolve for steady states, OrdinaryDiffEq for transients).
See `scripts/` for the end-to-end workflow.
"""
module RadiativeSurrogates

using LinearAlgebra
using SparseArrays
using Statistics
using Random
using Printf
using Gridap
using NonlinearSolve
using OrdinaryDiffEq
using OrdinaryDiffEqSDIRK: ImplicitEuler, Trapezoid   # v7 umbrella no longer exports the θ-family
using Lux
using Optimisers
using Zygote

include("physics.jl")
include("fom_steady.jl")
include("fom_transient.jl")
include("postprocess.jl")
include("pod.jl")
include("metrics.jl")
include("models.jl")

# physics
export RadiativeConfig, rad4, drad4, abs3, source_fn, equilibrium_estimate,
       mms_solution, mms_bulk, mms_bc_data
# steady FOM
export SteadyFOM, build_steady_fom, steady_form, solve_steady
# transient FOM
export mass_matrix, square_wave, load_switch_times, thermal_energy,
       solve_transient, ImplicitEuler, Trapezoid, FBDF
# postprocessing / QoIs
export radiated_power, source_power, energy_balance, dirichlet_influx,
       nodal_values, to_grid, node_axes, peak_temperature, edge_profile, l2_error
# reduced basis
export PODBasis, fit_pod, nmodes, modes_r, project, reconstruct,
       reconstruction_errors, energy_fraction
# metrics
export relative_l2_error, mean_relative_l2, coefficient_error, load_vector,
       steady_residual_norms, mean_relative_residual, qoi_errors
# models
export make_mlp, build_pod_mlp, build_direct_mlp, param_count, train!, predict

end # module
