using RadiativeSurrogates
using Test
using Gridap
using LinearAlgebra
using StableRNGs
using NonlinearSolve

@testset "RadiativeSurrogates" verbose = true begin
    include("test_steady_solver.jl")
    include("test_jacobian.jl")
    include("test_energy_balance.jl")
    include("test_manufactured_solution.jl")
    include("test_analytic_equilibrium.jl")
    include("test_transient.jl")
    include("test_pod_models.jl")
    include("test_deployed_models.jl")
end
