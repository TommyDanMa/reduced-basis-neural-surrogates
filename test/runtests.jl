using ReducedBasisSurrogates
using Test
using LinearAlgebra

@testset "ReducedBasisSurrogates.jl" begin
    include("test_solver_matrix.jl")
    include("test_boundary_conditions.jl")
    include("test_manufactured_solution.jl")
    include("test_pod_reconstruction.jl")
    include("test_model_shapes.jl")
end
