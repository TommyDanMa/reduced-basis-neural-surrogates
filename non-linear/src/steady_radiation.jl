# steady_radiation.jl
# Steady-state nonlinear radiation problem (space-relevant)
# Left: fixed high temperature (electronics)
# Right: nonlinear radiation to deep space (T_space = 3 K)
# Top/Bottom: insulated

using Gridap
using LineSearches

# ====================== PARAMETERS (space-relevant) ======================
const Lx = 1.0          # length (m)
const Ly = 0.5          # height (m)
const T_hot = 400.0     # fixed temperature on left (K) - electronics
const T_space = 3.0     # deep space sink temperature (K)
const ε = 0.85          # emissivity (parametric later)
const σ = 5.670374417e-8
const k = 1.0           # thermal conductivity (W/m/K) - can be made parametric

# ====================== MESH & SPACES ======================
model = CartesianDiscreteModel((0.0, Lx, 0.0, Ly), (40, 20))

Ω  = Triangulation(model)
Γ_rad = BoundaryTriangulation(model, tags="right")   # radiating face
dΩ = Measure(Ω, 2)
dΓ = Measure(Γ_rad, 2)

reffe = ReferenceFE(lagrangian, Float64, 1)
V = TestFESpace(model, reffe, dirichlet_tags=["left"])
U = TrialFESpace(V, T_hot)

# ====================== WEAK FORM ======================
function residual(u, v)
    # Diffusion (conduction)
    diff = ∫( k * ∇(v) ⋅ ∇(u) )dΩ

    # Nonlinear radiation to space (central term)
    rad = ∫( v * ε * σ * (u^4 - T_space^4) )dΓ

    return diff + rad
end

# ====================== NONLINEAR OPERATOR ======================
op = NonlinearFEOperator(residual, U, V)

# ====================== ROBUST SOLVER ======================
nls = NLSolver(
    show_trace = true,
    method = :newton,
    linesearch = Backtracking(order=3, maxstep=0.95)
)

# ====================== SOLVE ======================
uh = solve(nls, op)

# ====================== POST-PROCESS & VALIDATION ======================
# Compute total radiated power (should balance conduction into the domain)
radiated_power = sum(∫(ε * σ * (uh^4 - T_space^4))dΓ)

println("\n=== Validation ===")
println("Total radiated power (W/m): ", radiated_power)

# Simple energy balance check (for this setup, conduction in ≈ radiation out)
# We can also write a manufactured solution later for stricter validation.

# ====================== OUTPUT ======================
writevtk(Ω, "radiation_result", cellfields=["u" => uh])

println("\nDone. Results written to radiation_result.vtu")