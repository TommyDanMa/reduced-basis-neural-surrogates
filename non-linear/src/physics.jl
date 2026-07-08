# Problem definition: geometry, materials, sources, and the nonlinear radiation
# weak form. Two boundary configurations:
#   :hotwall — config A: fixed T_hot on "left" (Dirichlet), radiation on "right".
#              Matches the original draft; used as a validation stepping stone.
#   :source  — config B (canonical): internal chip-patch source, all edges
#              insulated except the radiating "right" edge. No Dirichlet DOFs,
#              so v ≡ 1 is in the test space and the discrete energy balance
#              P_source = P_radiated holds exactly at solver tolerance.

Base.@kwdef struct RadiativeConfig
    Lx::Float64      = 1.0                     # m
    Ly::Float64      = 0.5                     # m
    nx::Int          = 60
    ny::Int          = 30
    order::Int       = 1                       # P1 Lagrangian
    k::Float64       = 5.0                     # W/(m K), fixed effective panel value
    rho_c::Float64   = 1.0e5                   # J/(m^3 K), effective (transient)
    T_space::Float64 = 3.0                     # K, deep-space sink
    sigma::Float64   = 5.670374417e-8          # W/(m^2 K^4), Stefan–Boltzmann
    patch::NTuple{4,Float64} = (0.05, 0.20, 0.15, 0.35)  # chip (x1,x2,y1,y2), config B
    T_hot::Float64   = 400.0                   # K, config A Dirichlet value
    bc::Symbol       = :source                 # :source | :hotwall
    quad_bulk::Int   = 4
    quad_bnd::Int    = 6                       # v·u⁴ with P1 u is degree 5 on facets
end

# Radiation nonlinearity, written as u|u|³ so it is monotone on all of ℝ
# (equals u⁴ for u ≥ 0). Newton iterates that dip negative still see a
# restoring flux of the correct sign.
rad4(u) = u * abs(u)^3
drad4(u) = 4 * abs(u)^3          # d(rad4)/du, used by the hand-written Jacobian check
abs3(u) = abs(u)^3

"Uniform volumetric source of total power `Q` (W per m depth) on the chip patch."
function source_fn(cfg::RadiativeConfig, Q::Real)
    x1, x2, y1, y2 = cfg.patch
    qv = Q / ((x2 - x1) * (y2 - y1))
    return x -> (x1 <= x[1] <= x2 && y1 <= x[2] <= y2) ? qv : 0.0
end

"""
Uniform temperature that balances `Q` against radiation from the right edge —
the Newton initial guess. Falls back to 300 K when radiation is off or Q = 0.
"""
function equilibrium_estimate(cfg::RadiativeConfig, eps_r::Real, Q::Real)
    (eps_r <= 1e-12 || Q <= 0) && return 300.0
    return (Q / (cfg.Ly * eps_r * cfg.sigma) + cfg.T_space^4)^(1 / 4)
end

# ----- manufactured solution (nonlinear-BC MMS) ----------------------------------
# u*(x,y) = 300 + 50 cos(πx) cos(2πy) on (0,1)×(0,0.5):
#   ∂u*/∂n = 0 on all four edges (natural-BC compatible), u* ∈ [250, 350] > 0,
#   f = -kΔu* = 250π²k cos(πx)cos(2πy),
#   the radiation BC becomes inhomogeneous with data g = εσ(u*⁴ - T_s⁴) on Γ_r.
mms_solution(x) = 300.0 + 50.0 * cos(pi * x[1]) * cos(2pi * x[2])
mms_bulk(cfg::RadiativeConfig) = x -> 250.0 * pi^2 * cfg.k * cos(pi * x[1]) * cos(2pi * x[2])
mms_bc_data(cfg::RadiativeConfig, eps_r::Real) =
    x -> eps_r * cfg.sigma * (rad4(mms_solution(x)) - cfg.T_space^4)
