# Grid, the parametric diffusion coefficient a(x,y;μ), the forcing f(x,y),
# and a manufactured-forcing helper for the convergence study.

"""
    Grid(n, h)

Uniform grid on the unit square (0,1)² with `n` interior nodes per dimension and
spacing `h = 1/(n+1)`. Interior node `i` sits at `x = i·h` for `i = 1:n`; the
boundary nodes (index `0` and `n+1`) carry the homogeneous Dirichlet value `u = 0`
and are *not* stored as unknowns, so the linear system has `n²` unknowns.
Build one with [`make_grid`](@ref).
"""
struct Grid
    n::Int
    h::Float64
end

"""Construct a [`Grid`](@ref) with `n` interior nodes per dimension."""
make_grid(n::Integer) = Grid(Int(n), 1.0 / (n + 1))

"""Number of interior unknowns, `n²`."""
ndof(g::Grid) = g.n^2

"""Coordinate of node index `i` (`0` = left/bottom boundary, `n+1` = right/top)."""
@inline node_coord(g::Grid, i::Integer) = i * g.h

"""Interior coordinate vector `[h, 2h, …, n·h]` (length `n`)."""
interior_coords(g::Grid) = [i * g.h for i in 1:g.n]

"""Full coordinate vector including boundaries `[0, h, …, 1]` (length `n+2`)."""
full_coords(g::Grid) = [i * g.h for i in 0:(g.n + 1)]

"""Linear index of interior node `(i,j)`, with `i,j ∈ 1:n` and `i` varying fastest."""
@inline lin(g::Grid, i::Integer, j::Integer) = (j - 1) * g.n + i

"""
    diffusion(x, y, μ)

Parametric diffusion coefficient from the TR experiment,

    a(x,y;μ) = 1 + 0.3·μ₁·sin(πx)·sin(πy) + 0.2·μ₂·cos(2πy).

For `μ ∈ [-1,1]²` this stays `≥ 1 − 0.3 − 0.2 = 0.5 > 0`, so the operator is
uniformly elliptic and the problem is well posed.
"""
@inline function diffusion(x, y, μ)
    μ1, μ2 = μ[1], μ[2]
    return 1.0 + 0.3 * μ1 * sinpi(x) * sinpi(y) + 0.2 * μ2 * cospi(2y)
end

"""Default forcing term `f(x,y) = 1` (fixed; only `a` depends on the parameters)."""
@inline forcing(x, y) = 1.0

"""
    mms_forcing(ufun, afun, x, y)

Manufactured forcing `f = -∇·(a ∇u)` at `(x,y)` for an exact solution `ufun(x,y)`
and coefficient `afun(x,y)`, evaluated with `ForwardDiff` (no hand algebra) via

    ∇·(a∇u) = a·Δu + ∇a·∇u.

Used by the method-of-manufactured-solutions convergence test.
"""
function mms_forcing(ufun, afun, x, y)
    p = [float(x), float(y)]
    u = q -> ufun(q[1], q[2])
    a = q -> afun(q[1], q[2])
    gu = ForwardDiff.gradient(u, p)
    Hu = ForwardDiff.hessian(u, p)
    ga = ForwardDiff.gradient(a, p)
    laplacian = Hu[1, 1] + Hu[2, 2]
    divflux = afun(x, y) * laplacian + (ga[1] * gu[1] + ga[2] * gu[2])
    return -divflux
end
