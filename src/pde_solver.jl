# Full-order solver for  -∇·(a∇u) = f  on (0,1)² with homogeneous Dirichlet BC.
#
# Conservative second-order finite-volume / finite-difference scheme on a uniform
# grid. Face coefficients use the harmonic average of the two adjacent nodal
# values, which keeps the assembled matrix symmetric positive definite.

"""Harmonic average of two positive coefficient values (face coefficient)."""
@inline harmonic(a, b) = 2 * a * b / (a + b)

"""
    assemble_matrix(g::Grid, afun) -> SparseMatrixCSC

Assemble the symmetric positive-definite stiffness matrix `A` of the discrete
operator `-∇·(a∇·)` for diffusion field `afun(x,y)`. Uses a 5-point conservative
stencil; each interior face contributes `a_face/h²` with `a_face` the harmonic
average of its two nodal coefficients. Connections to boundary nodes fold into the
diagonal only (since `u = 0` there), so `A` is `n² × n²`.
"""
function assemble_matrix(g::Grid, afun)
    n, h = g.n, g.h
    h2 = h^2
    N = n^2
    I = Int[]; J = Int[]; V = Float64[]
    sizehint!(I, 5N); sizehint!(J, 5N); sizehint!(V, 5N)
    xc(i) = i * h
    @inbounds for j in 1:n, i in 1:n
        k = lin(g, i, j)
        aC = afun(xc(i), xc(j))
        aE = harmonic(aC, afun(xc(i + 1), xc(j)))   # east  face (between i and i+1)
        aW = harmonic(aC, afun(xc(i - 1), xc(j)))   # west  face
        aN = harmonic(aC, afun(xc(i), xc(j + 1)))   # north face
        aS = harmonic(aC, afun(xc(i), xc(j - 1)))   # south face
        push!(I, k); push!(J, k); push!(V, (aE + aW + aN + aS) / h2)
        i < n && (push!(I, k); push!(J, lin(g, i + 1, j)); push!(V, -aE / h2))
        i > 1 && (push!(I, k); push!(J, lin(g, i - 1, j)); push!(V, -aW / h2))
        j < n && (push!(I, k); push!(J, lin(g, i, j + 1)); push!(V, -aN / h2))
        j > 1 && (push!(I, k); push!(J, lin(g, i, j - 1)); push!(V, -aS / h2))
    end
    return sparse(I, J, V, N, N)
end

"""
    assemble_rhs(g::Grid, ffun) -> Vector

Right-hand side `b` with `b[k] = f(x_i, y_j)` at interior node `k = (i,j)`. No
boundary correction is needed because the Dirichlet data is homogeneous.
"""
function assemble_rhs(g::Grid, ffun)
    n, h = g.n, g.h
    b = Vector{Float64}(undef, n^2)
    @inbounds for j in 1:n, i in 1:n
        b[lin(g, i, j)] = ffun(i * h, j * h)
    end
    return b
end

"""
    solve_pde(g::Grid, afun, ffun) -> Vector

Assemble and solve `A u = b` for the interior solution (length `n²`). `A` is SPD,
so a sparse Cholesky factorization is used.
"""
function solve_pde(g::Grid, afun, ffun)
    A = assemble_matrix(g, afun)
    b = assemble_rhs(g, ffun)
    return cholesky(Symmetric(A)) \ b
end

"""
    solve_parametric(g::Grid, μ; ffun=forcing) -> Vector

Convenience: solve the parametric problem at parameter vector `μ` using the
default [`diffusion`](@ref) coefficient and forcing `ffun`.
"""
solve_parametric(g::Grid, μ; ffun = forcing) =
    solve_pde(g, (x, y) -> diffusion(x, y, μ), ffun)

"""Reshape an interior solution vector into an `n×n` matrix (`U[i,j]` at node `(i,j)`)."""
reshape_interior(g::Grid, u::AbstractVector) = reshape(collect(u), g.n, g.n)

"""
    embed_full(g::Grid, u) -> (n+2)×(n+2) matrix

Embed the interior solution into the full grid, padding with the zero Dirichlet
boundary. Pairs with [`full_coords`](@ref) for plotting on all of `[0,1]²`.
"""
function embed_full(g::Grid, u::AbstractVector)
    n = g.n
    U = zeros(n + 2, n + 2)
    U[2:(n + 1), 2:(n + 1)] .= reshape_interior(g, u)
    return U
end
