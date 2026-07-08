# Quantities of interest and field extraction. The design-tool QoIs are the
# radiated power and the peak temperature — those, not L² norms, rank designs.

"Total radiated power ∮ εσ(u|u|³ − T_s⁴) dΓ_r, in W per m depth."
function radiated_power(fom::SteadyFOM, uh, eps_r::Real)
    cfg = fom.cfg
    return sum(∫((eps_r * cfg.sigma) * ((rad4 ∘ uh) - cfg.T_space^4)) * fom.dΓ)
end

"Total injected power ∫ (q_patch + f) dΩ, in W per m depth."
function source_power(fom::SteadyFOM, Q::Real; f = nothing)
    qpatch = source_fn(fom.cfg, Q)
    fq = f === nothing ? qpatch : (x -> qpatch(x) + f(x))
    return sum(∫(CellField(fq, fom.Ω)) * fom.dΩ)
end

"""
    energy_balance(fom, uh; eps_r, Q, f = nothing) -> (; P_in, P_rad, rel)

Config B has no Dirichlet DOFs, so v ≡ 1 lies in the test space and the solved
discrete system satisfies P_in = P_rad up to the Newton residual tolerance.
"""
function energy_balance(fom::SteadyFOM, uh; eps_r::Real, Q::Real, f = nothing)
    P_in = source_power(fom, Q; f)
    P_rad = radiated_power(fom, uh, eps_r)
    return (; P_in, P_rad, rel = abs(P_rad - P_in) / max(abs(P_in), eps()))
end

"""
    dirichlet_influx(fom, uh; eps_r, Q = 0) -> W per m depth

Conductive power entering through the Dirichlet ("left") wall of config A,
computed exactly in the discrete sense as the reaction: evaluate the residual
form against the *unconstrained* test space and sum the rows attached to the
Dirichlet-wall nodes (v there sums to the wall's partition of unity).
"""
function dirichlet_influx(fom::SteadyFOM, uh; eps_r::Real, Q::Real = 0.0)
    res = steady_form(fom; eps_r, Q)
    r = assemble_vector(v -> res(uh, v), fom.Vn)
    xv = get_free_dof_values(interpolate_everywhere(x -> x[1], fom.Un))
    onwall = xv .< 1e-9
    # Testing against v ≡ 1 gives Σ_all r = P_rad, and the free rows vanish at
    # convergence, so the wall rows sum to the conductive influx P_in = P_rad.
    return sum(r[onwall])
end

"Nodal values of `uh` on the unconstrained P1 space (length (nx+1)(ny+1))."
function nodal_values(fom::SteadyFOM, uh)
    if fom.cfg.bc === :source
        return copy(get_free_dof_values(uh))
    end
    return copy(get_free_dof_values(interpolate_everywhere(uh, fom.Un)))
end

"Reshape a nodal vector to an (nx+1)×(ny+1) matrix on the structured grid."
function to_grid(fom::SteadyFOM, vals::AbstractVector)
    @assert !isempty(fom.gi) "to_grid/edge_profile require P1 elements (order = 1)"
    A = Matrix{Float64}(undef, fom.cfg.nx + 1, fom.cfg.ny + 1)
    @inbounds for d in eachindex(vals)
        A[fom.gi[d], fom.gj[d]] = vals[d]
    end
    return A
end

"Grid node coordinate axes (xs, ys) matching `to_grid`."
node_axes(fom::SteadyFOM) =
    (range(0.0, fom.cfg.Lx; length = fom.cfg.nx + 1),
     range(0.0, fom.cfg.Ly; length = fom.cfg.ny + 1))

"Peak temperature over all nodes (the design-limiting quantity)."
peak_temperature(fom::SteadyFOM, uh) = maximum(nodal_values(fom, uh))

"Temperature profile along the radiating edge x = Lx (returns ys, T(ys))."
function edge_profile(fom::SteadyFOM, uh)
    vals = nodal_values(fom, uh)
    A = to_grid(fom, vals)
    xs, ys = node_axes(fom)
    return collect(ys), A[end, :]
end

"L² norm of (uh − u_exact) using the refined bulk measure."
function l2_error(fom::SteadyFOM, uh, u_exact)
    e = u_exact - uh
    return sqrt(sum(∫(e * e) * fom.dΩe))
end
