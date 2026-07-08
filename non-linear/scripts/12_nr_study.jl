# "How many modes when radiation dominates?" — the σ-decay of ε-conditioned
# snapshot subsets, steady and space-time, against the radiation number
# Nr = εσT³L/k. The steady manifold stays ~rank-3 in every regime (levels and
# amplitudes of fixed shapes); the transient mode count is where the radiation
# regime actually bites.

include(joinpath(@__DIR__, "config.jl"))
using CairoMakie
CairoMakie.activate!()

println("=== 12_nr_study ===")

nbins = 3
edges = collect(range(EPS_RANGE...; length = nbins + 1))
binlabel(b) = @sprintf("ε ∈ [%.2f, %.2f]", edges[b], edges[b+1])
"Modes needed to reach a reconstruction tolerance (from a σ tail-energy bound)."
function modes_to_tol(svals, tol)
    tail = sqrt.(max.(reverse(cumsum(reverse(svals .^ 2))), 0.0)) ./ norm(svals)
    i = findfirst(<(tol), tail)
    return i === nothing ? length(svals) : i - 1
end

# ----- steady, conditioned on ε ----------------------------------------------------
ds = load_snapshots()
Us, mus_s = ds["U"], ds["mus"]
fom = build_steady_fom(CFG)

steady_decays = Vector{Vector{Float64}}()
Nr_s = Float64[]
for b in 1:nbins
    sel = findall(k -> edges[b] <= mus_s[1, k] < edges[b+1] + (b == nbins),
                  1:size(mus_s, 2))
    Pb = fit_pod(Us[:, sel])
    push!(steady_decays, Pb.svals ./ Pb.svals[1])
    T_c = mean(mean(to_grid(fom, Us[:, k])[end, :]) for k in sel)  # edge temp
    ε̄ = mean(mus_s[1, sel])
    push!(Nr_s, ε̄ * CFG.sigma * T_c^3 * CFG.Lx / CFG.k)
    @printf("steady   %s: %3d snaps, edge T̄ %.0f K, Nr %.3f, modes@1e-6 = %d\n",
            binlabel(b), length(sel), T_c, Nr_s[end],
            modes_to_tol(Pb.svals, 1e-6))
end

# ----- space-time, conditioned on ε ------------------------------------------------
dt_ = load(snapshots_t_path())
U3, mus_t = dt_["U"], dt_["mus"]
n, nt, _ = size(U3)

st_decays = Vector{Vector{Float64}}()
Nr_t = Float64[]
m4_s, m6_s, m4_t, m6_t = Int[], Int[], Int[], Int[]
for b in 1:nbins
    sel = findall(k -> edges[b] <= mus_t[1, k] < edges[b+1] + (b == nbins),
                  1:size(mus_t, 2))
    Pb = fit_pod(reshape(U3[:, :, sel], n, :))
    push!(st_decays, Pb.svals ./ Pb.svals[1])
    T_c = mean(mean(to_grid(fom, U3[:, end, k])[end, :]) for k in sel)
    ε̄ = mean(mus_t[1, sel])
    push!(Nr_t, ε̄ * CFG.sigma * T_c^3 * CFG.Lx / CFG.k)
    push!(m4_t, modes_to_tol(Pb.svals, 1e-4))
    push!(m6_t, modes_to_tol(Pb.svals, 1e-6))
    push!(m4_s, modes_to_tol(fit_pod(Us[:, findall(k -> edges[b] <= mus_s[1, k] <
        edges[b+1] + (b == nbins), 1:size(mus_s, 2))]).svals, 1e-4))
    push!(m6_s, modes_to_tol(steady_decays[b] .* 1.0, 1e-6))
    @printf("transient %s: %3d trajs, Nr %.3f, modes@1e-4 = %d, @1e-6 = %d\n",
            binlabel(b), length(sel), Nr_t[end], m4_t[end], m6_t[end])
end

jldsave(joinpath(DATA_DIR, "nr_study.jld2");
        edges, Nr_s, Nr_t, m4_t, m6_t, m4_s, m6_s)

fig = Figure(size = (960, 400))
ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = "mode index r", ylabel = "σᵣ/σ₁",
           title = "σ-decay by emissivity bin (solid: space-time, dash: steady)")
colors = Makie.wong_colors()
for b in 1:nbins
    ns = min(30, length(steady_decays[b]))
    lines!(ax1, 1:ns, max.(steady_decays[b][1:ns], 1e-17);
           color = colors[b], linestyle = :dash)
    nst = min(30, length(st_decays[b]))
    lines!(ax1, 1:nst, max.(st_decays[b][1:nst], 1e-17);
           color = colors[b], label = binlabel(b))
end
axislegend(ax1; position = :rt)
ax2 = Axis(fig[1, 2]; xlabel = "radiation number Nr = εσT³L/k",
           ylabel = "modes needed",
           title = "Space-time modes to reach tolerance vs Nr")
scatterlines!(ax2, Nr_t, Float64.(m4_t); label = "tol 1e-4")
scatterlines!(ax2, Nr_t, Float64.(m6_t); label = "tol 1e-6")
axislegend(ax2; position = :lt)
save(joinpath(FIG_DIR, "13_nr_study.png"), fig)

println("saved data/nr_study.jld2 and figures/13_nr_study.png")
