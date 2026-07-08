# Steady snapshot dataset: Sobol samples over μ = (ε, Q), one converged
# nonlinear FOM solve each. Every solve starts from the analytic radiative-
# equilibrium estimate (validated robust across the whole μ-box in script 01),
# which keeps the dataset embarrassingly deterministic.

include(joinpath(@__DIR__, "config.jl"))
using QuasiMonteCarlo

println("=== 02_generate_steady_dataset ===")

fom = build_steady_fom(CFG)
lb = [EPS_RANGE[1], Q_RANGE[1]]
ub = [EPS_RANGE[2], Q_RANGE[2]]
mus = QuasiMonteCarlo.sample(N_SAMPLES, lb, ub, SobolSample())   # 2 × M

n = fom.n_nodes
U = Matrix{Float64}(undef, n, N_SAMPLES)
peakT = Vector{Float64}(undef, N_SAMPLES)
Prad = Vector{Float64}(undef, N_SAMPLES)
nsteps = Vector{Int}(undef, N_SAMPLES)
times = Vector{Float64}(undef, N_SAMPLES)
worst_bal = 0.0

for k in 1:N_SAMPLES
    μ = (mus[1, k], mus[2, k])
    t = @elapsed s = solve_steady(fom, μ; abstol = 1e-9)
    @assert s.converged "FOM diverged at μ = $μ"
    U[:, k] = nodal_values(fom, s.uh)
    peakT[k] = maximum(view(U, :, k))
    Prad[k] = radiated_power(fom, s.uh, μ[1])
    nsteps[k] = s.nsteps
    times[k] = t
    global worst_bal = max(worst_bal,
                           energy_balance(fom, s.uh; eps_r = μ[1], Q = μ[2]).rel)
    k % 64 == 0 && @printf("  %3d / %d solves  (last: %.0f ms, %d steps)\n",
                           k, N_SAMPLES, 1e3 * t, s.nsteps)
end

# train/test split (fixed RNG, parent convention)
idx = shuffle(StableRNG(SEED), 1:N_SAMPLES)
ntest = round(Int, TEST_FRAC * N_SAMPLES)
test_idx = sort(idx[1:ntest])
train_idx = sort(idx[(ntest + 1):end])

# standardized network inputs in [-1,1]²
X = vcat(standardize.(mus[1:1, :], EPS_RANGE...),
         standardize.(mus[2:2, :], Q_RANGE...))

jldsave(snapshots_path();
        mus, X, U, train_idx, test_idx, peakT, Prad, nsteps, times,
        nx = CFG.nx, ny = CFG.ny)

@printf("done: %d snapshots (n = %d)   train %d / test %d\n",
        N_SAMPLES, n, length(train_idx), length(test_idx))
@printf("FOM solve: median %.0f ms, mean %.0f ms   Newton steps: %d–%d\n",
        1e3 * median(times), 1e3 * mean(times), minimum(nsteps), maximum(nsteps))
@printf("worst energy balance rel = %.2e   peak T ∈ [%.0f, %.0f] K\n",
        worst_bal, minimum(peakT), maximum(peakT))
println("saved ", snapshots_path())
