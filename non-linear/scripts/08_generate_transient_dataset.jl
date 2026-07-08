# Transient snapshot dataset: Sobol samples over μ = (ε, Q_peak, duty, period),
# one square-wave load-cycle trajectory each (NCYCLES periods, phase-uniform
# saves), started from the steady state at the cycle-mean load. FBDF with
# tstops at every load switch.

include(joinpath(@__DIR__, "config.jl"))
using QuasiMonteCarlo
using NonlinearSolve

println("=== 08_generate_transient_dataset ===")

fom = build_steady_fom(CFG)
lb = [EPS_RANGE[1], Q_RANGE[1], DUTY_RANGE[1], PERIOD_RANGE[1]]
ub = [EPS_RANGE[2], Q_RANGE[2], DUTY_RANGE[2], PERIOD_RANGE[2]]
mus = QuasiMonteCarlo.sample(N_MU_T, lb, ub, SobolSample())     # 4 × N_MU_T

nt = NCYCLES * NPHASE + 1
n = fom.n_nodes
U = Array{Float64,3}(undef, n, nt, N_MU_T)
times = Vector{Float64}(undef, N_MU_T)

for k in 1:N_MU_T
    ε, Qp, duty, period = mus[:, k]
    T_end = NCYCLES * period
    tsave = collect(range(0.0, T_end; length = nt))
    u0 = solve_steady(fom; eps_r = ε, Q = duty * Qp, abstol = 1e-9).x
    t = @elapsed sol = solve_transient(fom; eps_r = ε,
                                       load = square_wave(Qp, period, duty),
                                       u0, tspan = (0.0, T_end),
                                       tstops = load_switch_times(period, duty, 0.0, T_end),
                                       saveat = tsave,
                                       abstol = 1e-7, reltol = 1e-7)
    @assert NonlinearSolve.SciMLBase.successful_retcode(sol) "diverged at μ = $(mus[:, k])"
    @assert length(sol.t) == nt
    U[:, :, k] = Array(sol)
    times[k] = t
    k % 16 == 0 && @printf("  %3d / %d trajectories  (last: %.1f s)\n",
                           k, N_MU_T, t)
end

idx = shuffle(StableRNG(SEED + 1), 1:N_MU_T)
ntest = round(Int, TEST_FRAC * N_MU_T)
test_idx = sort(idx[1:ntest])
train_idx = sort(idx[(ntest + 1):end])

jldsave(snapshots_t_path(); mus, U, train_idx, test_idx, times,
        nt, ncycles = NCYCLES, nphase = NPHASE)

@printf("done: %d trajectories × %d states (n = %d) — train %d / test %d\n",
        N_MU_T, nt, n, length(train_idx), length(test_idx))
@printf("transient FOM: median %.1f s, total %.0f s\n", median(times), sum(times))
println("saved ", snapshots_t_path())
