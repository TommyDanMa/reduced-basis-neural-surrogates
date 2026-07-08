# Transient integrator checks: convergence to the steady state, dt-orders of
# the θ-family (ImplicitEuler ≈ 1, Trapezoid ≈ 2, measured against a tight-
# tolerance FBDF reference with a smooth load), and energy bookkeeping
# dE/dt = P_in − P_rad inside a smooth segment of a square-wave cycle.

@testset "transient" begin
    cfg = RadiativeConfig(nx = 24, ny = 12)
    fom = build_steady_fom(cfg)
    eps_r = 0.5

    # long-horizon constant load → the steady solution
    st = solve_steady(fom; eps_r, Q = 450.0, abstol = 1e-10)
    u0 = fill(equilibrium_estimate(cfg, eps_r, 450.0) - 40.0, fom.n)
    sol = solve_transient(fom; eps_r, load = t -> 450.0, u0,
                          tspan = (0.0, 2.0e5), abstol = 1e-8, reltol = 1e-8,
                          saveat = [2.0e5])
    @test NonlinearSolve.SciMLBase.successful_retcode(sol)
    # bounded by the integrator tolerance (reltol 1e-8 accumulates to ~1e-5),
    # not by the horizon (t = 2e5 s ≈ 17 radiative time constants)
    @test relative_l2_error(sol.u[end], st.x) < 5e-5

    # fixed-dt orders with a smooth sinusoidal load
    load = t -> 300.0 + 150.0 * sin(2pi * t / 2000.0)
    u0s = solve_steady(fom; eps_r, Q = 300.0, abstol = 1e-10).x
    T = 2000.0
    uref = solve_transient(fom; eps_r, load, u0 = u0s, tspan = (0.0, T),
                           abstol = 1e-11, reltol = 1e-11, saveat = [T]).u[end]
    dt_err(alg, dt) = relative_l2_error(
        solve_transient(fom; eps_r, load, u0 = u0s, tspan = (0.0, T),
                        alg, dt, saveat = [T]).u[end], uref)
    e_be = [dt_err(ImplicitEuler(), dt) for dt in (200.0, 100.0, 50.0)]
    r_be = [log2(e_be[i] / e_be[i+1]) for i in 1:2]
    e_tr = [dt_err(Trapezoid(), dt) for dt in (400.0, 200.0, 100.0)]
    r_tr = [log2(e_tr[i] / e_tr[i+1]) for i in 1:2]
    @info "transient dt-convergence" e_be r_be e_tr r_tr
    @test 0.75 <= r_be[end] <= 1.25
    @test 1.7 <= r_tr[end] <= 2.3

    # energy bookkeeping in the ON window of a square wave
    period, duty, Qp = 2000.0, 0.5, 600.0
    seg = collect(2050.0:25.0:2950.0)      # strictly inside the second ON window
    solq = solve_transient(fom; eps_r, load = square_wave(Qp, period, duty),
                           u0 = u0s, tspan = (0.0, 3000.0),
                           tstops = load_switch_times(period, duty, 0.0, 3000.0),
                           saveat = seg, abstol = 1e-9, reltol = 1e-9)
    E = [thermal_energy(fom, u) for u in solq.u]
    # discretely injected power: at this (non-aligned) grid the patch quadrature
    # captures ~91% of the nominal Q — bookkeeping must use the same quadrature
    Pin = Qp * source_power(fom, 1.0)
    Pnet = [Pin - radiated_power(fom, FEFunction(fom.Un, u), eps_r) for u in solq.u]
    dt = seg[2] - seg[1]
    work = sum((Pnet[i] + Pnet[i+1]) / 2 * dt for i in 1:length(Pnet)-1)
    @test abs((E[end] - E[1]) - work) / abs(E[end] - E[1]) < 0.01
end
