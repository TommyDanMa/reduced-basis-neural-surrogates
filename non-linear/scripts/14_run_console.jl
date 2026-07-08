# Live design console (GLMakie). Drag ε and Q and watch the steady FOM
# (solved live, warm-started), the POD-MLP surrogate, and their error update in
# real time; the transient panel shows the surrogate's peak-T trace for the
# current (ε, Qp, duty, period) instantly, with a button to overlay the true
# transient FOM. The console is the microscope, not the science.
#
# Headless smoke test (CI / no display):  CONSOLE_SMOKE=1 julia ... 14_run_console.jl

include(joinpath(@__DIR__, "config.jl"))
using NonlinearSolve
using Gridap
const SMOKE = get(ENV, "CONSOLE_SMOKE", "0") == "1"
if SMOKE
    using CairoMakie; CairoMakie.activate!()
else
    using GLMakie; GLMakie.activate!()
end
import Makie
const Observables = Makie.Observables

println("=== 14_run_console ===")

fom = build_steady_fom(CFG)
pod_s = load_pod()
Ps = pod_s.P
ms = load_mlp(model_path("pod_mlp_r$(DEFAULT_R)"))
pod_t = load(pod_t_path())
Pt = PODBasis(pod_t["mean"], pod_t["modes"], pod_t["svals"])
mt = load_mlp(model_path("pod_mlp_t_r$(DEFAULT_R_T)"))
xs, ys = node_axes(fom)
nt = NCYCLES * NPHASE + 1
warm = Ref{Union{Nothing,Vector{Float64}}}(nothing)

steady_sur(ε, Q) = reconstruct(Ps, ms.cscales .* predict(ms.model, ms.ps, ms.st,
                               [standardize_mu(ε, Q)...]), DEFAULT_R)
function steady_fom_solve(ε, Q)
    t = @elapsed s = solve_steady(fom; eps_r = ε, Q, u0 = warm[])
    warm[] = s.x
    return s.x, t
end
function transient_sur(ε, Qp, duty, period)
    μst = standardize_mu_t(ε, Qp, duty, period)
    X = reduce(hcat, [transient_features(μst, j) for j in 1:nt])
    Û = Matrix(reconstruct(Pt, mt.cscales .* predict(mt.model, mt.ps, mt.st, X),
                           DEFAULT_R_T))
    return vec(maximum(Û; dims = 1))
end

fig = Figure(size = (1380, 860))
Label(fig[0, 1:6], "RadiativeSurrogates console — nonlinear radiation BC, Gridap × SciML";
      fontsize = 20, font = :bold)

axf = Axis(fig[1, 1]; title = "steady FOM T (K)", aspect = DataAspect())
axs = Axis(fig[1, 3]; title = "POD-MLP r=$(DEFAULT_R) T (K)", aspect = DataAspect())
axe = Axis(fig[1, 5]; title = "|error| (K)", aspect = DataAspect())

sg = SliderGrid(fig[3, 1:3],
    (label = "ε", range = 0.10:0.01:0.95, startvalue = 0.5),
    (label = "Q / Qp (W/m)", range = 100.0:10.0:800.0, startvalue = 450.0),
    (label = "duty", range = 0.20:0.05:0.80, startvalue = 0.6),
    (label = "period (s)", range = 1800.0:200.0:10800.0, startvalue = 5400.0))
s_ε, s_Q, s_d, s_p = (s.value for s in sg.sliders)

info = Label(fig[2, 1:6], ""; fontsize = 14, tellwidth = false)

μ_steady = Observables.throttle(0.10, lift(tuple, s_ε, s_Q))
Ff = Observable(zeros(length(xs), length(ys)))
Fs = Observable(zeros(length(xs), length(ys)))
Fe = Observable(zeros(length(xs), length(ys)))
crange = Observable((250.0, 600.0))

hmf = heatmap!(axf, xs, ys, Ff; colormap = :inferno, colorrange = crange)
heatmap!(axs, xs, ys, Fs; colormap = :inferno, colorrange = crange)
hme = heatmap!(axe, xs, ys, Fe; colormap = :viridis)
Colorbar(fig[1, 2], hmf); Colorbar(fig[1, 6], hme)
Colorbar(fig[1, 4], hmf)

function update_steady((ε, Q))
    xf, tsolve = steady_fom_solve(ε, Q)
    tsur = @elapsed û = steady_sur(ε, Q)
    Ff[] = to_grid(fom, xf); Fs[] = to_grid(fom, û)
    Fe[] = to_grid(fom, abs.(û .- xf))
    crange[] = extrema(Ff[])
    rel = relative_l2_error(û, xf)
    uhf = FEFunction(fom.Un, xf); uhs = FEFunction(fom.Un, collect(û))
    info.text = @sprintf(
        "steady:  FOM %.0f ms  |  surrogate %.2f ms  |  rel L² %.2e  |  peak T %.1f / %.1f K  |  P_rad %.1f / %.1f W/m",
        1e3 * tsolve, 1e3 * tsur, rel,
        maximum(xf), maximum(û),
        radiated_power(fom, uhf, ε), radiated_power(fom, uhs, ε))
end
on(update_steady, μ_steady)

# ----- transient panel -------------------------------------------------------------
axt = Axis(fig[4, 1:4]; xlabel = "t / period", ylabel = "peak T (K)",
           title = "transient: surrogate live, FOM on demand")
τax = collect(range(0.0, NCYCLES; length = nt))
pk_sur = Observable(zeros(nt))
pk_fom = Observable(fill(NaN, nt))
lines!(axt, τax, pk_sur; color = :tomato, label = "surrogate")
lines!(axt, τax, pk_fom; color = :black, linestyle = :dash, label = "FOM")
axislegend(axt; position = :rt)

μ_trans = Observables.throttle(0.10, lift(tuple, s_ε, s_Q, s_d, s_p))
on(μ_trans) do (ε, Qp, d, p)
    pk_sur[] = transient_sur(ε, Qp, d, p)
    pk_fom[] = fill(NaN, nt)               # stale FOM overlay is worse than none
    autolimits!(axt)
end

btn = Button(fig[4, 5:6]; label = "run transient FOM (~10 s)")
on(btn.clicks) do _
    ε, Qp, d, p = s_ε[], s_Q[], s_d[], s_p[]
    T_end = NCYCLES * p
    u0 = solve_steady(fom; eps_r = ε, Q = d * Qp).x
    sol = solve_transient(fom; eps_r = ε, load = square_wave(Qp, p, d), u0,
                          tspan = (0.0, T_end),
                          tstops = load_switch_times(p, d, 0.0, T_end),
                          saveat = collect(range(0.0, T_end; length = nt)),
                          abstol = 1e-7, reltol = 1e-7)
    pk_fom[] = vec(maximum(Array(sol); dims = 1))
end

# σ-spectra minimap
axσ = Axis(fig[3, 4:6]; yscale = log10, ylabel = "σᵣ/σ₁",
           title = "σ-decay: steady (dots) vs space-time (line)")
scatter!(axσ, 1:12, max.(Ps.svals[1:12] ./ Ps.svals[1], 1e-17); color = :gray)
lines!(axσ, 1:40, max.(Pt.svals[1:40] ./ Pt.svals[1], 1e-17); color = :tomato)

notify(μ_steady); notify(μ_trans)

if SMOKE
    save(joinpath(FIG_DIR, "15_console.png"), fig)
    println("smoke mode: console rendered to figures/15_console.png")
else
    display(fig)
    println("console up — drag the sliders; close the window to exit.")
    wait(GLMakie.Screen(fig.scene))
end
