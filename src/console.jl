# Interactive live-calibration console (GLMakie).
#
# `build_console` wires a reactive Observable graph: moving a μ slider re-solves the
# full-order model, re-runs the selected surrogate, and refreshes every panel
# (coefficient field, true u, predicted û, |error|, PDE residual, σ-spectrum) plus
# the numeric read-outs. Launched by `scripts/09_run_console.jl`.

using GLMakie
using ReducedBasisSurrogates
using Printf

"""
    build_console(g, P, mu_train, models; figdir, param_lo, param_hi) -> Figure

`models` is a `Dict` mapping a menu label (e.g. "POD-MLP", "Direct MLP") to a
named tuple `(; kind, model, ps, st, outdim, nparams)`. A "POD projection" entry
that truncates the *exact* solution onto `r` modes is always added as a basis-only
reference.
"""
function build_console(g::Grid, P::PODBasis, mu_train::AbstractMatrix, models::AbstractDict;
                       figdir = pwd(), param_lo = -1.0, param_hi = 1.0)
    xs = full_coords(g)
    pod_models = [k for (k, v) in models if startswith(v.kind, "pod")]
    rmax = isempty(pod_models) ? min(20, nmodes(P)) : minimum(models[k].outdim for k in pod_models)
    options = vcat(collect(keys(models)), "POD projection")
    default_choice = "POD-MLP" in options ? "POD-MLP" : first(options)

    fig = Figure(size = (1500, 880))
    Label(fig[1, 1:4], "Reduced-Basis Surrogate Explorer";
          fontsize = 24, font = :bold, color = :gray15)

    # ---- controls -------------------------------------------------------------
    ctrl = GridLayout(fig[2:3, 1])
    sg = SliderGrid(ctrl[1, 1],
                    (label = "μ₁", range = param_lo:0.01:param_hi, startvalue = 0.4),
                    (label = "μ₂", range = param_lo:0.01:param_hi, startvalue = -0.4),
                    (label = "rank r", range = 1:rmax, startvalue = rmax))
    Label(ctrl[2, 1], "surrogate model"; fontsize = 14, halign = :left, font = :bold)
    menu = Menu(ctrl[3, 1]; options = options, default = default_choice)

    μ      = lift((a, b) -> [a, b], sg.sliders[1].value, sg.sliders[2].value)
    rval   = sg.sliders[3].value
    choice = menu.selection

    # ---- reactive computation -------------------------------------------------
    truebundle = lift(μ) do m
        t = @elapsed (u = solve_parametric(g, m))
        (; u, t, μ = m)
    end
    predbundle = lift(truebundle, rval, choice) do tb, r, c
        t = @elapsed begin
            u = if c == "POD projection"
                    reconstruct(P, project(P, tb.u, r), r)
                elseif c == "Direct MLP"
                    predict(models[c].model, models[c].ps, models[c].st, tb.μ)
                else                                   # any POD-* coefficient model
                    rr = min(r, models[c].outdim)
                    ĉ = predict(models[c].model, models[c].ps, models[c].st, tb.μ)
                    reconstruct(P, ĉ[1:rr], rr)
                end
        end
        (; u, t)
    end

    Atrue  = lift(tb -> [diffusion(x, y, tb.μ) for x in xs, y in xs], truebundle)
    Utrue  = lift(tb -> embed_full(g, tb.u), truebundle)
    Uhat   = lift(pb -> embed_full(g, pb.u), predbundle)
    Uerr   = lift((a, b) -> abs.(a .- b), Utrue, Uhat)
    Rfield = lift(truebundle, predbundle) do tb, pb
        embed_full(g, residual(g, (x, y) -> diffusion(x, y, tb.μ), forcing, pb.u))
    end
    rescr  = lift(R -> (m = maximum(abs, R); m = iszero(m) ? 1.0 : m; (-m, m)), Rfield)
    ufield_range = lift(U -> (mn = minimum(U); mx = maximum(U); mn == mx ? (mn, mn + 1e-9) : (mn, mx)), Utrue)

    # ---- heatmap panels -------------------------------------------------------
    function panel!(pos, title, Z; colormap = :viridis, colorrange = Makie.automatic)
        ax = Axis(pos[1, 1]; title, aspect = DataAspect(), xticklabelsvisible = false,
                  yticklabelsvisible = false)
        hm = heatmap!(ax, xs, xs, Z; colormap, colorrange)
        Colorbar(pos[1, 2], hm)
        return ax
    end
    panel!(fig[2, 2], "a(x,y;μ)", Atrue)
    panel!(fig[2, 3], "true u  (FOM)", Utrue; colorrange = ufield_range)
    panel!(fig[2, 4], "predicted û", Uhat; colorrange = ufield_range)
    panel!(fig[3, 2], "|û − u|", Uerr; colormap = :magma)
    panel!(fig[3, 3], "PDE residual  A û − b", Rfield; colormap = :balance, colorrange = rescr)

    # ---- σ spectrum -----------------------------------------------------------
    axσ = Axis(fig[3, 4]; yscale = log10, title = "singular values",
               xlabel = "mode i", ylabel = "σᵢ/σ₁")
    nshow = min(40, nmodes(P))
    s = max.(P.svals[1:nshow] ./ P.svals[1], 1e-16)
    scatterlines!(axσ, 1:nshow, s; color = :crimson, markersize = 6)
    vlines!(axσ, rval; color = :gray, linestyle = :dash)

    # ---- read-outs + parameter-space minimap + save button --------------------
    nparams_of(c) = c == "POD projection" ? "0 (basis only)" : string(models[c].nparams)
    info = lift(truebundle, predbundle, choice) do tb, pb, c
        rows = (("rel L² error", @sprintf("%.2e", relative_l2_error(pb.u, tb.u))),
                ("rel residual", @sprintf("%.2e", relative_residual(g, tb.μ, pb.u))),
                ("parameters", nparams_of(c)),
                ("FOM solve", @sprintf("%.2f ms", 1000tb.t)),
                ("predict", @sprintf("%.3f ms", 1000pb.t)),
                ("speedup", @sprintf("%.0f×", tb.t / max(pb.t, eps()))))
        join((rpad(k, 13) * ":  " * v for (k, v) in rows), "\n")
    end
    Label(ctrl[4, 1], info; halign = :left, justification = :left, fontsize = 15)

    axmap = Axis(ctrl[5, 1]; title = "parameter space", xlabel = "μ₁", ylabel = "μ₂",
                 aspect = DataAspect())
    scatter!(axmap, mu_train[1, :], mu_train[2, :]; color = (:steelblue, 0.25), markersize = 5)
    scatter!(axmap, lift(tb -> Point2f(tb.μ[1], tb.μ[2]), truebundle);
             color = :red, markersize = 15, marker = :xcross)
    limits!(axmap, param_lo - 0.1, param_hi + 0.1, param_lo - 0.1, param_hi + 0.1)

    btn = Button(ctrl[6, 1]; label = "save screenshot", tellwidth = false)
    on(btn.clicks) do _
        path = joinpath(figdir, "console_screenshot.png")
        save(path, fig)
        @info "saved $path"
    end

    rowsize!(ctrl, 5, Relative(0.28))
    colsize!(fig.layout, 1, Relative(0.24))
    DataInspector(fig)
    return fig
end
