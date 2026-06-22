# Backend-agnostic Makie plotting helpers.
#
# Loaded by the figure script (with CairoMakie active) and reused by the console.
# Every helper takes plain arrays so it stays decoupled from the solver package.

using Makie

"Heatmap of a scalar field Z[i,j] sampled at coordinates (xs, ys), with a colorbar."
function field_figure(xs, ys, Z; title = "", colormap = :viridis,
                      colorrange = Makie.automatic, size = (470, 400))
    fig = Figure(; size)
    ax = Axis(fig[1, 1]; title, xlabel = "x", ylabel = "y", aspect = DataAspect())
    hm = heatmap!(ax, xs, ys, Z; colormap, colorrange)
    Colorbar(fig[1, 2], hm)
    return fig
end

"Log-log convergence plot with an O(h²) reference line (method of manufactured solutions)."
function convergence_figure(hs, errs; title = "Solver convergence (MMS)")
    fig = Figure(; size = (540, 430))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
              xlabel = "grid spacing h", ylabel = "L² error", title)
    scatterlines!(ax, hs, errs; color = :dodgerblue, markersize = 11, label = "measured")
    href = [minimum(hs), maximum(hs)]
    c = errs[argmax(hs)] / maximum(hs)^2
    lines!(ax, href, c .* href .^ 2; linestyle = :dash, color = :black, label = "O(h²)")
    axislegend(ax; position = :lt)
    return fig
end

"Semi-log singular-value spectrum (normalised), with optional dashed rank markers."
function sigma_decay_figure(svals; ranks = (), nshow = min(60, length(svals)),
                            title = "Singular value decay")
    fig = Figure(; size = (580, 430))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "mode index i",
              ylabel = "σᵢ / σ₁", title)
    s = max.(svals[1:nshow] ./ svals[1], 1e-16)
    scatterlines!(ax, 1:nshow, s; color = :crimson, markersize = 7)
    for r in ranks
        r ≤ nshow || continue
        vlines!(ax, r; color = :gray, linestyle = :dash)
        text!(ax, r, s[r]; text = "  r=$r", align = (:left, :center), fontsize = 12)
    end
    return fig
end

"Reconstruction error (mean relative L²) versus retained rank."
function recon_error_figure(ranks, errs; title = "POD reconstruction error vs rank")
    fig = Figure(; size = (540, 410))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "rank r",
              ylabel = "mean relative L² error", title)
    scatterlines!(ax, collect(ranks), errs; color = :seagreen, markersize = 10)
    return fig
end

"Generic labelled bar chart; values are annotated above each bar."
function bar_figure(names, values; ylabel = "", title = "", color = :steelblue,
                    fmt = x -> string(round(x; sigdigits = 3)))
    n = length(names)
    fig = Figure(; size = (120 + 130n, 420))
    ax = Axis(fig[1, 1]; ylabel, title, xticks = (1:n, collect(String.(names))))
    barplot!(ax, 1:n, collect(values); color)
    ymax = maximum(values)
    for (i, v) in enumerate(values)
        text!(ax, i, v + 0.02ymax; text = fmt(v), align = (:center, :bottom), fontsize = 12)
    end
    ylims!(ax, 0, 1.18ymax)
    return fig
end

"Three-panel comparison: ground-truth field, prediction (shared scale) and |error|."
function comparison_triptych(xs, ys, Utrue, Upred; title = "")
    fig = Figure(; size = (1120, 380))
    cr = extrema(vcat(vec(Utrue), vec(Upred)))
    ax1 = Axis(fig[1, 1]; title = "true u", aspect = DataAspect())
    hm1 = heatmap!(ax1, xs, ys, Utrue; colorrange = cr)
    ax2 = Axis(fig[1, 2]; title = "predicted û", aspect = DataAspect())
    heatmap!(ax2, xs, ys, Upred; colorrange = cr)
    Colorbar(fig[1, 3], hm1)
    ax3 = Axis(fig[1, 4]; title = "|û − u|", aspect = DataAspect())
    hm3 = heatmap!(ax3, xs, ys, abs.(Utrue .- Upred); colormap = :magma)
    Colorbar(fig[1, 5], hm3)
    isempty(title) || Label(fig[0, :], title; fontsize = 16, font = :bold)
    return fig
end

"Static multi-panel 'dashboard' (a stand-in screenshot for the live console)."
function dashboard_figure(xs, ys, A, Utrue, Upred, svals; r = 0, μ = nothing)
    fig = Figure(; size = (1150, 720))
    err = abs.(Utrue .- Upred)
    cr = extrema(vcat(vec(Utrue), vec(Upred)))
    panels = [("a(x,y;μ)", A, :viridis, Makie.automatic),
              ("true u", Utrue, :viridis, cr),
              ("predicted û", Upred, :viridis, cr),
              ("|û − u|", err, :magma, Makie.automatic)]
    for (k, (ttl, Z, cmap, crange)) in enumerate(panels)
        i, j = fldmod1(k, 2)
        ax = Axis(fig[i, 2j - 1]; title = ttl, aspect = DataAspect())
        hm = heatmap!(ax, xs, ys, Z; colormap = cmap, colorrange = crange)
        Colorbar(fig[i, 2j], hm)
    end
    ax = Axis(fig[3, 1:2]; yscale = log10, xlabel = "mode i", ylabel = "σᵢ/σ₁",
              title = "spectrum")
    s = max.(svals[1:min(50, length(svals))] ./ svals[1], 1e-16)
    scatterlines!(ax, 1:length(s), s; color = :crimson, markersize = 6)
    r > 0 && vlines!(ax, r; color = :gray, linestyle = :dash)
    hdr = "Reduced-Basis Surrogate Explorer" *
          (μ === nothing ? "" : "   —   μ = ($(round(μ[1];digits=2)), $(round(μ[2];digits=2))), r=$r")
    Label(fig[0, :], hdr; fontsize = 17, font = :bold)
    return fig
end
