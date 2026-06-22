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

"Rank sweep: relative L² error and PDE residual vs retained rank for two models."
function rank_sweep_figure(ranks, mlp_l2, kan_l2, mlp_res, kan_res)
    fig = Figure(; size = (980, 410))
    ax1 = Axis(fig[1, 1]; yscale = log10, xlabel = "rank r",
               ylabel = "mean relative L² error", title = "Accuracy vs rank")
    scatterlines!(ax1, collect(ranks), mlp_l2; color = :seagreen, markersize = 10, label = "POD-MLP")
    scatterlines!(ax1, collect(ranks), kan_l2; color = :purple, markersize = 10, label = "POD-KAN")
    axislegend(ax1; position = :rt)
    ax2 = Axis(fig[1, 2]; yscale = log10, xlabel = "rank r",
               ylabel = "mean relative PDE residual", title = "Physics consistency vs rank")
    scatterlines!(ax2, collect(ranks), mlp_res; color = :seagreen, markersize = 10, label = "POD-MLP")
    scatterlines!(ax2, collect(ranks), kan_res; color = :purple, markersize = 10, label = "POD-KAN")
    axislegend(ax2; position = :rt)
    return fig
end

"Basis stress-test: reconstruction error vs rank for POD / sine / random bases."
function basis_comparison_figure(ranks, pod_errs, sine_errs, rand_errs)
    fig = Figure(; size = (560, 430))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "rank r",
              ylabel = "mean relative reconstruction error",
              title = "Which basis is \"right\"?  (test snapshots)")
    scatterlines!(ax, collect(ranks), pod_errs; color = :crimson, markersize = 8, label = "POD (data-optimal)")
    scatterlines!(ax, collect(ranks), sine_errs; color = :dodgerblue, markersize = 8, label = "sine (Laplacian)")
    scatterlines!(ax, collect(ranks), rand_errs; color = :gray, markersize = 8, label = "random orthonormal")
    axislegend(ax; position = :rt)
    return fig
end

"""
    operator_diagram(mode_imgs, xs; r, N)

Schematic of the factorisation `G = R ∘ N`: parameters → coefficients → field, with
a strip of the first POD modes underneath illustrating the reduced basis `Φᵣ`.
`mode_imgs` are full-grid (boundary-padded) mode matrices.
"""
function operator_diagram(mode_imgs, xs; r, N)
    nmodes = length(mode_imgs)
    fig = Figure(; size = (1180, 560))
    ax = Axis(fig[1, 1:nmodes]; title = "Solution operator  G = R ∘ N")
    hidedecorations!(ax); hidespines!(ax)
    limits!(ax, 0, 1, 0, 1)

    boxes = [(0.06, "μ ∈ ℝ²", "parameters", :gray85),
             (0.42, "c ∈ ℝ^$r", "coefficients", :gray85),
             (0.78, "u(·;μ) ∈ ℝ^$N", "PDE solution", (:seagreen, 0.35))]
    w, h, yc = 0.16, 0.26, 0.62
    for (x, top, bot, col) in boxes
        poly!(ax, Rect(x, yc - h/2, w, h); color = col, strokecolor = :gray30, strokewidth = 1.5)
        text!(ax, x + w/2, yc + 0.045; text = top, align = (:center, :center), fontsize = 19, font = :bold)
        text!(ax, x + w/2, yc - 0.055; text = bot, align = (:center, :center), fontsize = 13, color = :gray30)
    end
    arrow!(p, q) = (lines!(ax, [p, q]; color = :black, linewidth = 2.5);
                    scatter!(ax, [q]; marker = :rtriangle, markersize = 16, color = :black))
    arrow!(Point2f(0.22, yc), Point2f(0.42, yc))
    arrow!(Point2f(0.58, yc), Point2f(0.78, yc))
    text!(ax, 0.32, yc + 0.075; text = "N", align = (:center, :center), fontsize = 18, font = :bold, color = :purple)
    text!(ax, 0.32, yc - 0.085; text = "small net\n(MLP / KAN)", align = (:center, :center), fontsize = 12, color = :purple)
    text!(ax, 0.68, yc + 0.075; text = "R", align = (:center, :center), fontsize = 18, font = :bold, color = :crimson)
    text!(ax, 0.68, yc - 0.085; text = "ū + Φᵣ c\n(fixed POD)", align = (:center, :center), fontsize = 12, color = :crimson)
    text!(ax, 0.5, 0.06; text = "learned: 2 → r       reconstruction: r → N        (r ≪ N)",
          align = (:center, :center), fontsize = 13, color = :gray40)

    for (k, M) in enumerate(mode_imgs)
        axk = Axis(fig[2, k]; title = "φ$k", aspect = DataAspect())
        heatmap!(axk, xs, xs, M; colormap = :balance)
        hidedecorations!(axk)
    end
    Label(fig[3, 1:nmodes], "Φᵣ — the reduced basis (first $nmodes of r POD modes)";
          fontsize = 13, color = :gray30)
    rowsize!(fig.layout, 1, Relative(0.6))
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
