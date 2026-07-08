# Prints every headline table in report-markdown form, straight from
# data/*.jld2. Diff this output against report/report.md whenever numbers are
# touched; frozen numbers can then never silently drift (the parent project's
# freeze lesson, institutionalized).

include(joinpath(@__DIR__, "config.jl"))

println("=== 18_report_numbers ===\n")

need(f) = isfile(f) || (println("missing $(basename(f)); run its script first\n"); false)

if need(eval_path())
    ev = load(eval_path())
    println("## §8.1 steady evaluation (deployed r = $(DEFAULT_R))")
    @printf("POD floor: rel L² %.1e, residual %.1e; FOM residual floor %.1e; FOM %.0f ms\n",
            ev["floor_rel"], ev["floor_resid"], ev["fom_resid"], ev["fom_ms"])
    for name in ("direct_mlp", "pod_mlp", "pod_kan")
        r = ev["results"][name]
        @printf("| %s | %.2e | %.2e | %s | %.1e | %.1f | %d |\n", name,
                r[:rel], r[:resid], isnan(r[:coef]) ? "—" : @sprintf("%.1e", r[:coef]),
                r[:dP], r[:dT], r[:params])
    end
    sw = ev["sweep"]
    println("rank sweep:")
    for (i, r) in enumerate(sw["ranks"])
        @printf("| %d | %.2e / %.2e | %.2e / %.2e |\n", r,
                sw["mlp_rel"][i], sw["mlp_resid"][i], sw["kan_rel"][i], sw["kan_resid"][i])
    end
    println()
end

if need(joinpath(DATA_DIR, "benchmark.jld2"))
    b = load(joinpath(DATA_DIR, "benchmark.jld2"))
    println("## §8.2 benchmark (batch = $(b["batch"]), FOM $(round(b["t_fom_ms"], digits=1)) ms)")
    for (i, nm) in enumerate(b["names"])
        @printf("| %s | %.3f | %.4f | %.0f× | %.0f× | %d |\n", nm,
                b["t1"][i], b["tb"][i], b["s1"][i], b["sb"][i], b["allocs"][i])
    end
    println()
end

if need(eval_t_path())
    et = load(eval_t_path())
    println("## §8.4 transient sweep (deployed r = $(et["deployed_r"]))")
    for (i, r) in enumerate(et["ranks"])
        @printf("| %d | %.2e | %.1e |\n", r, et["sweep_rel"][i], et["floor_rel"][i])
    end
    @printf("peak-T err: median %.1f K, max %.1f K; P_rad tracking: median %.1e, max %.1e\n\n",
            median(et["dTpk"]), maximum(et["dTpk"]), median(et["dPq"]), maximum(et["dPq"]))
end

if need(joinpath(DATA_DIR, "robustness.jld2"))
    rb = load(joinpath(DATA_DIR, "robustness.jld2"))
    println("## §8.7 robustness")
    @printf("POD-MLP  r=%d: deployed %.2e, 5-seed [%.2e, %.2e]\n", DEFAULT_R,
            rb["dep_mlp"], minimum(rb["mlp_rels"]), maximum(rb["mlp_rels"]))
    @printf("POD-KAN  r=%d: deployed %.2e, 5-seed [%.2e, %.2e]\n", DEFAULT_R,
            rb["dep_kan"], minimum(rb["kan_rels"]), maximum(rb["kan_rels"]))
    @printf("POD-MLP-t r=%d: deployed %.2e, 5-seed [%.2e, %.2e]\n", DEFAULT_R_T,
            rb["dep_t"], minimum(rb["t_rels"]), maximum(rb["t_rels"]))
    @printf("OOD: in-box median %.1e; outside median %.1e, worst %.1e\n\n",
            median(rb["in_rel"]), median(rb["ood_rel"]), maximum(rb["ood_rel"]))
end

if need(joinpath(DATA_DIR, "grid_independence.jld2"))
    g = load(joinpath(DATA_DIR, "grid_independence.jld2"))
    println("## §5 grid independence (σ₃/σ₁)")
    @printf("60×30 P1 %.1e | 120×60 P1 %.1e | 60×30 P2 %.1e\n\n",
            g["d_base"][3], g["d_fine"][3], g["d_p2"][3])
end

if need(joinpath(DATA_DIR, "design_study.jld2"))
    ds = load(joinpath(DATA_DIR, "design_study.jld2"))
    println("## §8.6 design studies")
    for op in ds["ops"]
        @printf("Qp=%.0f, per=%.0f: %d cands, FOM %.0f s vs sur %.2f s (%.0f×), Spearman %.4f, top-5 %d/5, QoI err med %.1f K max %.1f K\n",
                op[:Qp], op[:period], length(op[:fomQ]), op[:t_fom], op[:t_sur],
                op[:t_fom] / op[:t_sur], op[:spearman], op[:top5],
                median(abs.(op[:surQ] .- op[:fomQ])),
                maximum(abs.(op[:surQ] .- op[:fomQ])))
    end
    println()
end

if need(joinpath(DATA_DIR, "mapper_diagnostics.jld2"))
    md_ = load(joinpath(DATA_DIR, "mapper_diagnostics.jld2"))
    println("## §8.4 learning curve")
    for (M, e) in zip(md_["sizes"], md_["lc"])
        @printf("| %d trajectories | %.2e |\n", M, e)
    end
    @printf("slopes: %s\n", join((@sprintf("%.2f", s) for s in md_["slopes"]), "  "))
end
