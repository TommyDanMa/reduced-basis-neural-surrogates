# One-command reproduction: runs scripts 01–18 in order, each in a fresh Julia
# process (no cross-script global state), with per-script timing. Full
# regeneration takes on the order of 2 hours, dominated by the transient
# dataset (08) and the two design-study FOM sweeps (13).
#
#   julia --project=non-linear non-linear/scripts/00_run_all.jl

const SCRIPTS_DIR = @__DIR__
const PROJ = normpath(joinpath(SCRIPTS_DIR, ".."))

scripts = [
    "01_validate_steady.jl",
    "02_generate_steady_dataset.jl",
    "03_compute_pod.jl",
    "04_train_steady_mappers.jl",
    "05_evaluate_steady.jl",
    "06_benchmark_steady.jl",
    "07_validate_transient.jl",
    "08_generate_transient_dataset.jl",
    "09_compute_spacetime_pod.jl",
    "10_train_transient_mapper.jl",
    "11_evaluate_transient.jl",
    "12_nr_study.jl",
    "13_design_study.jl",
    "14_run_console.jl",           # CONSOLE_SMOKE=1: headless render
    "15_robustness.jl",
    "16_grid_independence.jl",
    "17_mapper_diagnostics.jl",
    "18_report_numbers.jl",
]

t_total = time()
for s in scripts
    path = joinpath(SCRIPTS_DIR, s)
    cmd = addenv(`$(Base.julia_cmd()) --project=$(PROJ) $(path)`,
                 "CONSOLE_SMOKE" => "1")
    print("→ $s ... ")
    t = @elapsed run(cmd)
    println("done ($(round(t; digits = 1)) s)")
end
println("all scripts completed in $(round((time() - t_total) / 60; digits = 1)) min")
