# 09 — Launch the interactive live-calibration console (GLMakie).
# Requires an OpenGL-capable display. Run after 03 and at least one trained model
# (05 / 04). With no model files it still runs in "POD projection" mode.

include("config.jl")
using GLMakie
GLMakie.activate!(; title = "Reduced-Basis Surrogate Explorer")
include(joinpath(PROJECT_DIR, "src", "console.jl"))

d   = load_snapshots()
pod = load_pod()
g   = make_grid(d["grid_n"])

# Collect whatever trained models are available into the menu.
models = Dict{String,Any}()
for (label, key) in (("Direct MLP", "direct"), ("POD-MLP", "pod_mlp"))
    isfile(model_path(key)) || continue
    m = load_mlp(model_path(key))
    models[label] = (; m.kind, m.model, m.ps, m.st, m.outdim, nparams = param_count(m.model))
end
if isfile(model_path("pod_kan"))            # P1: load the KAN with its own loader
    include(joinpath(PROJECT_DIR, "src", "models_kan.jl"))
    m = load_kan(model_path("pod_kan"))
    models["POD-KAN"] = (; m.kind, m.model, m.ps, m.st, m.outdim, nparams = param_count(m.model))
end
# (POD projection always works, even with no trained models.)

fig = build_console(g, pod.P, d["mu_train"], models;
                    figdir = FIG_DIR, param_lo = PARAM_LO, param_hi = PARAM_HI)

screen = display(fig)
println("Console open — drag the μ sliders, switch models, move the rank slider.")
println("Close the window to exit.")
wait(screen)
