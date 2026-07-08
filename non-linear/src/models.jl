# Neural surrogate models built on Lux — ported from the parent project.
#
#   • direct MLP    μ ↦ u   (output dimension = n, the whole nodal field)
#   • POD-MLP       μ ↦ c   (output dimension = r, reduced-basis coefficients)
#   • POD-KAN       μ ↦ c   (see models_kan.jl, kept out of the core module)
#
# Training uses full-batch Adam with a mean-squared-error data loss; inputs are
# the standardized parameters in [-1,1]^d.

"""
    make_mlp(indim, outdim; hidden=64, depth=3, σ=tanh, rng) -> (model, ps, st)

Build a fully-connected `tanh` network with `depth` hidden layers of width
`hidden`, and initialise its parameters/state with `rng`.
"""
function make_mlp(indim::Integer, outdim::Integer;
                  hidden::Integer = 64, depth::Integer = 3, σ = tanh,
                  rng::AbstractRNG = Random.default_rng())
    hiddens = (Dense(hidden => hidden, σ) for _ in 1:(depth - 1))
    model = Chain(Dense(indim => hidden, σ), hiddens..., Dense(hidden => outdim))
    ps, st = Lux.setup(rng, model)
    # Float64 throughout to match the POD pipeline and avoid mixed-precision matmuls.
    return model, Lux.f64(ps), Lux.f64(st)
end

"""POD coefficient mapper `μ ∈ ℝ^indim ↦ c ∈ ℝ^r` (steady: indim = 2)."""
build_pod_mlp(r::Integer; indim::Integer = 2, kw...) = make_mlp(indim, r; kw...)

"""Direct field predictor `μ ∈ ℝ^indim ↦ u ∈ ℝ^{ndof}`."""
build_direct_mlp(ndof::Integer; indim::Integer = 2, kw...) = make_mlp(indim, ndof; kw...)

"""Total number of learnable parameters in a Lux model."""
param_count(model) = Lux.parameterlength(model)

"""
    train!(model, ps, st, X, Y; epochs, lr, lr_final, extra, λ, keep_best, rng, …)
        -> (ps, losses)

Full-batch Adam training minimising `mean((model(X) − Y)²)`. `X` is `indim × M`,
`Y` is `outdim × M`.

- The learning rate is annealed geometrically from `lr` to `lr_final` over the run
  (set `lr_final == lr` to disable), which avoids the end-of-training oscillation
  full-batch Adam shows at a fixed rate.
- With `keep_best = true` the parameters with the lowest loss seen are returned,
  a cheap safety net against any residual bounce.
- If `extra` is supplied it is called as `extra(Ŷ, ps)` and added with weight `λ`.
"""
function train!(model, ps, st, X::AbstractMatrix, Y::AbstractMatrix;
                epochs::Integer = 3000, lr::Real = 1e-3, lr_final::Real = 1e-5,
                extra = nothing, λ::Real = 0.0, keep_best::Bool = true,
                rng::AbstractRNG = Random.default_rng(),
                verbose::Bool = true, logevery::Integer = 500)
    opt = Optimisers.setup(Optimisers.Adam(lr), ps)
    γ = epochs > 1 ? (lr_final / lr)^(1 / (epochs - 1)) : 1.0
    lr_t = float(lr)
    losses = Float64[]
    best_loss, best_ps = Inf, ps
    function lossfn(p)
        Ŷ, _ = model(X, p, st)
        l = mean(abs2, Ŷ .- Y)
        extra === nothing || (l += λ * extra(Ŷ, p))
        return l
    end
    for epoch in 1:epochs
        res = Zygote.withgradient(lossfn, ps)
        push!(losses, res.val)
        # snapshot the params that produced this loss *before* stepping past them
        if keep_best && res.val < best_loss
            best_loss, best_ps = res.val, deepcopy(ps)
        end
        opt, ps = Optimisers.update(opt, ps, res.grad[1])
        if γ != 1.0
            lr_t *= γ
            Optimisers.adjust!(opt, lr_t)
        end
        if verbose && (epoch == 1 || epoch % logevery == 0)
            @printf("    epoch %5d / %d   loss %.4e   (lr %.1e)\n",
                    epoch, epochs, res.val, lr_t)
        end
    end
    return (keep_best ? best_ps : ps), losses
end

"""
    predict(model, ps, st, X) -> Ŷ

Deterministic forward pass in evaluation mode (`Lux.testmode`). `X` is `indim × M`.
"""
function predict(model, ps, st, X::AbstractMatrix)
    Ŷ, _ = model(X, ps, Lux.testmode(st))
    return Ŷ
end
predict(model, ps, st, x::AbstractVector) = vec(predict(model, ps, st, reshape(x, :, 1)))
