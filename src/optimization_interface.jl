# Optimization.jl interface: solve(prob, IPBFGS(); maxiters, maxtime, callback).
# Missing bounds are infinite. Tolerances are fields of IPBFGS; abstol and reltol are ignored.

SciMLBase.has_init(::IPBFGS) = true
SciMLBase.allowsbounds(::IPBFGS) = true
SciMLBase.requiresgradient(::IPBFGS) = true
SciMLBase.allowscallback(::IPBFGS) = true
SciMLBase.allowsfg(::IPBFGS) = true
OptimizationBase.supports_sense(::IPBFGS) = true

function SciMLBase.__init(
        prob::SciMLBase.OptimizationProblem, opt::IPBFGS;
        callback = OptimizationBase.DEFAULT_CALLBACK,
        maxiters::Union{Number, Nothing} = nothing,
        maxtime::Union{Number, Nothing} = nothing,
        abstol::Union{Number, Nothing} = nothing,
        reltol::Union{Number, Nothing} = nothing,
        progress = false,
        kwargs...
    )
    (abstol !== nothing || reltol !== nothing) &&
        @warn "IPBFGS ignores abstol and reltol. Set the tolerances with IPBFGS(; tolfun, tolx)." maxlog = 1
    maxiters = OptimizationBase._check_and_convert_maxiters(maxiters)
    maxtime = OptimizationBase._check_and_convert_maxtime(maxtime)
    return OptimizationBase.OptimizationCache(
        prob, opt; callback, maxiters, maxtime, abstol, reltol, progress, kwargs...
    )
end

const _RETCODES = (
    first_order = SciMLBase.ReturnCode.Success, step = SciMLBase.ReturnCode.Success,
    maxiters = SciMLBase.ReturnCode.MaxIters, maxfevals = SciMLBase.ReturnCode.MaxIters,
    maxtime = SciMLBase.ReturnCode.MaxTime, callback = SciMLBase.ReturnCode.Terminated,
)

function SciMLBase.__solve(cache::OptimizationBase.OptimizationCache{IPBFGS})
    sgn = cache.sense === OptimizationBase.MaxSense ? -1.0 : 1.0
    function fg(x)
        G = similar(x)
        if cache.f.fg === nothing
            f = cache.f.f(x, cache.p)
            cache.f.grad(G, x)
        else
            f = cache.f.fg(G, x)
        end
        return sgn * first(f), sgn .* G
    end
    callback = cache.callback === OptimizationBase.DEFAULT_CALLBACK ? nothing : function (s)
        state = OptimizationBase.OptimizationState(;
            iter = s.iter, u = s.x, objective = sgn * s.f, grad = sgn .* s.g, original = s, p = cache.p
        )
        halt = cache.callback(state, sgn * s.f)
        halt isa Bool ||
            error("The callback should return a boolean `halt` for whether to stop the optimization process.")
        return halt
    end
    n = length(cache.u0)
    lb = cache.lb === nothing ? fill(-Inf, n) : cache.lb
    ub = cache.ub === nothing ? fill(Inf, n) : cache.ub
    maxiters = cache.solver_args.maxiters
    alg = maxiters === nothing ? cache.opt :
        IPBFGS(; (; (k => getfield(cache.opt, k) for k in fieldnames(IPBFGS))..., maxiters)...)
    maxtime = cache.solver_args.maxtime === nothing ? Inf : cache.solver_args.maxtime
    t0 = time()
    r = ipbfgs(fg, cache.u0, lb, ub, alg; callback, maxtime)
    stats = OptimizationBase.OptimizationStats(;
        iterations = r.iterations, time = time() - t0, fevals = r.fevals, gevals = r.fevals
    )
    return SciMLBase.build_solution(
        cache, cache.opt, r.x, sgn * r.f; original = r, retcode = _RETCODES[r.status], stats
    )
end
