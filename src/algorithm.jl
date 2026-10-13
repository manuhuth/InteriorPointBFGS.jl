# Primal-dual interior-point method for min f(x) subject to lb ≤ x ≤ ub, after
#
#   R. A. Waltz, J. L. Morales, J. Nocedal, D. Orban. An interior algorithm for nonlinear
#   optimization that combines line search and trust region steps. Mathematical Programming
#   107, 391–408 (2006). doi:10.1007/s10107-004-0560-5
#
# specialised to bound constraints with a BFGS Hessian approximation. The slacks
# sl = x − lb, su = ub − x are eliminated, zl, zu are the bound multipliers, and an infinite
# bound simply drops its slack, multiplier and barrier term.
#
# - Multipliers: least squares, min ‖g − zl + zu‖² + ‖Sl zl − μ‖² + ‖Su zu − μ‖², decoupled
#   per component. Initially z ≤ 0 → μ/s and then z ≥ μ; after a trust-region step only
#   z ≤ 0 → μ/s.
# - Direct step: (B + Zl/Sl + Zu/Su) dx = −(g − μ/sl + μ/su) with the primal-dual dz,
#   fraction to the boundary τ separately for x and z. If min(αx, αz) < δ the direct step is
#   skipped. Line search on the merit φ = f − μ Σ log s, Armijo condition, at most `maxbt`
#   backtracks with a common factor t for x and z (halving; the first reduction after a
#   trust-region iteration is min(1/2, Δ/‖step‖_D)).
# - Trust-region step when the line search fails: Steihaug CG on the same model in the
#   scaled norm ‖p‖_D, D² = 1 + 1/sl² + 1/su², scaled back to the fraction to the boundary.
#   ρ = actual / predicted merit reduction; a rejected trial sets Δ = ‖p‖_D/2.
# - Radius: Δ0 = √n. Δ = 2‖step‖_D after a direct step; after an accepted trust-region step
#   Δ = max(7‖p‖_D, Δ) for ρ ≥ 0.9, max(2‖p‖_D, Δ) for ρ ≥ 0.3, else unchanged.
# - Hessian: damped BFGS (Powell, 0.2), B0 = I.
# - Barrier parameter: the barrier problem counts as solved when
#   max(‖∇L‖∞, ‖S z − μ‖∞)/max(1, ‖g‖∞) ≤ μ (at iteration 0 also with the raw least-squares
#   multipliers). Then μ /= 100 if it took ≤ 2 iterations, else μ /= 5, and Δ = max(5Δ, 1),
#   repeated while the test holds.
# - Stop: max(‖∇L‖∞, maxᵢ min(sᵢzᵢ, sᵢ, zᵢ)) ≤ tolfun·max(1, ‖g‖∞) (:first_order), a trial
#   step with maxᵢ |Δxᵢ|/max(1, |xᵢ|) < tolx (:step), the iteration, evaluation or time budget,
#   or the callback.

"""
    IPBFGS(; kwargs...)

Primal-dual interior-point method for bound-constrained minimization with a BFGS Hessian
approximation, combining line-search and trust-region steps (Waltz, Morales, Nocedal and Orban,
Mathematical Programming 107, 2006). Use it directly with [`ipbfgs`](@ref) or as an
Optimization.jl algorithm, `solve(prob, IPBFGS())`.

Keyword arguments (defaults in parentheses):
- `μ0` (0.1): initial barrier parameter.
- `τ` (0.995): fraction-to-the-boundary parameter.
- `armijo` (1e-8): sufficient-decrease constant of the line search.
- `maxbt` (3): maximum number of backtracks before the trust-region step.
- `δ` (1e-5): smallest step length to the boundary for which a direct step is tried.
- `ηtr` (1e-8): acceptance threshold of a trust-region step (actual/predicted reduction).
- `tolfun` (1e-6): first-order optimality tolerance, relative to max(1, ‖∇f‖∞).
- `tolx` (1e-10): step tolerance, maxᵢ |Δxᵢ|/max(1, |xᵢ|).
- `maxiters` (1000): maximum number of iterations.
- `maxfevals` (35000): maximum number of objective evaluations.
"""
Base.@kwdef struct IPBFGS
    μ0::Float64 = 0.1
    τ::Float64 = 0.995
    armijo::Float64 = 1.0e-8
    maxbt::Int = 3
    δ::Float64 = 1.0e-5
    ηtr::Float64 = 1.0e-8
    tolfun::Float64 = 1.0e-6
    tolx::Float64 = 1.0e-10
    maxiters::Int = 1000
    maxfevals::Int = 35000
end

# Least-squares bound multipliers, one 2×2 system per component (fewer for infinite bounds).
function ls_multipliers(g, sl, su, hl, hu, μ)
    zl, zu = zero(g), zero(g)
    for i in eachindex(g)
        a, b = sl[i], su[i]
        if hl[i] && hu[i]
            D = a^2 + b^2 + a^2 * b^2
            rl, ru = g[i] + μ * a, -g[i] + μ * b
            zl[i] = ((1 + b^2) * rl + ru) / D
            zu[i] = (rl + (1 + a^2) * ru) / D
        elseif hl[i]
            zl[i] = (g[i] + μ * a) / (1 + a^2)
        elseif hu[i]
            zu[i] = (-g[i] + μ * b) / (1 + b^2)
        end
    end
    return zl, zu
end

# Steihaug CG for min q'p + p'Mp/2 subject to p'D2 p ≤ Δ².
function steihaug(M, q, D2, Δ; tol = 1.0e-2, tolabs = 1.0e-10, maxit = 2length(q))
    p = zero(q)
    r = -q
    z = r ./ D2
    d = copy(z)
    rz = dot(r, z)
    rz0 = rz
    function toboundary(p, d)
        A = dot(d, D2 .* d)
        B = 2dot(p, D2 .* d)
        C = dot(p, D2 .* p) - Δ^2
        return p .+ ((-B + sqrt(B^2 - 4A * C)) / 2A) .* d
    end
    for _ in 1:maxit
        Md = M * d
        κ = dot(d, Md)
        κ <= 0 && return toboundary(p, d)
        α = rz / κ
        pn = p .+ α .* d
        dot(pn, D2 .* pn) >= Δ^2 && return toboundary(p, d)
        p = pn
        r .-= α .* Md
        z = r ./ D2
        rzn = dot(r, z)
        (sqrt(rzn) <= tol * sqrt(rz0) || sqrt(rzn) <= tolabs) && return p
        d = z .+ (rzn / rz) .* d
        rz = rzn
    end
    return p
end

# Largest α ≤ 1 with v + α dv ≥ (1 − τ) v for v > 0 (an infinite v never binds).
ftb(v, dv, τ) = minimum((-τ * v[i] / dv[i] for i in eachindex(v) if dv[i] < 0); init = 1.0)

# Σ log s over the finite bounds, as one array sum (the same rounding as without masks).
logsum(s, h) = all(h) ? sum(log, s) : sum(log, s[h]; init = 0.0)
# ‖v‖∞ over the finite bounds.
maxabs(v, h) = all(h) ? norm(v, Inf) : maximum(abs, v[h]; init = 0.0)
# Complementarity measure maxᵢ min(sᵢzᵢ, sᵢ, zᵢ) over the finite bounds.
complementarity(s, z, h) = maximum((min(s[i] * z[i], s[i], z[i]) for i in eachindex(s) if h[i]); init = 0.0)

"""
    ipbfgs(fg, x0, lb, ub, alg = IPBFGS(); callback = nothing, maxtime = Inf)

Minimize f over lb ≤ x ≤ ub, where `fg(x)` returns `(f, ∇f)`, from `x0` strictly inside the
bounds. Bounds may be infinite. A non-finite f rejects the trial point. `callback(state)` is
called once per iteration with `state = (; iter, fevals, x, f, g, optimality, μ, Δ, tr_step)`
and stops the run when it returns `true`; `false` or `nothing` continue. Returning `:refresh`
tells the method that the objective has changed at the current `x` (for example an inner
problem of the objective was re-solved): `fg(x)` is evaluated once more, counted in `fevals`,
and the iteration continues with the new value and gradient. The BFGS matrix, the multipliers,
μ and Δ are kept.

Returns `(; x, f, g, iterations, fevals, status, zl, zu)` with `status` one of
`:first_order`, `:step`, `:maxiters`, `:maxfevals`, `:maxtime`, `:callback`.
"""
function ipbfgs(fg, x0, lb, ub, alg::IPBFGS = IPBFGS(); callback = nothing, maxtime = Inf)
    o = alg
    n = length(x0)
    length(lb) == n && length(ub) == n || throw(DimensionMismatch("x0, lb and ub must have the same length."))
    all(lb .< x0 .< ub) || throw(ArgumentError("x0 must lie strictly inside the bounds, lb < x0 < ub."))
    t_start = time()
    hl, hu = isfinite.(lb), isfinite.(ub)
    x = float.(collect(x0))
    nf = 0
    f, g = fg(x)
    nf += 1
    isfinite(f) || throw(ArgumentError("the objective is not finite at x0."))
    μ, Δ = o.μ0, sqrt(n)
    B = Matrix{Float64}(I, n, n)
    sl, su = x .- lb, ub .- x
    zl, zu = ls_multipliers(g, sl, su, hl, hu, μ)
    zl0, zu0 = copy(zl), copy(zu)                 # raw estimate, used in the first barrier test
    zl .= ifelse.(hl, max.(ifelse.(zl .<= 0, μ ./ sl, zl), μ), 0.0)
    zu .= ifelse.(hu, max.(ifelse.(zu .<= 0, μ ./ su, zu), μ), 0.0)
    it, nsub, lasttr, status, tr_step = 0, 0, false, :maxiters, false
    rel(dx) = maximum(abs.(dx) ./ max.(1, abs.(x)))
    while true
        gL = g .- zl .+ zu
        sc = max(1, norm(g, Inf))
        opt = max(norm(gL, Inf), complementarity(sl, zl, hl), complementarity(su, zu, hu))
        if callback !== nothing
            cb = callback((; iter = it, fevals = nf, x, f, g, optimality = opt, μ, Δ, tr_step))
            if cb === true
                status = :callback
                break
            elseif cb === :refresh                   # the objective changed at x
                f, g = fg(x)
                nf += 1
                isfinite(f) || throw(ArgumentError("the objective is not finite at x after a callback refresh."))
                gL = g .- zl .+ zu
                sc = max(1, norm(g, Inf))
                opt = max(norm(gL, Inf), complementarity(sl, zl, hl), complementarity(su, zu, hu))
            elseif !(cb === false || cb === nothing)
                throw(ArgumentError("the callback must return true, false, nothing or :refresh; got $(repr(cb))."))
            end
        end
        opt <= o.tolfun * sc && (status = :first_order; break)
        it >= o.maxiters && (status = :maxiters; break)
        nf >= o.maxfevals && (status = :maxfevals; break)
        time() - t_start >= maxtime && (status = :maxtime; break)
        E(zl, zu, μ) = max(norm(g .- zl .+ zu, Inf), maxabs(sl .* zl .- μ, hl), maxabs(su .* zu .- μ, hu)) / sc
        while (it == 0 ? min(E(zl, zu, μ), E(zl0, zu0, μ)) : E(zl, zu, μ)) <= μ
            μ /= nsub <= 2 ? 100 : 5
            Δ = max(5Δ, 1.0)
            nsub = 0
        end
        it += 1
        nsub += 1
        Σ = zl ./ sl .+ zu ./ su
        q = g .- μ ./ sl .+ μ ./ su
        M = B + Diagonal(Σ)
        dx = -(Symmetric(M) \ q)                      # Bunch-Kaufman LDLᵀ
        dzl = μ ./ sl .- zl .- zl ./ sl .* dx
        dzu = μ ./ su .- zu .+ zu ./ su .* dx
        αx = min(ftb(sl, dx, o.τ), ftb(su, -dx, o.τ))
        αz = min(ftb(zl, dzl, o.τ), ftb(zu, dzu, o.τ))
        φ(fv, xv) = all(lb .< xv .< ub) ? fv - μ * (logsum(xv .- lb, hl) + logsum(ub .- xv, hu)) : Inf
        φ0 = f - μ * (logsum(sl, hl) + logsum(su, hu))
        Dφ = dot(q, dx)
        normD(p) = sqrt(sum(p .^ 2 .* (1 .+ 1 ./ sl .^ 2 .+ 1 ./ su .^ 2)))
        direct = min(αx, αz) >= o.δ
        t, xn, fn, gn = 1.0, x, f, g
        stop = false
        if direct
            for bt in 0:o.maxbt
                xn = x .+ (t * αx) .* dx
                fn, gn = fg(xn)
                nf += 1
                φ(fn, xn) <= φ0 + o.armijo * t * αx * Dφ && break      # false for NaN/Inf
                nf >= o.maxfevals && (stop = true; status = :maxfevals; break)
                bt == o.maxbt && (direct = false; break)
                t = bt == 0 && lasttr ? min(0.5, Δ / normD(t * αx .* dx)) : t / 2
            end
        end
        if !direct && !stop                          # trust-region step
            D2 = 1 .+ 1 ./ sl .^ 2 .+ 1 ./ su .^ 2
            while true
                p = steihaug(M, q, D2, Δ)
                p .*= min(ftb(sl, p, o.τ), ftb(su, -p, o.τ))
                xn = x .+ p
                fn, gn = fg(xn)
                nf += 1
                ρ = (φ0 - φ(fn, xn)) / -(dot(q, p) + dot(p, M * p) / 2)
                if ρ > o.ηtr
                    Δ = ρ >= 0.9 ? max(7normD(p), Δ) : ρ >= 0.3 ? max(2normD(p), Δ) : Δ
                    break
                end
                Δ = normD(p) / 2
                if rel(p) < o.tolx
                    stop, status = true, :step
                    break
                elseif nf >= o.maxfevals
                    stop, status = true, :maxfevals
                    break
                end
            end
        end
        stop && break
        s = xn .- x
        tr_step = !direct
        direct && (Δ = 2normD(s))
        lasttr = tr_step
        if direct
            zl .+= (t * αz) .* dzl
            zu .+= (t * αz) .* dzu
        else
            sln, sun = xn .- lb, ub .- xn
            zl, zu = ls_multipliers(gn, sln, sun, hl, hu, μ)
            zl .= ifelse.(zl .<= 0, μ ./ sln, zl)
            zu .= ifelse.(zu .<= 0, μ ./ sun, zu)
        end
        y = gn .- g
        Bs = B * s
        sBs = dot(s, Bs)
        sy = dot(s, y)
        if sy < 0.2sBs
            θ = 0.8sBs / (sBs - sy)
            y = θ .* y .+ (1 - θ) .* Bs
            sy = dot(s, y)
        end
        sBs > eps() && sy > eps() && (B .+= y * y' ./ sy .- Bs * Bs' ./ sBs)
        tolstop = rel(s) < o.tolx
        x, f, g = xn, fn, gn
        sl, su = x .- lb, ub .- x
        tolstop && (status = :step; break)
    end
    return (; x, f, g, iterations = it, fevals = nf, status, zl, zu)
end
