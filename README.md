# InteriorPointBFGS.jl

[![CI](https://github.com/manuhuth/InteriorPointBFGS.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/manuhuth/InteriorPointBFGS.jl/actions/workflows/CI.yml)

InteriorPointBFGS.jl minimizes a smooth function subject to bound constraints `lb ≤ x ≤ ub` with a primal-dual interior-point method. The Hessian of the objective is approximated by a dense damped BFGS matrix. Bounds may be infinite, so one-sided and unconstrained problems are covered as well. The method can be called directly or as an algorithm of [Optimization.jl](https://github.com/SciML/Optimization.jl).

## Algorithm

The method follows the interior-point algorithm of Waltz, Morales, Nocedal and Orban (2006), specialised to bound constraints. Each iteration first computes a primal-dual Newton step for the barrier problem with barrier parameter μ and accepts it after an Armijo backtracking line search on the merit function `f(x) − μ Σ log s`, where `s` are the distances to the finite bounds. If the step to the boundary is too short or the line search fails after `maxbt` backtracks, a trust-region step is computed instead by Steihaug conjugate gradients in a norm scaled by the distances to the bounds. The bound multipliers are updated along the primal-dual step after a line-search step and by least squares after a trust-region step. The barrier parameter is decreased once the barrier problem is solved to accuracy μ. The iteration stops when the first-order optimality measure falls below `tolfun · max(1, ‖∇f‖∞)` or a step becomes shorter than `tolx`.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/manuhuth/InteriorPointBFGS.jl")
```

## Usage

### Direct call

`ipbfgs(fg, x0, lb, ub, alg = IPBFGS(); callback = nothing, maxtime = Inf)` takes a function `fg(x)` that returns the objective value and its gradient. The start point must lie strictly inside the bounds.

```julia
using InteriorPointBFGS

rosen(x) = (100(x[2] - x[1]^2)^2 + (1 - x[1])^2,
            [-400x[1] * (x[2] - x[1]^2) - 2(1 - x[1]), 200(x[2] - x[1]^2)])

r = ipbfgs(rosen, [-1.2, 1.0], [-2.0, -2.0], [0.8, Inf])
r.x       # [0.8, 0.64]
r.status  # :first_order
```

The result is a named tuple with fields `x`, `f`, `g`, `iterations`, `fevals`, `status`, `zl` and `zu`. The vectors `zl` and `zu` hold the multipliers of the lower and upper bounds. The status is one of `:first_order`, `:step`, `:maxiters`, `:maxfevals`, `:maxtime` and `:callback`. The callback is called once per iteration with a named tuple `(; iter, fevals, x, f, g, optimality, μ, Δ, tr_step)` and stops the run when it returns `true`. It returns `:refresh` when the objective has changed at the current point, for example after an inner problem of the objective was re-solved. The method then evaluates `fg(x)` once more and continues with the new value and gradient, keeping the BFGS matrix, the multipliers, μ and Δ.

### Optimization.jl

```julia
using InteriorPointBFGS, Optimization, ForwardDiff

f(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
optf = OptimizationFunction(f, AutoForwardDiff())
prob = OptimizationProblem(optf, [-1.2, 1.0], [1.0, 100.0]; lb = [-2.0, -2.0], ub = [0.8, 2.0])
sol = solve(prob, IPBFGS(); maxiters = 500)
```

The method needs gradients, either from an AD backend or from a `grad` function in `OptimizationFunction`. Missing `lb` or `ub` are treated as infinite bounds. `MaxSense` problems are solved by negating the objective. The common arguments `maxiters`, `maxtime` and `callback` are supported (the callback returns a boolean or `:refresh`), and `maxiters` overrides the field of `IPBFGS`. The tolerances are fields of `IPBFGS`, and the common arguments `abstol` and `reltol` are ignored. The return codes are `Success` for `:first_order` and `:step`, `MaxIters` for `:maxiters` and `:maxfevals`, `MaxTime` for `:maxtime` and `Terminated` for `:callback`. The result tuple of `ipbfgs` is stored in `sol.original`.

## Options

| Field | Default | Meaning |
|---|---|---|
| `μ0` | `0.1` | initial barrier parameter |
| `τ` | `0.995` | fraction-to-the-boundary parameter |
| `armijo` | `1e-8` | sufficient-decrease constant of the line search |
| `maxbt` | `3` | maximum number of backtracks before a trust-region step |
| `δ` | `1e-5` | smallest step length to the boundary for which a line-search step is tried |
| `ηtr` | `1e-8` | acceptance threshold for the ratio of actual to predicted reduction of a trust-region step |
| `tolfun` | `1e-6` | first-order optimality tolerance, relative to `max(1, ‖∇f‖∞)` |
| `tolx` | `1e-10` | step tolerance on `maximum(abs.(Δx) ./ max.(1, abs.(x)))` |
| `maxiters` | `1000` | maximum number of iterations |
| `maxfevals` | `35000` | maximum number of objective evaluations |

Options are set as keyword arguments, for example `IPBFGS(tolfun = 1e-8, maxiters = 200)`.

## Reference

R. A. Waltz, J. L. Morales, J. Nocedal, D. Orban. An interior algorithm for nonlinear optimization that combines line search and trust region steps. *Mathematical Programming* 107, 391–408 (2006). [doi:10.1007/s10107-004-0560-5](https://doi.org/10.1007/s10107-004-0560-5)
