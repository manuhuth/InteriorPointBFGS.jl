using InteriorPointBFGS, Optimization, ForwardDiff, LinearAlgebra, Test

rosen(x) = 100(x[2] - x[1]^2)^2 + (1 - x[1])^2
rosen_fg(x) = (rosen(x), [-400x[1] * (x[2] - x[1]^2) - 2(1 - x[1]), 200(x[2] - x[1]^2)])

@testset "InteriorPointBFGS" begin
    @testset "bounded quadratic with active bounds" begin
        c = [2.0, -3.0, 0.5]
        r = ipbfgs(x -> (sum(abs2, x .- c), 2(x .- c)), zeros(3), -ones(3), ones(3))
        @test r.status == :first_order
        @test r.x ≈ [1.0, -1.0, 0.5] atol = 1.0e-6
        # KKT: g = zl − zu, zl, zu ≥ 0, multipliers of the active bounds 2|xᵢ − cᵢ|
        @test all(r.zl .>= 0) && all(r.zu .>= 0)
        @test r.g ≈ r.zl .- r.zu atol = 1.0e-5
        @test r.zu[1] ≈ 2 atol = 1.0e-5
        @test r.zl[2] ≈ 4 atol = 1.0e-5
        @test r.zl[3] < 1.0e-5 && r.zu[3] < 1.0e-5
        @test maximum((r.x .+ 1) .* r.zl) < 1.0e-5 && maximum((1 .- r.x) .* r.zu) < 1.0e-5
    end

    @testset "Rosenbrock" begin
        r = ipbfgs(rosen_fg, [-1.2, 1.0], [-2.0, -2.0], [2.0, 2.0])
        @test r.status == :first_order
        @test r.x ≈ [1.0, 1.0] atol = 1.0e-5
        r = ipbfgs(rosen_fg, [-1.2, 1.0], [-2.0, -2.0], [0.8, 2.0])
        @test r.status == :first_order
        @test r.x ≈ [0.8, 0.64] atol = 1.0e-5
        @test r.f ≈ 0.04 atol = 1.0e-6
        @test r.zu[1] > 0
    end

    @testset "one-sided and unbounded" begin
        r = ipbfgs(rosen_fg, [-1.2, 1.0], fill(-Inf, 2), fill(Inf, 2))
        @test r.x ≈ [1.0, 1.0] atol = 1.0e-6
        @test r.zl == zeros(2) && r.zu == zeros(2)
        r = ipbfgs(x -> ((x[1] - 2)^2, [2(x[1] - 2)]), [5.0], [3.0], [Inf])
        @test r.x[1] ≈ 3 atol = 1.0e-6
        @test r.zl[1] ≈ 2 atol = 1.0e-5
        r = ipbfgs(x -> ((x[1] - 2)^2, [2(x[1] - 2)]), [0.0], [-Inf], [1.0])
        @test r.x[1] ≈ 1 atol = 1.0e-6
        c = [2.0, -2.0, 0.5]
        r = ipbfgs(x -> (sum(abs2, x .- c), 2(x .- c)), zeros(3), [-Inf, -1, -Inf], [1.0, Inf, Inf])
        @test r.x ≈ [1.0, -1.0, 0.5] atol = 1.0e-5
    end

    @testset "stopping" begin
        calls = Ref(0)
        r = ipbfgs(rosen_fg, [-1.2, 1.0], [-2.0, -2.0], [2.0, 2.0];
            callback = s -> (calls[] += 1; s.iter >= 3))
        @test r.status == :callback && r.iterations == 3 && calls[] == 4
        r = ipbfgs(rosen_fg, [-1.2, 1.0], [-2.0, -2.0], [2.0, 2.0], IPBFGS(maxiters = 5))
        @test r.status == :maxiters && r.iterations == 5
        r = ipbfgs(rosen_fg, [-1.2, 1.0], [-2.0, -2.0], [2.0, 2.0], IPBFGS(maxfevals = 7))
        @test r.status == :maxfevals && r.fevals == 7
        @test_throws ArgumentError ipbfgs(rosen_fg, [3.0, 0.0], [-2.0, -2.0], [2.0, 2.0])
        @test_throws ArgumentError ipbfgs(rosen_fg, [2.0, 0.0], [-2.0, -2.0], [2.0, 2.0])
        @test_throws DimensionMismatch ipbfgs(rosen_fg, [0.0, 0.0], [-2.0], [2.0, 2.0])
    end

    @testset "Optimization.jl" begin
        f(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
        p = [1.0, 100.0]
        prob = OptimizationProblem(OptimizationFunction(f, AutoForwardDiff()), [-1.2, 1.0], p;
            lb = [-2.0, -2.0], ub = [0.8, 2.0])
        sol = solve(prob, IPBFGS())
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [0.8, 0.64] atol = 1.0e-5
        @test sol.objective ≈ 0.04 atol = 1.0e-6
        @test sol.original.status == :first_order
        @test sol.stats.iterations == sol.original.iterations && sol.stats.fevals == sol.original.fevals

        # no bounds, user gradient
        grad(G, x, p) = (G .= rosen_fg(x)[2])
        sol = solve(OptimizationProblem(OptimizationFunction(f; grad), [-1.2, 1.0], p), IPBFGS())
        @test sol.retcode == ReturnCode.Success
        @test sol.u ≈ [1.0, 1.0] atol = 1.0e-6

        sol = solve(prob, IPBFGS(); maxiters = 4)
        @test sol.retcode == ReturnCode.MaxIters && sol.stats.iterations == 4

        iters = Int[]
        sol = solve(prob, IPBFGS(); callback = (s, l) -> (push!(iters, s.iter); s.iter >= 2))
        @test sol.retcode == ReturnCode.Terminated && iters == [0, 1, 2]
        @test_throws ErrorException solve(prob, IPBFGS(); callback = (s, l) -> nothing)

        # maximize −(x − c)² over a box
        g(x, p) = -sum(abs2, x .- p)
        sol = solve(OptimizationProblem(OptimizationFunction(g, AutoForwardDiff()), zeros(2), [2.0, 0.3];
            lb = -ones(2), ub = ones(2), sense = MaxSense), IPBFGS())
        @test sol.u ≈ [1.0, 0.3] atol = 1.0e-6
        @test sol.objective ≈ -1 atol = 1.0e-6
    end
end
