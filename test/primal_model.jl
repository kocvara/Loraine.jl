module TestPrimalInterface

using Test, LinearAlgebra
import JuMP
import Loraine
import LowRankOpt as LRO
import NLPModels
import MathOptInterface as MOI

@testset "Primal solver statistics and objective constants" begin
    for maximize in (false, true)
        model = JuMP.Model(Loraine.Optimizer)
        JuMP.set_silent(model)
        JuMP.@variable(model, x)
        if maximize
            JuMP.@constraint(model, x <= 2)
            JuMP.@objective(model, Max, 3x + 7)
        else
            JuMP.@constraint(model, x >= 1)
            JuMP.@objective(model, Min, 3x + 7)
        end
        JuMP.optimize!(model)
        optimizer = JuMP.unsafe_backend(model)
        @test optimizer.model isa LRO.Model
        @test optimizer.model.meta.nvar == 1
        @test JuMP.termination_status(model) == MOI.OPTIMAL
        @test JuMP.value(x) ≈ (maximize ? 2 : 1) atol = 1e-5
        @test JuMP.objective_value(model) ≈ (maximize ? 13 : 10) atol = 1e-5
        @test MOI.get(optimizer, MOI.DualObjectiveValue()) ≈ JuMP.objective_value(model) atol =
            1e-5
        @test optimizer.solver.stats.solution ≈ [JuMP.value(x)]
        expected = maximize ? -JuMP.objective_value(model) : JuMP.objective_value(model)
        @test NLPModels.obj(optimizer.model, optimizer.solver.stats.solution) ≈ expected
        @test optimizer.solver.stats.objective ≈ expected
    end
end

@testset "PSD constraint blocks use ordinary primal variables" begin
    model = JuMP.Model(Loraine.Optimizer)
    JuMP.set_silent(model)
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [t 1; 1 t] in JuMP.PSDCone())
    JuMP.@objective(model, Min, t + 5)
    JuMP.optimize!(model)
    optimizer = JuMP.unsafe_backend(model)
    @test optimizer.model.meta.nvar == 1
    @test optimizer.model.meta.ncon == 4
    @test JuMP.objective_value(model) ≈ 6 atol = 1e-5
    primal = optimizer.solver.stats.solution
    multipliers = optimizer.solver.stats.multipliers
    @test norm(
        NLPModels.grad(optimizer.model, primal) +
        NLPModels.jtprod(optimizer.model, primal, multipliers),
        Inf,
    ) < 1e-5
    @test LRO.dual_model(optimizer.model) isa LRO.DualModel
    @test MOI.get(optimizer, LRO.DualSolution()) isa LRO.AbstractConeVector
end

end
