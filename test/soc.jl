module TestSOC
using Test, LinearAlgebra
import JuMP
import Loraine
import LowRankOpt as LRO
import NLPModels
import MathOptInterface as MOI

@testset "Native SOC interface with explicit Loraine PSD adapter" begin
    for rotated in (false, true)
        m = JuMP.Model(Loraine.Optimizer)
        JuMP.set_silent(m)
        JuMP.@variable(m, t)
        if rotated
            JuMP.@constraint(m, [t, 0.5, 3, 4] in MOI.RotatedSecondOrderCone(4))
        else
            JuMP.@constraint(m, [t, 3, 4] in MOI.SecondOrderCone(3))
        end
        JuMP.@objective(m, Min, t+2)
        JuMP.optimize!(m)
        o = JuMP.unsafe_backend(m)
        @test JuMP.termination_status(m) == MOI.OPTIMAL
        @test JuMP.value(t) ≈ (rotated ? 25 : 5) atol=1e-5
        @test LRO.num_second_order_cones(o.model) == 1
        @test LRO.num_matrices(o.model) == 0
        @test o.model.meta.nvar == 1
        @test norm(
            NLPModels.grad(o.model, o.solver.stats.solution)+NLPModels.jtprod(
                o.model,
                o.solver.stats.solution,
                o.solver.stats.multipliers,
            ),
            Inf,
        ) < 1e-5
        @test MOI.get(o, MOI.DualObjectiveValue()) ≈ JuMP.objective_value(m) atol=1e-5
    end
end
end
