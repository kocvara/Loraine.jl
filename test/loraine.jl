using Test
using JuMP
import MathOptInterface as MOI
import Loraine

const THETA1 = joinpath(dirname(@__DIR__), "examples", "data", "theta1.dat-s")

@testset "loraine" begin
    @testset "direct solver" begin
        model = Loraine.loraine(THETA1, Dict("verb" => 0))
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(model, MOI.ObjectiveValue()) ≈ 23 rtol = 1e-6
    end

    # The options are forwarded as `MOI.RawOptimizerAttribute`s
    @testset "iterative solver" begin
        model = Loraine.loraine(THETA1, Dict("verb" => 0, "kit" => 1))
        @test MOI.get(model, MOI.RawOptimizerAttribute("kit")) == 1
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(model, MOI.ObjectiveValue()) ≈ 23 rtol = 1e-6
    end

    @testset "default options" begin
        model = Loraine.loraine(THETA1)
        @test MOI.get(model, MOI.ObjectiveValue()) ≈ 23 rtol = 1e-6
    end
end

# A 1x1 block has no residual eigenvalues at the default preconditioner rank.
@testset "CG with a 1x1 PSD block" begin
    for preconditioner in (1, 2), aamat in (0, 1), scalar_first in (false, true)
        @testset "preconditioner=$preconditioner aamat=$aamat scalar_first=$scalar_first" begin
            model = Model(Loraine.Optimizer)
            for (key, value) in ("kit" => 1, "verb" => 0, "preconditioner" => preconditioner, "aamat" => aamat)
                set_attribute(model, key, value)
            end
            @variable(model, y[1:3])
            @variable(model, z)
            X = [y[1] -1 -1; -1 y[2] -1; -1 -1 y[3]]
            Z = reshape([z], 1, 1)
            for block in (scalar_first ? (Z, X) : (X, Z))
                @constraint(model, block in PSDCone())
            end
            @objective(model, Min, sum(y) + z)
            optimize!(model)
            @test termination_status(model) == MOI.OPTIMAL
            @test objective_value(model) ≈ 6 atol = 1e-5
        end
    end
end
