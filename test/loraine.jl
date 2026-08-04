using Test
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
