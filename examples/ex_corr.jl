using JuMP
import LinearAlgebra
# import SCS
import Loraine
import Test

function example_correlation_problem()
    model = Model(Loraine.Optimizer)
    set_silent(model)
    MOI.set(model, MOI.RawOptimizerAttribute("kit"), 0)
    @variable(model, X[1:3, 1:3], PSD)
    S = ["A", "B", "C"]
    ρ = Containers.DenseAxisArray(X, S, S)
    @constraint(model, [i in S], ρ[i, i] == 1)
    @constraint(model, -0.2 <= ρ["A", "B"] <= -0.1)
    @constraint(model, 0.4 <= ρ["B", "C"] <= 0.5)
    @objective(model, Max, ρ["A", "C"])
    
    optimize!(model)
    println("An upper bound for ρ_AC is $(value(ρ["A", "C"]))")
    @objective(model, Min, ρ["A", "C"])
    optimize!(model)
    println("A lower bound for ρ_AC is $(value(ρ["A", "C"]))")
    return
end

example_correlation_problem()