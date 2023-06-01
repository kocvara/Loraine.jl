using Test, LinearAlgebra
using JuMP, Loraine

function test()
    model = Model(Loraine.Optimizer)
    @variable(model, x)
    @objective(model, Max, x)
    @constraint(model, Symmetric([1 x; x 1]) in PSDCone())
    optimize!(model)
    #return solution_summary(model), value(x)
    return value(x)
end

test()
