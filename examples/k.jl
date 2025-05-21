using JuMP
import MathOptInterface as MOI
import Loraine
# import SCS
using MultiFloats

# model = Model(Loraine.Optimizer)
model = Model(Loraine.Optimizer{Float64x2})
# model = JuMP.GenericModel{Float64x2}(Loraine.Optimizer{Float64x2})

# @variable(model, x >= 0)
# @variable(model, 0 <= y <= 3)
# @objective(model, Min, 12x + 20y)
# @constraint(model, c1, 6x + 8y >= 100)
# @constraint(model, c2, 7x + 12y >= 120)
@variable(model, x)
@objective(model, Max, 2*x)
# @constraint(model, csdp, [x 0; 0  x] - [1 0 ; 0 1] >= 0, PSDCone())
@constraint(model, c1, x >= 1)
@constraint(model, c2, x <= 2)

print(model)

optimize!(model)

solution_summary(model)

using Test
@test termination_status(model) == MOI.OPTIMAL
@test primal_status(model) == MOI.FEASIBLE_POINT
@test dual_status(model) == MOI.FEASIBLE_POINT
@test objective_value(model) ≈ 4 rtol = 1e-6

@test value(x) ≈ 2 rtol = 1e-6

@test shadow_price(c1) ≈ 0 atol = 1e-6

@test shadow_price(c2) ≈ 2 rtol = 1e-6
