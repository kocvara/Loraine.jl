model = Model(Loraine.Optimizer)
@variable(model, x)
@constraint(model, con_ref, x <= 2)
@objective(model, Max, 3x)
optimize!(model)
@test value(x) ≈ 2 rtol = 1e-6
@test objective_value(model) ≈ 6 rtol = 1e-6
@test termination_status(model) == MOI.OPTIMAL
@test primal_status(model) == MOI.FEASIBLE_POINT
@test dual_status(model) == MOI.FEASIBLE_POINT
@test dual(con_ref) ≈ -3 rtol = 1e-6

model = Model(Loraine.Optimizer)
@variable(model, x)
@variable(model, y)
@constraint(model, cx, x >= 0)
@constraint(model, cy, y <= 0)
@objective(model, Min, x - y)
optimize!(model)
@test value(x) ≈ 0 atol = 1e-6
@test value(y) ≈ 0 atol = 1e-6
@test objective_value(model) ≈ 0 atol = 1e-6
@test termination_status(model) == MOI.OPTIMAL
@test primal_status(model) == MOI.FEASIBLE_POINT
@test dual_status(model) == MOI.FEASIBLE_POINT
@test dual(cx) ≈ 1 rtol = 1e-6
@test dual(cy) ≈ -1 rtol = 1e-6
