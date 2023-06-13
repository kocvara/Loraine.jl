using JuMP
import MathOptInterface as MOI
import Loraine


model = Model(Loraine.Optimizer)
# @variable(model, x >= 0)
# @variable(model, 0 <= y <= 3)
# @objective(model, Min, 12x + 20y)
# @constraint(model, c1, 6x + 8y >= 100)
# @constraint(model, c2, 7x + 12y >= 120)
@variable(model, x)
@objective(model, Min, x)
@constraint(model, csdp, [x 0; 0  x] - [1 0 ; 0 1] >= 0, PSDCone())
# @constraint(model, c1, x - 1 >= 0)

print(model)

optimize!(model)

# solution_summary(model)

termination_status(model)

# primal_status(model)

# dual_status(model)

# objective_value(model)

# value(x)

# value(y)

# shadow_price(c1)

# shadow_price(c2)