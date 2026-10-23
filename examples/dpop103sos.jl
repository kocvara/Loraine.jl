
# y2(1 − x2)2 + x2(1 − y2)2 + x2y2(x2 + y2 − 2)2 

using LinearAlgebra, DynamicPolynomials
using JuMP
using CSDP
using Loraine
using SumOfSquares
using MultiFloats

loraine = Loraine.Optimizer{Float64x4}
import Dualization
dual_loraine = Dualization.dual_optimizer(loraine)
solver = optimizer_with_attributes(dual_loraine, MOI.Silent() => false)

# solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => false)

model = SOSModel(solver)
set_attribute(model, "initpoint", 1)
set_attribute(model, "eDIMACS", 1e-14)
set_attribute(model, "maxit", 300)

@polyvar x y

k = 6
epsi = 1/(2^(k - 1))

xe = x / epsi
ye = y / epsi
Meps =  xe^4 * ye^2 + xe^2 * ye^4 + 1 - 3 * xe^2 * ye^2
p = Meps
S = @set 1 - x^2 >= 0 && 1 - y^2 >= 0 && 1 - x^2 - y^2 >= 0
d = 17

@variable(model, α)
@objective(model, Max, α)
@constraint(model, c1, p >= α, domain = S, maxdegree = 2*d)
optimize!(model)

solution_summary(model)

