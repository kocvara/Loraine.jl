# min x * y s.t. -x^2-y^2+1 >= 0 , x >= 0 , y >= 0

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

p = x * y
S = @set -x^2-y^2+1 >= 0 && x >= 0 && y >= 0
d = 17

@variable(model, α)
@objective(model, Max, α)
@constraint(model, c1, p >= α, domain = S, maxdegree = 2*d)
optimize!(model)

solution_summary(model)

