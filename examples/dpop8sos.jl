# min (1-x^2) * (1-y^2) s.t. 1-x^2 >= 0 , 1 - y^2 >= 0

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

@polyvar x y z

g1 = (1 - (0.5 + x)^2 - y^2 - z^2)
g2 = (1 - (0.5 - x)^2 - y^2 - z^2)
p = g1 * g2
S = @set g1 >= 0 && g2 >= 0

d = 4

@variable(model, α)
@objective(model, Max, α)
@constraint(model, c1, p >= α, domain = S, maxdegree = 2*d)
optimize!(model)

solution_summary(model)

# ν3 = moment_matrix(c1)
# atomic_measure(ν3, 1e-3)
