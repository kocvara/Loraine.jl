# min (1-x) * (1-y) 
#   s.t. 1 - x >= 0 , 1 - y >= 0 , 1 + x >= 0 , 1 + y >= 0

using LinearAlgebra, DynamicPolynomials
using JuMP
using CSDP
using Loraine
using SumOfSquares
using MultiFloats
using Base.Threads

# Plain threaded mul! — no LoopVectorization, just @threads over columns

loraine = Loraine.Optimizer{Float64x4}
import Dualization
dual_loraine = Dualization.dual_optimizer(loraine)
solver = optimizer_with_attributes(dual_loraine, MOI.Silent() => false)

# solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => false)

model = SOSModel(solver)
set_attribute(model, "initpoint", 1)
set_attribute(model, "eDIMACS", 1e-14)
set_attribute(model, "maxit", 300)

n = 3
@polyvar x[1:n]

p = prod(x)
S = @set x[1] * (1 - x[1]) >= 0
for i in 2:n
    add_inequality!(S, x[i] * (1 - x[i]))
end
# S = @set x[1] * (1 - x[1]) >= 0 && x[2] * (1 - x[2]) >= 0 
# S = @set x[1] * (1 - x[1]) >= 0 && x[2] * (1 - x[2]) >= 0 && x[3] * (1 - x[3]) >= 0 && x[4] * (1 - x[4]) >= 0 
# S = @set x[1] * (1 - x[1]) >= 0 && x[2] * (1 - x[2]) >= 0 && x[3] * (1 - x[3]) >= 0 && x[4] * (1 - x[4]) >= 0 && x[5] * (1 - x[5]) >= 0 && x[6] * (1 - x[6]) >= 0 
# S = @set x[1] * (1 - x[1]) >= 0 && x[2] * (1 - x[2]) >= 0 && x[3] * (1 - x[3]) >= 0 && x[4] * (1 - x[4]) >= 0 && x[5] * (1 - x[5]) >= 0 && x[6] * (1 - x[6]) >= 0 && x[7] * (1 - x[7]) >= 0 && x[8] * (1 - x[8]) >= 0 
d = 10

@variable(model, α)
@objective(model, Max, α)
@constraint(model, c1, p >= α, domain = S, maxdegree = 2*d)
optimize!(model)

solution_summary(model)


