using LinearAlgebra, DynamicPolynomials, SemialgebraicSets
using JuMP
using SumOfSquares
using Brisk

solver = optimizer_with_attributes(Brisk.Optimizer, MOI.Silent() => false)
model = SOSModel(solver)
# set_attribute(model, "prec", "qp")
set_attribute(model, "prec", 256)

@polyvar x y
p = x
S = @set x^3 - y^2 >= 0 && 1 - x^2 - y^2 >= 0
d = 6
@variable(model, α)
@objective(model, Max, α)
@constraint(model, c1, p >= α, domain = S, maxdegree = 2*d)
optimize!(model)