using LinearAlgebra, DynamicPolynomials
using JuMP
using SumOfSquares
ENV["BRISK_OMP_STATS"] = "1"
using Brisk

solver = optimizer_with_attributes(Brisk.Optimizer, MOI.Silent() => false)
model = SOSModel(solver)
# set_attribute(model, "prec", "qd")
# set_attribute(model, "hptol", 1e-20)
# set_attribute(model, "prec", 500)
# set_attribute(model, MOI.NumberOfThreads(), 8)

@polyvar x y
p = x
S = @set x^3 - y^2 >= 0 && 1 - x^2 - y^2 >= 0
d = 5
@variable(model, α)
@objective(model, Max, α)
@constraint(model, c1, p >= α, domain = S, maxdegree = 2*d)
optimize!(model)
# optimize!(model)