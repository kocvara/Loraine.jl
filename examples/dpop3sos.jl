# min 1 - x^2 s.t. (1-x^2)^3 >= 0

using LinearAlgebra, DynamicPolynomials
using JuMP
using CSDP
using Loraine
using SumOfSquares
using MultiFloats
using Mosek, MosekTools

# loraine = Loraine.Optimizer{Float64x4}
# import Dualization
# dual_loraine = Dualization.dual_optimizer(loraine)
# solver = optimizer_with_attributes(dual_loraine, MOI.Silent() => false)

solver = optimizer_with_attributes(Mosek.Optimizer, MOI.Silent() => false)

model = SOSModel(solver)
# set_attribute(model, "initpoint", 1)
# set_attribute(model, "eDIMACS", 1e-16)
# set_attribute(model, "maxit", 300)

@polyvar x

p = 1-x^2
S = @set (1-x^2)^3 >= 0
d = 8

@variable(model, α)
@objective(model, Max, α)
@constraint(model, c1, p >= α, domain = S, maxdegree = 2*d)
optimize!(model)

Q = gram_matrix(c1).Q
println(eigvals(Symmetric(Matrix(Q))))

ν = moment_matrix(c1)
println(eigvals(Symmetric(Matrix(ν.Q))))

