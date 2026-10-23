using LinearAlgebra, DynamicPolynomials
using JuMP
using CSDP
using Loraine
using SumOfSquares
using MultiFloats
using Printf

loraine = Loraine.Optimizer{Float64x6}
import Dualization
dual_loraine = Dualization.dual_optimizer(loraine)
solver = optimizer_with_attributes(dual_loraine, MOI.Silent() => false)

# solver = optimizer_with_attributes(Brisk.Optimizer, MOI.Silent() => false)

model = SOSModel(solver)
set_attribute(model, "initpoint", 1)
set_attribute(model, "eDIMACS", 1e-20)
set_attribute(model, "maxit", 300)

# # set_attribute(model, "mfipm", 1)
# set_attribute(model, "prec", 256)
# set_attribute(model, "nopolishr", 0)

@polyvar x y

k = 15
epsi = 1/(2^(k - 1))
# epsi = 0

g1 = x
g2 = 1 - x^2 - y^2
g3 = y
# g4 = 1 - y
# p = x^2 + y^2 + 3 * x * y + epsi * (x + 2 * y)
p = x * y + x^3 + y^3 + epsi * (x + y)
S = @set g1 >= 0 && g2 >= 0 && g3 >= 0 #&& g4 >= 0
d = 14

@variable(model, α)
@objective(model, Max, α)
@constraint(model, c1, p >= α, domain = S, maxdegree = 2*d)
optimize!(model)

solution_summary(model)

# --- effective condition number ---
# Filters out numerically-zero eigenvalues (rank-deficiency at the optimum
# is expected/generic for SOS relaxations) before taking the ratio, so the
# result reflects conditioning on the matrix's effective range space
# rather than being dominated by ~0 (or slightly negative) noise eigenvalues.
function effective_cond(M::AbstractMatrix; reltol = 1e-12)
    ev = eigvals(Symmetric(Matrix(M)))
    λmax = maximum(abs, ev)
    sig = filter(λ -> λ > reltol * λmax, ev)
    isempty(sig) && return Inf
    # @show sig
    return maximum(sig) / minimum(sig)
end

Q = gram_matrix(c1).Q
# evQ = eigvals(Symmetric(Matrix(Q)))
# println("Gram matrix eigenvalues: ", evQ)
println("Gram matrix effective condition number: ", effective_cond(Q))

ν = moment_matrix(c1)
# evν = eigvals(Symmetric(Matrix(ν.Q)))
# println("Moment matrix eigenvalues: ", evν)
println("Moment matrix effective condition number: ", effective_cond(ν.Q))

println("Effective condition number: ", @sprintf("%.2e", effective_cond(ν.Q) * effective_cond(Q)))
println("Effective ratio:            ", @sprintf("%.2e", effective_cond(ν.Q) / effective_cond(Q)))

