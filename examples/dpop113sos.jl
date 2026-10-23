
# y2(1 − x2)2 + x2(1 − y2)2 + x2y2(x2 + y2 − 2)2 

using LinearAlgebra, DynamicPolynomials
using JuMP
using CSDP
using Loraine
using SumOfSquares
using MultiFloats
using Base.Threads
using Printf

# Plain threaded mul! — no LoopVectorization, just @threads over columns
function LinearAlgebra.mul!(
        C::Matrix{T},
        A::Matrix{T},
        B::Matrix{T}) where {T <: MultiFloats.MultiFloat}
    m, k = size(A)
    _, n = size(B)
    fill!(C, zero(T))
    @threads for j in 1:n
        for l in 1:k
            @inbounds blj = B[l, j]
            for i in 1:m
                @inbounds C[i, j] += A[i, l] * blj
            end
        end
    end
    return C
end

loraine = Loraine.Optimizer{Float64x4}
import Dualization
dual_loraine = Dualization.dual_optimizer(loraine)
solver = optimizer_with_attributes(dual_loraine, MOI.Silent() => false)

# solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => false)

model = SOSModel(solver)
set_attribute(model, "initpoint", 1)
set_attribute(model, "eDIMACS", 1e-20)
set_attribute(model, "maxit", 300)

@polyvar x y

k = 4
epsi = 1/(2^(k - 1))

g1 = y * (x^2 + y^2)
g2 = 1 - x^2 - y^2
p = y + (x - epsi)^2
S = @set g1 >= 0 && g2 >= 0
d = 3

@variable(model, α)
@objective(model, Max, α)
@constraint(model, c1, p >= α, domain = S, maxdegree = 2*d)
optimize!(model)

# solution_summary(model)

# --- effective condition number ---
# Filters out numerically-zero eigenvalues (rank-deficiency at the optimum
# is expected/generic for SOS relaxations) before taking the ratio, so the
# result reflects conditioning on the matrix's effective range space
# rather than being dominated by ~0 (or slightly negative) noise eigenvalues.
function effective_cond(M::AbstractMatrix; reltol = 1e-8)
    ev = eigvals(Symmetric(Matrix(M)))
    λmax = maximum(abs, ev)
    sig = filter(λ -> λ > reltol * λmax, ev)
    isempty(sig) && return Inf
    return maximum(sig) / minimum(sig)
end

Q = gram_matrix(c1).Q
evQ = eigvals(Symmetric(Matrix(Q)))
# println("Gram matrix eigenvalues: ", evQ)
println("Gram matrix effective condition number: ", effective_cond(Q))

ν = moment_matrix(c1)
evν = eigvals(Symmetric(Matrix(ν.Q)))
# println("Moment matrix eigenvalues: ", evν)
println("Moment matrix effective condition number: ", effective_cond(ν.Q))

println("Effective condition number: ", @sprintf("%.2e", effective_cond(ν.Q) * effective_cond(Q)))
println("Effective ratio:            ", @sprintf("%.2e", effective_cond(ν.Q) / effective_cond(Q)))



