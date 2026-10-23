
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

loraine = Loraine.Optimizer{Float64x8}
import Dualization
dual_loraine = Dualization.dual_optimizer(loraine)
solver = optimizer_with_attributes(dual_loraine, MOI.Silent() => false)

# solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => false)

model = SOSModel(solver)
set_attribute(model, "initpoint", 1)
set_attribute(model, "eDIMACS", 1e-20)
set_attribute(model, "maxit", 300)

@polyvar x y

k = 1
epsi = 1/(2^(k - 1))
epsi = 0

g1 = x
g2 = 1 - x^2 - y^2
# g1 = x - 1
# g2 = y * x
# g3 = 10 - x^2 - y^2
# p = x^2 + (y + epsi)^2 
p = x^2 + y^2 + 3 * x * y + epsi * (x - y)
S = @set g1 >= 0 && g2 >= 0
# S = @set g1 >= 0 && g2 >= 0 && g3 >= 0
d = 2

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
println("Gram matrix eigenvalues: ", evQ)
println("Gram matrix effective condition number: ", effective_cond(Q))

ν = moment_matrix(c1)
evν = eigvals(Symmetric(Matrix(ν.Q)))
println("Moment matrix eigenvalues: ", evν)
println("Moment matrix effective condition number: ", effective_cond(ν.Q))

println("Effective condition number: ", @sprintf("%.2e", effective_cond(ν.Q) * effective_cond(Q)))
println("Effective ratio:            ", @sprintf("%.2e", effective_cond(ν.Q) / effective_cond(Q)))

# Try to decompose it into a (weighted) sum of Dirac measures (the minimizers)
atoms = atomic_measure(ν, 1e-4)   # tolerance for the rank/flatness test

if atoms === nothing
    println("moment matrix is not (numerically) low-rank at this tolerance; ",
            "try a looser/tighter tol or a larger maxdegree")
else
    iy = findfirst(isequal(y), atoms.variables)  # don't assume [x, y] order
    ix = findfirst(isequal(x), atoms.variables)
    for atom in atoms.atoms
        println("weight = ", atom.weight,
                "  x = ", atom.center[ix],
                "  y = ", atom.center[iy])
    end
end

