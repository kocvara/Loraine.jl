# min x s.t. x^3 - y^2 >= 0 , 1 - x^2 - y^2 >= 0

using LinearAlgebra, DynamicPolynomials
using JuMP
using Loraine
using SumOfSquares
using MultiFloats
using Base.Threads
# using SCS
# using COSMO
# using DSDP
using CSDP
# using Mosek, MosekTools
# using Hypatia
# using Clarabel


# Plain threaded mul! — no LoopVectorization, just @threads over columns
# function LinearAlgebra.mul!(
#         C::Matrix{T},
#         A::Matrix{T},
#         B::Matrix{T}) where {T <: MultiFloats.MultiFloat}
#     m, k = size(A)
#     _, n = size(B)
#     fill!(C, zero(T))
#     @threads for j in 1:n
#         for l in 1:k
#             @inbounds blj = B[l, j]
#             for i in 1:m
#                 @inbounds C[i, j] += A[i, l] * blj
#             end
#         end
#     end
#     return C
# end
# println("Running with ", nthreads(), " threads")

# loraine = Loraine.Optimizer{Float64x4}
# import Dualization
# dual_loraine = Dualization.dual_optimizer(loraine)
# solver = optimizer_with_attributes(dual_loraine, MOI.Silent() => false)

solver = optimizer_with_attributes(CSDP.Optimizer, MOI.Silent() => false)

model = SOSModel(solver)
# set_attribute(model, "initpoint", 1)
# set_attribute(model, "eDIMACS", 1e-14)
# set_attribute(model, "maxit", 300)

@polyvar x y

p = x
S = @set x^3 - y^2 >= 0 && 1 - x^2 - y^2 >= 0
d = 15
@variable(model, α)
@objective(model, Max, α)
@constraint(model, c1, p >= α, domain = S, maxdegree = 2*d)
optimize!(model)

solution_summary(model)

println("Termination status: ", termination_status(model))
println("Primal status:      ", primal_status(model))
println("Dual status:        ", dual_status(model))
println("Raw status:         ", raw_status(model))

