using LinearAlgebra, DynamicPolynomials
using JuMP
using Loraine
using SumOfSquares
using MultiFloats
# using Base.Threads

# # Plain threaded mul! — no LoopVectorization, just @threads over columns
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

loraine = Loraine.Optimizer{Float64x8}
import Dualization
dual_loraine = Dualization.dual_optimizer(loraine)
solver = optimizer_with_attributes(dual_loraine, MOI.Silent() => false)

model = SOSModel(solver)
set_attribute(model, "eDIMACS", 1e-14)
set_attribute(model, "maxit", 300)
set_attribute(model, "kit", 0)

@polyvar x y u v
p = x^4*y^2*v^2 + y^4*u^2*v^2 + x^2*u^4*v^2 - 3*x^2*y^2*u^2*v^2 + u^8

S = @set x^2 + y^2 + u^2 + v^2 - 1 == 0

d = 4
@variable(model, α)
@objective(model, Max, α)
@constraint(model, c1, p - α in NonnegPolyInnerCone{MOI.PositiveSemidefiniteConeTriangle}(),
            domain = S, maxdegree = 2*d)

optimize!(model)
solution_summary(model)