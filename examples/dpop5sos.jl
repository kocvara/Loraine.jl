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
set_attribute(model, "eDIMACS", 1e-14)
set_attribute(model, "maxit", 300)

@polyvar x y

p = (1 - x) * (1 - y)
S = @set 1 - x >= 0 && 1 - y >= 0 && 1 + x >= 0 && 1 + y >= 0 
d = 13

@variable(model, α)
@objective(model, Max, α)
@constraint(model, c1, p >= α, domain = S, maxdegree = 2*d)
optimize!(model)

solution_summary(model)


