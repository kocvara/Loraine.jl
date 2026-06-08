using Base.Threads
using LinearAlgebra
using MultiFloats

# Threaded mul! for MultiFloat matrices
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