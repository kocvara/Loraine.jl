###########################################################################
#   SUPPORTING FUNCTIONS for Loraine
###########################################################################
function my_kron(A::Matrix{T}, B, C) where {T}
    # TMP = B * C * A'
    TMP1 = Matrix{T}(undef,size(C,1),size(A,1))
    TMP = Matrix{T}(undef,size(B,1),size(C,1))
    mul!(TMP1,C,A')
    mul!(TMP,B,TMP1)
    return TMP
end
###########################################################################
function mat(vecA)
    n = isqrt(length(vecA))
    Atmp = reshape(vecA, n, n)
    return (Atmp + Atmp') ./ 2
    # return Hermitian(reshape(vecA, n, n))
end

###########################################################################


# How to create a vector of sparse matrices:
# v = SparseMatrixCSC{Float64,Int}[
# a=sparse([3 0;0 5])
# push!(v,copy(a))
# a=sparse([7 0;0 5])
# push!(v,copy(a))
# v[2]
