###########################################################################
#   SUPPORTING FUNCTIONS for Loraine
###########################################################################
# Writes `B * C * A'` into `dest`, using `tmp` (same size as
# `C * A'`) as scratch. No allocation.
function my_kron!(dest, A, B, C, tmp)
    mul!(tmp, C, A')   # tmp = C * A'
    mul!(dest, B, tmp) # dest = B * (C * A')
    return dest
end
###########################################################################
function mat(vecA)
    n = isqrt(length(vecA))
    Atmp = reshape(vecA, n, n)
    return @. (Atmp + Atmp') / 2
end

function mat!(dst::AbstractMatrix, vecA)
    n = isqrt(length(vecA))
    Atmp = reshape(vecA, n, n)
    @. dst = (Atmp + Atmp') / 2
end

###########################################################################


# How to create a vector of sparse matrices:
# v = SparseMatrixCSC{Float64,Int}[
# a=sparse([3 0;0 5])
# push!(v,copy(a))
# a=sparse([7 0;0 5])
# push!(v,copy(a))
# v[2]
