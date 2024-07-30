###########################################################################
#   SUPPORTING FUNCTIONS for Loraine
###########################################################################
function my_kron(A, B, C)
    # TMP = B * C * A'
    TMP1 = Matrix{Float64x8}(undef,size(C,1),size(A,1))
    TMP = Matrix{Float64x8}(undef,size(B,1),size(C,1))
    mul!(TMP1,C,A')
    mul!(TMP,B,TMP1)
    return vec(TMP) 
end
###########################################################################
function mat(vecA)
    n = isqrt(length(vecA))
    Atmp = reshape(vecA, n, n)
    return (Atmp + Atmp') ./ 2
    # return Hermitian(reshape(vecA, n, n))
end

###########################################################################
function btrace(nlmi, X, S)
    # compute sum of traces of products of block matrices
    trXS = 0
    @inbounds for i = 1:nlmi
        trXS += sum(sum(X[i] .* S[i]))
    end
    return trXS
end



# How to create a vector of sparse matrices:
# v = SparseMatrixCSC{Float64}[
# a=sparse([3 0;0 5])
# push!(v,copy(a))
# a=sparse([7 0;0 5])
# push!(v,copy(a))
# v[2]





