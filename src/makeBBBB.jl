function makeBBBB_rank1(n,nlmi,B,G)
    tmp = zeros(Float64, n, n)
    BBBB = zeros(Float64, n, n)
    for ilmi = 1:nlmi
        BB = transpose(B[ilmi] * G[ilmi])
        mul!(tmp,BB',BB)
        if ilmi == 1
            BBBB = tmp .^ 2
        else
            BBBB += tmp .^ 2
        end
    end
    return BBBB
end

#########################

function makeBBBBs(n,nlmi,A,AA,W,qA,sigmaA)
    BBBB = zeros(Float64, n, n)
    @inbounds for ilmi = 1:nlmi
        Wilmi = W[ilmi]
        AAilmi = AA[ilmi]
        Ailmi = A[ilmi,:]
        BBBB += makeBBBBsi(ilmi,Ailmi,AAilmi,Wilmi,n,qA,sigmaA)
    end

    return BBBB
end

# Computes `⟨A * W, W * B⟩` for symmetric sparse matrices `A` and `B`
function _dot(A::SparseMatrixCSC, B::SparseMatrixCSC, W::Matrix)
    @assert LinearAlgebra.checksquare(W) == LinearAlgebra.checksquare(A) == LinearAlgebra.checksquare(B)
    # After these asserts, we know that `A`, `B` and `W` are square and
    # have the same sizes so we can safely use `@inbounds`
    result = zero(eltype(A))
    @inbounds for i in axes(A, 2)
        nzA = nzrange(A, i)
        if !isempty(nzA)
            for j in axes(B, 2)
                nzB = nzrange(B, j)
                if !isempty(nzB)
                    AW = zero(result)
                    for k in nzA
                        AW += nonzeros(A)[k] * W[rowvals(A)[k], j]
                    end
                    WB = zero(result)
                    for k in nzB
                        WB += W[i, rowvals(B)[k]] * nonzeros(B)[k]
                    end
                    result += AW * WB
                end
            end
        end
    end
    return result
end

#####
function makeBBBBsi(ilmi,Ailmi,AAilmi,Wilmi::Matrix{T},n,qA,sigmaA) where {T}
    BBBB = zeros(T, n, n)
    tmp1 = Matrix{T}(undef,size(Wilmi, 2), size(Ailmi[1], 1))
    # tmp = Matrix{Float64}(undef,size(Wilmi, 1), size(Wilmi, 1))
    tmp2 = Matrix{T}(undef,size(AAilmi, 1), 1)
    tmp3 = Vector{Float64}(undef,size(Wilmi, 1))
    ilmi1 = (ilmi-1)*n

    @inbounds for ii = 1:n
        # tmp1 = zeros(Float64,size(Wilmi, 2), size(Ailmi[1], 1))
        i = sigmaA[ii,ilmi]
        if nnz(Ailmi[i+1]) > 0
            if ii <= qA[1,ilmi]
                tmp  = zeros(T,size(Wilmi, 2), size(Ailmi[1], 1))
                mul!(tmp1,Wilmi,Ailmi[i+1])
                tmp = tmp1 * Wilmi
                tmp2 = AAilmi * vec(tmp)
                indi = sigmaA[ii:end,ilmi]
                BBBB[indi,i] .= -tmp2[indi]
                BBBB[i,indi] .= -tmp2[indi]
            else
                if !iszero(nnz(Ailmi[i+1]))
                    if nnz(Ailmi[i+1]) > 1
                        @inbounds for jj = ii:n
                            j = sigmaA[jj,ilmi]
                            if !iszero(nnz(Ailmi[j+1]))
                                ttt = _dot(Ailmi[i+1], Ailmi[j+1], Wilmi)
                                if i >= j
                                    BBBB[i,j] = ttt
                                else
                                    BBBB[j,i] = ttt
                                end
                            end  
                        end   
                    else
                        # A is symmetric
                        iiiiAi = jjjiAi = only(rowvals(Ailmi[i+1]))
                        vvvi = only(nonzeros(Ailmi[i+1]))
                        @inbounds for jj = ii:n
                            j = sigmaA[jj,ilmi]
                            Ajjj = Ailmi[j+1]
                            # As we sort the matrices in decreasing `nnz` order,
                            # the rest of matrices is either zero or have only
                            # one entry
                            if !iszero(nnz(Ajjj))
                                iiijAj = jjjjAj = only(rowvals(Ajjj))
                                vvvj = only(nonzeros(Ajjj))
                                ttt = vvvi * Wilmi[iiiiAi,iiijAj] * Wilmi[jjjiAi,jjjjAj] * vvvj
                                if i >= j
                                    BBBB[i,j] = ttt
                                else
                                    BBBB[j,i] = ttt
                                end
                            end
                        end 
                    end   
                end
            end
        end
    end
    return BBBB
end
