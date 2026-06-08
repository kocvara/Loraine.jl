function makeBBBB_rank1(n,nlmi,B,G,to)
    @timeit to "BBBB_rank1" begin
    BBBB = zeros(Float64, n, n)
    tmp = zeros(Float64, n, n)
    for ilmi = 1:nlmi
        BB = transpose(B[ilmi] * G[ilmi])
        mul!(tmp, BB', BB)
        @. BBBB += tmp * tmp
    end
    end
    return BBBB
end

#########################

function makeBBBBs(n,nlmi,A,AA,W,to,qA,sigmaA)
    T = eltype(eltype(W))
    BBBB = zeros(T, n, n)
    @inbounds for ilmi = 1:nlmi
        Wilmi = W[ilmi]
        AAilmi = AA[ilmi]
        Ailmi = A[ilmi,:]
        @timeit to "BBBBs" begin
        makeBBBBsi!(BBBB,ilmi,Ailmi,AAilmi,Wilmi,n,to,qA,sigmaA)
        end
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
function makeBBBBsi!(BBBB,ilmi,Ailmi,AAilmi,Wilmi::Matrix{T},n,to,qA,sigmaA) where {T}
    m = size(Wilmi, 1)
    tmp1 = Matrix{T}(undef, m, m)
    tmp_mat = Matrix{T}(undef, m, m)
    tmp2 = Vector{T}(undef, size(AAilmi, 1))

    @inbounds for ii = 1:n
        i = sigmaA[ii,ilmi]
        if nnz(Ailmi[i+1]) > 0
            if ii <= qA[1,ilmi]
                @timeit to "BBBBone" begin
                    @timeit to "BBBBone1" begin
                        mul!(tmp1,Wilmi,Ailmi[i+1])
                    end
                    @timeit to "BBBBone2" begin
                        mul!(tmp_mat,tmp1,Wilmi)
                    end
                    @timeit to "BBBBone3" begin
                        mul!(tmp2,AAilmi,vec(tmp_mat))
                    end
                    @timeit to "BBBBone4" begin
                        indi = @view sigmaA[ii:end,ilmi]
                        @inbounds for k in indi
                            v = -tmp2[k]
                            BBBB[k,i] += v
                            if k != i
                                BBBB[i,k] += v
                            end
                        end
                    end
                end
            elseif 1==0
                @timeit to "BBBBtwo" begin
                mul!(tmp1,Ailmi[i+1],Wilmi)
                @inbounds for jj = ii:n
                    j = sigmaA[jj,ilmi]
                    Ajjj = Ailmi[j+1]
                    if !iszero(nnz(Ajjj))
                        ttt = 0.0
                        @inbounds for jjjjAj in axes(Ajjj, 2)
                            for k in nzrange(Ajjj, jjjjAj)
                                iiijAj = rowvals(Ajjj)[k]
                                ttt1 = dot(tmp1[:,iiijAj],Wilmi[:,jjjjAj])
                                ttt += ttt1 * nonzeros(Ajjj)[k]
                            end
                        end
                        BBBB[i,j] += ttt
                        if !=(i,j)
                            BBBB[j,i] += ttt
                        end
                    end
                end
                end
            else
                @timeit to "BBBBthree" begin
                if !iszero(nnz(Ailmi[i+1]))
                    if nnz(Ailmi[i+1]) > 1
                        @inbounds for jj = ii:n
                            j = sigmaA[jj,ilmi]
                            if !iszero(nnz(Ailmi[j+1]))
                                ttt = _dot(Ailmi[i+1], Ailmi[j+1], Wilmi)
                                if i >= j
                                    BBBB[i,j] += ttt
                                else
                                    BBBB[j,i] += ttt
                                end
                            end
                        end
                    else
                        @timeit to "BBBBthree=1" begin
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
                                    BBBB[i,j] += ttt
                                else
                                    BBBB[j,i] += ttt
                                end
                            end
                        end
                        end
                    end
                end
                end
            end
        end
    end
end


function makeRHS(nlmi,AA,W,S,Rp,Rd)
    h = copy(Rp)
    for i = 1:nlmi
        tmp = W[i]*(Rd[i]+S[i])*W[i]
        mul!(h, AA[i], vec(tmp), true, true)
    end
    return h
end
