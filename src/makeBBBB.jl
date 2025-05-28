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

function makeBBBBs(model::MyModel{T}, W) where {T}
    n = num_constraints(model)
    BBBB = zeros(T, n, n)
    for mat_idx in matrix_indices(model)
        BBBB += makeBBBBsi(model, mat_idx, W[mat_idx.value])
    end
    return BBBB
end

# Computes `⟨A * W, W * B⟩` for symmetric sparse matrices `A` and `B`
function _dot(A::SparseMatrixCSC, B::SparseMatrixCSC, W::AbstractMatrix)
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
function makeBBBBsi(model, mat_idx, Wilmi::AbstractMatrix{T}) where {T}
    ilmi = mat_idx.value
    n = num_constraints(model)
    BBBB = zeros(T, n, n)
    dim = side_dimension(model, mat_idx)
    tmp1 = Matrix{T}(undef, size(Wilmi, 2), dim)
    # tmp = Matrix{Float64}(undef,size(Wilmi, 1), size(Wilmi, 1))
    tmp3 = Vector{T}(undef, size(Wilmi, 1))

    for ii = 1:n
        i = model.sigmaA[ii,ilmi]
        Ai = model.A[ilmi, i]
        if nnz(Ai) > 0
            if ii <= model.qA[1,ilmi]
                tmp  = zeros(T, size(Wilmi, 2), dim)
                mul!(tmp1, Wilmi, Ai)
                tmp = tmp1 * Wilmi # TODO mul!
                tmp2 = jprod(model, mat_idx, tmp)
                indi = model.sigmaA[ii:end,ilmi]
                BBBB[indi,i] .= -tmp2[indi]
                BBBB[i,indi] .= -tmp2[indi]
            else
                if !iszero(nnz(Ai))
                    if nnz(Ai) > 1
                        @inbounds for jj = ii:n
                            j = model.sigmaA[jj,ilmi]
                            Aj = model.A[ilmi, j]
                            if !iszero(nnz(Aj))
                                ttt = _dot(Ai, Aj, Wilmi)
                                if i >= j
                                    BBBB[i,j] = ttt
                                else
                                    BBBB[j,i] = ttt
                                end
                            end  
                        end   
                    else
                        # A is symmetric
                        iiiiAi = jjjiAi = only(rowvals(Ai))
                        vvvi = only(nonzeros(Ai))
                        @inbounds for jj = ii:n
                            j = model.sigmaA[jj,ilmi]
                            Ajjj = model.A[ilmi, j]
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
