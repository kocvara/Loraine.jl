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

function buffer_for_schur_complement(model::MyModel, κ)
    n = num_constraints(model)
    σ = zeros(Int64, n, num_matrices(model))
    last_dense = zeros(Int64, num_matrices(model))

    for mat_idx in matrix_indices(model)
        i = mat_idx.value
        nzA = [nnz(model.A[i, j]) for j in 1:n]
        σ[:,i] = sortperm(nzA, rev = true)
        sorted = nzA[σ[:,i]]

        last_dense[i] = something(findlast(Base.Fix1(isless, κ), sorted), 0)
    end

    return σ, last_dense
end

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

function schur_complement(buffer, model::MyModel, W, ::Type{MatrixIndex})
    n = num_constraints(model)
    BBBB = zeros(eltype(eltype(W)), n, n)
    for mat_idx in matrix_indices(model)
        BBBB += schur_complement(buffer, model, mat_idx, W[mat_idx.value])
    end
    return BBBB
end

#####
function schur_complement(buffer, model, mat_idx, W::AbstractMatrix{T}) where {T}
    σ, last_dense = buffer
    ilmi = mat_idx.value
    n = num_constraints(model)
    BBBB = zeros(T, n, n)
    dim = side_dimension(model, mat_idx)
    @assert dim == size(W, 1) == size(W, 2)
    tmp1 = Matrix{T}(undef, size(W, 2), dim)
    tmp  = zeros(T, size(W, 2), dim)
    tmp3 = Vector{T}(undef, size(W, 1))

    for ii = 1:n
        i = σ[ii,ilmi]
        Ai = model.A[ilmi, i]
        if nnz(Ai) > 0
            if ii <= last_dense[ilmi]
                mul!(tmp1, W, Ai)
                mul!(tmp, tmp1, W)
                tmp2 = jprod(model, mat_idx, tmp)
                indi = σ[ii:end,ilmi]
                BBBB[indi,i] .= -tmp2[indi]
                BBBB[i,indi] .= -tmp2[indi]
            else
                if !iszero(nnz(Ai))
                    if nnz(Ai) > 1
                        @inbounds for jj = ii:n
                            j = σ[jj,ilmi]
                            Aj = model.A[ilmi, j]
                            if !iszero(nnz(Aj))
                                ttt = _dot(Ai, Aj, W)
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
                            j = σ[jj,ilmi]
                            Ajjj = model.A[ilmi, j]
                            # As we sort the matrices in decreasing `nnz` order,
                            # the rest of matrices is either zero or have only
                            # one entry
                            if !iszero(nnz(Ajjj))
                                iiijAj = jjjjAj = only(rowvals(Ajjj))
                                vvvj = only(nonzeros(Ajjj))
                                ttt = vvvi * W[iiiiAi,iiijAj] * W[jjjiAi,jjjjAj] * vvvj
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

# [HKS24, (5b)]
# Returns the matrix equal to the sum, for each equation, of
# ⟨A_i, WA_jW⟩
function schur_complement(buffer, model::MyModel, w, W::AbstractVector)
    H = MA.Zero()
    if num_matrices(model) > 0
        H = MA.add!!(H, schur_complement(buffer, model, W, MatrixIndex))
    end
    if num_scalars(model) > 0
        H = MA.add!!(H, schur_complement(model, w, ScalarIndex))
    end
    if H isa MA.Zero
        n = num_constraints(model)
        H = zeros(eltype(w), n, n)
    end
    return Hermitian(H, :L)
end

function schur_complement(model::MyModel, w, ::Type{ScalarIndex})
    return model.C_lin * spdiagm(w) * model.C_lin'
end

# [HKS24, (5b)]
# Returns the matrix equal to the sum, for each equation, of
# ⟨A_i, WA(y)W⟩
function eval_schur_complement!(buffer, result, model::MyModel, w, W, y)
    result .= 0.0
    for mat_idx in matrix_indices(model)
        i = mat_idx.value
        result .-= jprod(model, mat_idx, W[i] * jtprod!(buffer[i], model, mat_idx, y) * W[i])
    end
    result .+= model.C_lin * (w .* (model.C_lin' * y))
    return result
end
