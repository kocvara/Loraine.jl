export prepare_model_data, MyModel

using SparseArrays
using Printf
using TimerOutputs
using LinearAlgebra
import MathOptInterface as MOI

"""
    MyModel

Model representing the problem:
```math
\\begin{aligned}
\\max {} & b^\\top y - b_\\text{const}
\\\\
& \\sum_{j=1}^n y_j A_{i,j} \\preceq C_i
\\qquad
\\forall i \\in \\{1,\\ldots,\\text{nlmi}\\}
\\\\
& C_\\text{lin}^\\top y \\le d_\\text{lin}
\\end{aligned}
```
The fields of the `struct` as related to the arrays of the above formulation as follows:

* The ``i``th PSD constraint is of size `msize[i] × msisze[i]`
* The matrix ``C_i`` is given by `C[i]` which should be equal to `-A[i,1]`.
* The matrix ``A_{i,j}`` is given by `-A[i,j+1]` as well as `myA[(i-1)*n + j]`.
* The vectorization `vec(A[i,j+1])` is also given by `-AA[i][:,j]`
* If `datarank == -1`, ``A_{i,j}`` is also equal to `-B[i][j,:] * B[i][j,:]'`.
* The matrix ``A_{i,j}`` has `nzA[j,i]` nonzero entries
* The index `j = sigmaA[k,i]` is the `k`th matrix ``A_{i,j}`` of the largest number of nonzeros.
* The first `qA[1,i] = qA[2,i]` matrices are considered as dense in the computation.
"""
mutable struct MyModel
    A::Matrix{SparseArrays.SparseMatrixCSC{Float64,Int}}
    AA::Vector{SparseArrays.SparseMatrixCSC{Float64,Int}}
    B::Vector{SparseArrays.SparseMatrixCSC{Float64,Int}}
    C::Vector{SparseArrays.SparseMatrixCSC{Float64,Int}}
    nzA::Matrix{Int64}
    sigmaA::Matrix{Int64}
    qA::Matrix{Int64}
    b::Vector{Float64}
    b_const::Float64
    d_lin::SparseArrays.SparseVector{Float64, Int64}
    C_lin::SparseArrays.SparseMatrixCSC{Float64, Int64}
    n::Int64
    msizes::Vector{Int64}
    nlin::Int64
    nlmi::Int64

    function MyModel(
        A::Matrix{SparseArrays.SparseMatrixCSC{Float64,Int}},
        AA::Vector{SparseArrays.SparseMatrixCSC{Float64,Int}},
        B::Vector{SparseArrays.SparseMatrixCSC{Float64,Int}},
        C::Vector{SparseArrays.SparseMatrixCSC{Float64,Int}},
        nzA::Matrix{Int64},
        sigmaA::Matrix{Int64},
        qA::Matrix{Int64},
        b::Vector{Float64},
        b_const::Float64,
        d_lin::SparseArrays.SparseVector{Float64, Int64},
        C_lin::SparseArrays.SparseMatrixCSC{Float64, Int64},
        n::Int64,
        msizes::Vector{Int64},
        nlin::Int64,
        nlmi::Int64
        ) 

        model = new()
        model.A = A
        model.AA = AA
        model.B = B
        model.C = C
        model.nzA = nzA
        model.sigmaA = sigmaA
        model.qA = qA
        model.b = b
        model.b_const = b_const
        model.d_lin = d_lin
        model.C_lin = C_lin
        model.n = n
        model.msizes = msizes
        model.nlin = nlin
        model.nlmi = nlmi
        return model
    end
end


function prepare_model_data(d,drank)

msizes = Vector{Int64}
n = Int64(get(d, "nvar", 1));
msizesa = get(d, "msizes", 1)
if length(msizesa) == 1
    msizes = [convert.(Int64,msizesa)]
else
    msizes = convert.(Int64,msizesa[:])
end
nlin = Int64(get(d, "nlin", 1))
nlmi = Int64(get(d, "nlmi", 1))
A = get(d, "A", 1);
b = -get(d, "c", 1);
b_const = -get(d, "b_const", 1);

if nlin > 0
    d_lin = -get(d, "d", 1)
    d_lin = d_lin[:]
    C_lin = -get(d, "C", 1)
else
    d_lin = sparse([0.; 0.])
    C_lin = sparse([0. 0.;0. 0.])
end

model = MyModel(A, _prepare_A(A,drank,κ)..., b, b_const, d_lin, C_lin, n, msizes, nlin, nlmi)

return model
end

function _prepare_A(A, datarank, κ)

    nlmi = size(A, 1)
    n = size(A, 2) - 1
    AA = SparseMatrixCSC{Float64,Int}[]
    B = SparseMatrixCSC{Float64,Int}[]
    C = SparseMatrixCSC{Float64,Int}[]
    nzA = zeros(Int64,n,nlmi)
    sigmaA = zeros(Int64,n,nlmi)
    qA = zeros(Int64,2,nlmi)

    for i = 1:nlmi
        
        push!(C, copy(-A[i, 1]))

        Ai = A[i,:]
        m = size(Ai,1)
        AAA = prep_AA!(Ai,n)
        push!(AA, copy(AAA'))

        if datarank == -1
            Btmp = prep_B(A,n,i)
            push!(B, Btmp)
        end

        prep_sparse!(A,n,m,i,nzA,sigmaA,qA,κ)

    end

    return AA, B, C, nzA, sigmaA, qA
end


function prep_sparse!(A,n,m,i,nzA,sigmaA,qA,κ)
    # Simplified data sparsity handling

    for j = 1:n
        nzA[j,i] = nnz(A[i,j+1])
    end
    sigmaA[:,i] = sortperm(nzA[:,i], rev = true)
    sisi = nzA[sigmaA[:,i],i]
    # @show sisi

    qA[1,i] = n
    kappa = κ
    for j = 1:n
        if sisi[j] <= kappa
            qA[1,i] = j-1
            break
        end
    end
    qA[2,i] = qA[1,i]

    # @show qA
end

function prep_B(A,n,i)
    m = size(A[i, 1],1)
    Btmp = spzeros(n,m)

    for k = 1:n
        ii = rowvals(A[i, k + 1])
        bidx = unique(ii)
        if !isempty(bidx)
            tmp = Matrix(A[i, k + 1][bidx, bidx])
            # utmp, vtmp = eigen(Hermitian(tmp))
            utmp, vtmp = eigen((tmp + tmp') ./ 2)
            bbb = sign.(vtmp[:, end]) .* sqrt.(diag(tmp))
            tmp2 = bbb * bbb'
            if norm(tmp - tmp2) > 5.0e-6
                error("Obtained an error of `$(norm(tmp - tmp2)) > 5e-6` when converting matrix into rank `1`, use `datarank = 0` to disable the rank-1 conversion.")
            end
            Btmp[k, bidx] = bbb
        end
    end

    return Btmp
end

function prep_AA!(Ai,n)

    @inbounds Threads.@threads for j = 1:n
        if isempty(Ai[j+1])
            Ai[j+1][1, 1] = 0
        end
    end

    ntmp = size(Ai[1], 1) * size(Ai[1], 2)
    
    nnz = 0
    @inbounds for j = 1:n
        nnz += SparseArrays.nnz(Ai[j+1])
    end

    iii = zeros(Int64, nnz)
    jjj = zeros(Int64, nnz)
    vvv = zeros(Float64, nnz)
    lb = 1
    @inbounds for j = 1:n      
        ii,vv = findnz(-(Ai[j+1])[:])
        lf = lb+length(ii)-1
        iii[lb:lf] = ii
        jjj[lb:lf] = j .* ones(Int64,length(ii))
        vvv[lb:lf] = float(vv)
        lb = lf+1
    end
    AAA = sparse(iii,jjj,vvv,ntmp,n)

    return AAA
end

struct ScalarIndex
    value::Int64
end

num_scalars(model::MyModel) = length(model.d_lin)

function scalar_indices(model::MyModel)
    return MOI.Utilities.LazyMap{ScalarIndex}(ScalarIndex, Base.OneTo(num_scalars(model)))
end

struct MatrixIndex
    value::Int64
end

num_matrices(model::MyModel) = model.nlmi

function matrix_indices(model::MyModel)
    return MOI.Utilities.LazyMap{MatrixIndex}(MatrixIndex, Base.OneTo(num_matrices(model)))
end

side_dimension(model::MyModel, i::MatrixIndex) = model.msizes[i.value]

struct ConstraintIndex
    value::Int64
end
num_constraints(model::MyModel) = length(model.b)
function constraint_indices(model::MyModel)
    return MOI.Utilities.LazyMap{ConstraintIndex}(ConstraintIndex, Base.OneTo(num_constraints(model)))
end

jac(model::MyModel, i::ConstraintIndex, ::Type{ScalarIndex}) = model.C_lin[i.value,:]
jac(model::MyModel, i::MatrixIndex) = model.AA[i.value]'

function obj(model::MyModel, X, i::MatrixIndex)
    return -dot(model.C[i.value], X)
end

function obj(model::MyModel, X_lin, ::Type{ScalarIndex})
    return -dot(model.d_lin, X_lin)
end

function obj(model::MyModel, X_lin, X)
    return model.b_const - btrace(model.nlmi, model.C, X) - dot(model.d_lin, X_lin)
end

dual_obj(model::MyModel, y) = -dot(model.b, y) + model.b_const

function jtprod(model::MyModel, ::Type{ScalarIndex}, y)
    return -model.C_lin' * y
end

function dual_cons(model::MyModel, ::Type{ScalarIndex}, y, S)
    return model.d_lin - S + jtprod(model, ScalarIndex, y)
end

function jtprod(model::MyModel, mat_idx::MatrixIndex, y)
    return -mat(model.AA[mat_idx.value]' * y)
end

function dual_cons(model::MyModel, mat_idx::MatrixIndex, y, S)
    i = mat_idx.value
    return model.C[i] - S[i] + jtprod(model, mat_idx, y)
end

objgrad(model::MyModel, ::Type{ScalarIndex}) = model.d_lin
objgrad(model::MyModel, i::MatrixIndex) = model.C[i.value]

cons_constant(model::MyModel) = model.b

function cons(model::MyModel, x, X)
    return model.b - jprod(model, x, X)
end

function jprod(model::MyModel, i::MatrixIndex, W)
    return model.AA[i.value] * vec(W)
end

function jprod(model::MyModel, w, W)
    h = model.C_lin * w
    for i in matrix_indices(model)
        h += jprod(model, i, W[i.value])
    end
    return h
end

# [HKS24, (5b)]
# Returns the matrix equal to the sum, for each equation, of
# ⟨A_i, WA_jW⟩
function schur_complement(model::MyModel, w, W, G, datarank)
    if model.nlmi > 0
        if datarank == -1
        # if 1 == 0
            H = makeBBBB_rank1(model.n, model.nlmi, model.B, G)
        else
            H = makeBBBBs(model.n, model.nlmi, model.A, model.AA, W, model.qA, model.sigmaA)
        end
    else
        H = zeros(eltype(w), model.n, model.n)
    end
    if model.nlin > 0
        H .+= schur_complement(model, w, ScalarIndex)
    end
    return Hermitian(H, :L)
end

function schur_complement(model::MyModel, w, ::Type{ScalarIndex})
    return model.C_lin * spdiagm(w) * model.C_lin'
end

# [HKS24, (5b)]
# Returns the matrix equal to the sum, for each equation, of
# ⟨A_i, WA(y)W⟩
function eval_schur_complement!(result, model::MyModel, w, W, y)
    result .= 0.0
    for mat_idx in matrix_indices(model)
        i = mat_idx.value
        result .+= model.AA[i] * (W[i] * mat(transpose(model.AA[i]) * y) * W[i])[:]
    end
    result .+= model.C_lin * (w .* (model.C_lin' * y))
    return result
end


# end #module


# function prep_sparse!(A,n,m,i,nzA,sigmaA,qA)
# This is the Kojima et al data sparsity handling
#     d1 = zeros(Float64,n)
#     d2 = zeros(Float64,n)
#     d3 = zeros(Float64,n)

#     kappa = 100
#     kappa = 500000/m
#     for j = 1:n
#         nzA[j,i] = nnz(A[i,j+1])
#     end
#     sigmaA[:,i] = sortperm(nzA[:,i], rev = true)
#     @show nzA[:,i]
#     # @show nzA[sigmaA[:,i],i]
#     sisi = sort(nzA[sigmaA[:,i],i], rev = true)
#     # @show sigmaA[:,i]
#     # @show sisi
#     cs = cumsum(sisi[end:-1:1])
#     cs = cs[n:-1:1]
#     # @show cs

#     for j = 1:n
#         d1[j] = kappa * m * nzA[sigmaA[j,i],i] + m^3 + kappa * cs[j]
#         d2[j] = kappa * m * nzA[sigmaA[j,i],i] + kappa * (n+1) * cs[j]
#         d3[j] = kappa * (2 * kappa * nzA[sigmaA[j,i],i] + 1) * cs[j]
#     end


#     qA[1,i] = 0
#     ktmp = 0
#     for j = 1:n
#         if d1[j] > min(d2[j],d3[j])
#             qA[1,i] = j-1
#             ktmp = 1
#             break
#         end
#     end
#     if ktmp == 0
#         qA[1,i] = n
#         qA[2,i] = n
#     else
#         qA[2,i] = 0
#         for j = max(1,qA[1,i]):n
#             if d2[j] >= d1[j] || d2[j] > d3[j]
#                 qA[2,i] = j-1
#                 break
#             end
#         end
#     end
#     qA[2,i] = max(qA[2,i],qA[1,i])

#     @show qA

# end
