export prepare_model_data, MyModel

using SparseArrays
using Printf
using TimerOutputs
using LinearAlgebra
import MutableArithmetics as MA
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
* The matrix ``C_i`` is given by `C[i]`.
* The matrix ``A_{i,j}`` is given by `-A[i,j]`.
* The index `j = sigmaA[k,i]` is the `k`th matrix ``A_{i,j}`` of the largest number of nonzeros.
* The first `qA[1,i] = qA[2,i]` matrices are considered as dense in the computation.
"""
mutable struct MyModel{T,A<:AbstractMatrix{T}}
    A::Matrix{A}
    C::Vector{SparseArrays.SparseMatrixCSC{T,Int}}
    sigmaA::Matrix{Int64}
    qA::Matrix{Int64}
    b::Vector{T}
    b_const::T
    d_lin::SparseArrays.SparseVector{T, Int64}
    C_lin::SparseArrays.SparseMatrixCSC{T, Int64}
    msizes::Vector{Int64}

    function MyModel(
        A::Matrix{AT},
        C::Vector{SparseArrays.SparseMatrixCSC{T,Int}},
        sigmaA::Matrix{Int64},
        qA::Matrix{Int64},
        b::Vector{T},
        b_const::T,
        d_lin::SparseArrays.SparseVector{T, Int64},
        C_lin::SparseArrays.SparseMatrixCSC{T, Int64},
        msizes::Vector{Int64},
    ) where {T,AT<:AbstractMatrix{T}}

        model = new{T,AT}()
        model.A = A
        model.C = C
        model.sigmaA = sigmaA
        model.qA = qA
        model.b = b
        model.b_const = b_const
        model.d_lin = d_lin
        model.C_lin = C_lin
        model.msizes = msizes
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
@assert size(A, 1) == nlmi
b = -get(d, "c", 1);
@assert length(b) == n
b_const = -get(d, "b_const", 1);

if nlin > 0
    d_lin = -get(d, "d", 1)
    d_lin = d_lin[:]
    C_lin = -get(d, "C", 1)
else
    d_lin = sparse([0.; 0.])
    C_lin = sparse([0. 0.;0. 0.])
end

model = MyModel(A[:,2:end], _prepare_A(A,drank,κ)..., b, b_const, d_lin, C_lin, msizes)

return model
end

function _prepare_A(A::Matrix{SparseMatrixCSC{T,Int}}, datarank, κ) where {T}

    nlmi = size(A, 1)
    n = size(A, 2) - 1
    C = SparseMatrixCSC{T,Int}[]
    sigmaA = zeros(Int64,n,nlmi)
    qA = zeros(Int64,2,nlmi)

    for i = 1:nlmi
        
        push!(C, copy(-A[i, 1]))

        Ai = A[i,:]
        m = size(Ai,1)

        prep_sparse!(A,n,i,sigmaA,qA,κ)

    end

    return C, sigmaA, qA
end


function prep_sparse!(A,n,i,sigmaA,qA,κ)
    # Simplified data sparsity handling

    nzA = [nnz(A[i,j+1]) for j in 1:n]
    sigmaA[:,i] = sortperm(nzA, rev = true)
    sisi = nzA[sigmaA[:,i]]
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

num_matrices(model::MyModel) = length(model.C)

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

# Should be only used with `norm`
jac(model::MyModel, i::ConstraintIndex, ::Type{ScalarIndex}) = model.C_lin[i.value,:]
function norm_jac(model::MyModel{T}, i::MatrixIndex) where {T}
    if isempty(model.A)
        return zero(T)
    end
    return norm(model.A[i.value, :])
end

function obj(model::MyModel, X, i::MatrixIndex)
    return -dot(model.C[i.value], X)
end

function obj(model::MyModel, X, ::Type{MatrixIndex})
    result = zero(eltype(eltype(X)))
    for mat_idx in matrix_indices(model)
        result += obj(model, X[mat_idx.value], mat_idx)
    end
    return result
end

function obj(model::MyModel, X_lin, ::Type{ScalarIndex})
    return -dot(model.d_lin, X_lin)
end

function obj(model::MyModel, X_lin, X)
    return model.b_const + obj(model, X, MatrixIndex) - dot(model.d_lin, X_lin)
end

dual_obj(model::MyModel, y) = -dot(model.b, y) + model.b_const

function jtprod(model::MyModel, ::Type{ScalarIndex}, y)
    return -model.C_lin' * y
end

function dual_cons(model::MyModel, ::Type{ScalarIndex}, y, S)
    return model.d_lin - S + jtprod(model, ScalarIndex, y)
end

function buffer_for_jtprod(model::MyModel)
    if iszero(num_matrices(model))
        return
    end
    return map(Base.Fix1(buffer_for_jtprod, model), matrix_indices(model))
end

function buffer_for_jtprod(model::MyModel, mat_idx::MatrixIndex)
    if iszero(num_constraints(model))
        return
    end
    # FIXME: at some point, switch to dense
    return sum(
        abs.(model.A[mat_idx.value, j])
        for j in 1:num_constraints(model)
    )
end

function _add_mul!(A::SparseMatrixCSC, B::SparseMatrixCSC, α)
    for col in axes(A, 2)
        range_A = SparseArrays.nzrange(A, col)
        it_A = iterate(range_A)
        for k in SparseArrays.nzrange(B, col)
            row_B = SparseArrays.rowvals(B)[k]
            while SparseArrays.rowvals(A)[it_A[1]] < row_B
                it_A = iterate(range_A, it_A[2])
            end
            @assert row_B == SparseArrays.rowvals(A)[it_A[1]]
            SparseArrays.nonzeros(A)[it_A[1]] += SparseArrays.nonzeros(B)[k] * α
        end
    end
end

_zero!(A::SparseMatrixCSC) = fill!(SparseArrays.nonzeros(A), 0.0)

function jtprod!(buffer, model::MyModel, mat_idx::MatrixIndex, y)
    if iszero(num_constraints(model))
        return MA.Zero()
    end
    _zero!(buffer)
    for j in eachindex(y)
        _add_mul!(buffer, model.A[mat_idx.value, j], y[j])
    end
    return buffer
end

function dual_cons!(buffer, model::MyModel, mat_idx::MatrixIndex, y, S)
    i = mat_idx.value
    return jtprod!(buffer[i], model, mat_idx, y) + model.C[i] - S[i]
end

objgrad(model::MyModel, ::Type{ScalarIndex}) = model.d_lin
objgrad(model::MyModel, i::MatrixIndex) = model.C[i.value]

cons_constant(model::MyModel) = model.b

function cons(model::MyModel, x, X)
    return model.b - jprod(model, x, X)
end

function jprod(model::MyModel, i::MatrixIndex, W)
    return eltype(W)[
        -dot(model.A[i.value, j], W) for j in 1:num_constraints(model)
    ]
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
function schur_complement(model::MyModel, w, W)
    H = MA.Zero()
    if num_matrices(model) > 0
        H = MA.add!!(H, makeBBBBs(model, W))
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
