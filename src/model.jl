# module Model

export prepare_model_data, MyModel, SpMa

using SparseArrays
using Printf
using TimerOutputs
using LinearAlgebra

struct SpMa{Tv,Ti<:Integer}
    n::Int
    iind::Vector{Ti}
    jind::Vector{Ti}
    nzval::Vector{Tv}
end

mutable struct MyModel
    A::Matrix{Any}
    AA::Vector{SparseArrays.SparseMatrixCSC{Float64}}
    myA::Vector{SpMa{Float64}}
    C::Vector{SparseArrays.SparseMatrixCSC{Float64}}
    b::SparseArrays.SparseMatrixCSC{Float64, Int64}
    d_lin::SparseArrays.SparseVector{Float64, Int64}
    C_lin::SparseArrays.SparseMatrixCSC{Float64, Int64}
    n::Int64
    msizes::Vector{Int64}
    nlin::Int64
    nlmi::Int64

    function MyModel(
        A::Matrix{Any},
        AA::Vector{SparseArrays.SparseMatrixCSC{Float64}},
        myA::Vector{SpMa{Float64}},
        C::Vector{SparseArrays.SparseMatrixCSC{Float64}},
        b::SparseArrays.SparseMatrixCSC{Float64, Int64},
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
        model.myA = myA
        model.C = C
        model.b = b
        model.d_lin = d_lin
        model.C_lin = C_lin
        model.n = n
        model.msizes = msizes
        model.nlin = nlin
        model.nlmi = nlmi
        return model
    end
end


function prepare_model_data(d)

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

if nlin > 0
    d_lin = -get(d, "d", 1)
    d_lin = d_lin[:]
    C_lin = -get(d, "C", 1)
else
    d_lin = sparse([0.; 0.])
    C_lin = sparse([0. 0.;0. 0.])
end

model = MyModel(A, _prepare_A(A)..., b, d_lin, C_lin, n, msizes, nlin, nlmi)

return model
end

function _prepare_A(A)
    nlmi = size(A, 1)
    n = size(A, 2) - 1
    AA = SparseMatrixCSC{Float64}[]
    myA = SpMa{Float64}[]
    C = SparseMatrixCSC{Float64}[]

    for i = 1:nlmi
        
        push!(C, copy(-A[i, 1]))

        Ai = A[i,:]
        AAA = prep_AA!(myA,Ai,n)
        push!(AA, copy(AAA'))

    end
    return AA, myA, C
end

function prep_AA!(myA,Ai,n)

    @inbounds Threads.@threads for j = 1:n
        if isempty(Ai[j+1])
            Ai[j+1][1, 1] = 0
        end
    end

    ntmp = size(Ai[1], 1) * size(Ai[1], 2)
    
    nnz = 0
    @inbounds for j = 1:n
        ii,jj,vv = findnz(-(Ai[j+1]))
        push!(myA,SpMa(length(ii),ii,jj,float(vv)))
        nnz += length(ii)
    end

    iii = zeros(Int, nnz)
    jjj = zeros(Int, nnz)
    vvv = zeros(Float64, nnz)
    AAA1 = spzeros(ntmp, n)
    lb = 1
    @inbounds for j = 1:n      
        ii,vv = findnz(-(Ai[j+1])[:])
        lf = lb+length(ii)-1
        iii[lb:lf] = ii
        jjj[lb:lf] = j .* ones(Int,length(ii))
        vvv[lb:lf] = float(vv)
        lb = lf+1
    end
    AAA = sparse(iii,jjj,vvv,ntmp,n)

    return AAA
end

# end #module
