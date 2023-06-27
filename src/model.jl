# module Model

# NOT WORKING!!!

export prepare_model_data, MyModel, SpMa

using SparseArrays
using Printf
using TimerOutputs
using LinearAlgebra

struct SpMa{Tv,Ti<:Integer}
    n::Int64
    iind::Vector{Ti}
    jind::Vector{Ti}
    nzval::Vector{Tv}
end

mutable struct MyModel
    A::Matrix{Any}
    AA::Vector{SparseArrays.SparseMatrixCSC{Float64}}
    myA::Vector{SpMa{Float64}}
    B::Vector{SparseArrays.SparseMatrixCSC{Float64}}
    C::Vector{SparseArrays.SparseMatrixCSC{Float64}}
    b::Vector{Float64}
    b_const::Float64
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
        B::Vector{SparseArrays.SparseMatrixCSC{Float64}},
        C::Vector{SparseArrays.SparseMatrixCSC{Float64}},
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
        model.myA = myA
        model.B = B
        model.C = C
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
b_const = -get(d, "b_const", 1);

if nlin > 0
    d_lin = -get(d, "d", 1)
    d_lin = d_lin[:]
    C_lin = -get(d, "C", 1)
else
    d_lin = sparse([0.; 0.])
    C_lin = sparse([0. 0.;0. 0.])
end

drank = 0
model = MyModel(A, _prepare_A(A,drank)..., b, b_const, d_lin, C_lin, n, msizes, nlin, nlmi)

return model
end

function _prepare_A(A, datarank)

    nlmi = size(A, 1)
    n = size(A, 2) - 1
    AA = SparseMatrixCSC{Float64}[]
    myA = SpMa{Float64}[]
    B = SparseMatrixCSC{Float64}[]
    C = SparseMatrixCSC{Float64}[]

    for i = 1:nlmi
        
        push!(C, copy(-A[i, 1]))

        Ai = A[i,:]
        AAA = prep_AA!(myA,Ai,n)
        push!(AA, copy(AAA'))

        # if 1 == 0
        if datarank == -1
            Btmp = prep_B!(A,n,i)
            push!(B, Btmp)
        end

    end

    return AA, myA, B, C
end

function prep_B!(A,n,i)
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
                drank = 0
                println("\n WARNING: data conversion problem, switching to datarank = 0")
                break
            end
            Btmp[k, bidx] = bbb
        end
    end

    return Btmp
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
        push!(myA,SpMa(Int64(length(ii)),ii,jj,float(vv)))
        nnz += length(ii)
    end

    iii = zeros(Int64, nnz)
    jjj = zeros(Int64, nnz)
    vvv = zeros(Float64, nnz)
    AAA1 = spzeros(ntmp, n)
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

# end #module
