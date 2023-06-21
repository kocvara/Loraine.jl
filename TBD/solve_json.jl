
# module tvp

using SparseArrays
using LinearAlgebra
using Printf
using TimerOutputs
using JSON
# using MKL
# using Profile
# using ProfileView

using Loraine

const to = TimerOutput()

# READING the input file in POEMA-JSON format

tmp = JSON.parsefile("Loraine/examples/tru3.json")

didi = Dict()
didi["name"] = get(tmp, "name", 1)
didi["type"] = get(tmp, "type", 1)
nvar = get(tmp, "nvar", 1)
didi["nvar"] = convert(Int64,nvar)
didi["c"] = Vector{Float64}(get(tmp, "objective", 1)[:])

tmpcon = get(tmp, "constraints",1)
nlmi = get(tmpcon, "nlmi", 1)
didi["nlmi"] = convert(Int64,nlmi)
msizes = get(tmpcon, "msizes", 1)
if nlmi == 1
    didi["msizes"] = msizes
else
    didi["msizes"] = Vector{Int64}(msizes)
end

tmpA = get(tmpcon, "lmi_symat", 1)

lA = length(tmpA)
BB = zeros(nlmi, nvar + 1, maximum(tmpA)[4], maximum(tmpA)[5])
for i = 1:length(tmpA)
    BB[tmpA[i][3],tmpA[i][2]+1,tmpA[i][4],tmpA[i][5]] = tmpA[i][1]
end

A = SparseMatrixCSC{Float64}[]
# A = Matrix{Any}
if nlmi == 1  
    Btmp = SparseMatrixCSC{Float64}[]
    Ctmp = sparse(BB[1,1,:,:])
    Ctmp = Ctmp + Ctmp' - spdiagm(diag(Ctmp))
    push!(Btmp,Ctmp)
    for j = 1:nvar+1
        push!(A,sparse(BB[1,j,:,:]))
    end
else
    for ilmi = 1:nlmi
        Btmp = SparseMatrixCSC{Float64}[]
        for j = 1:nvar+1
            push!(Btmp,sparse(BB[ilmi,j,:,:]))
        end
        push!(A,Btmp)
    end
end
didi["A"] = A

nlsi = get(tmpcon, "nlsi", 1)
didi["nlsi"] = convert(Int64,nlsi)
tmpC = get(tmpcon, "lsi_mat", 1)
iii = zeros(nlsi)
jjj = zeros(nlsi)
vvv = zeros(nlsi)
for i = 1:nlsi
    iii[i] = tmpC[i][2]
    jjj[i] = tmpC[i][3]
    vvv[i] = tmpC[i][1]
end
didi["C"] = sparse(iii,jjj,vvv,nlsi,nvar)
dtmp = sparse(get(tmpcon, "lsi_vec", 1)[:])
didi["d"] = convert(SparseVector{Float64},dtmp)
didi["lsi_op"] = convert(Vector{Int64},get(tmpcon, "lsi_op", 1)[:])

@show didi

# OPTIONS FOR myIP
options = Dict{String,Float64}()
options["kit"]          = 1         # kit = 0 for direct solver; kit = 1 for CG
options["tol_cg"]       = 1e-1      # tolerance for CG solver [1e-2]
options["tol_cg_up"]    = 0.5       # tolerance update [0.5]
options["tol_cg_min"]   = 1e-6      # minimal tolerance for CG solver [1e-6]
options["eDIMACS"]      = 3e-5      # epsilon for DIMACS error stopping criterion [1e-5]

options["prec"]  = 1    # 0...no; 1...H_tilde with SMW; 2...H_hat; 3...H_tilde inverse; 4...hybrid
options["erank"] = 1    # estimated rank()

options["aamat"] = 2    # 0 ... A^TA; 1 ... diag(A^TA); 2 ... identity

# Use fig_ev=1 only for small problems!!! switches to preconditioner = 3 !!!
options["fig_ev"]   = 0  # 0...no figure; 1.1.figure with eigenvalues of H; H_til; etc in every IP iteration

options["verb"]     = 1   # 2..full output; 1..short output; 0..no output
options["maxit"]    = 200

# CALLING Loraine

loraine(didi,options)

# end