
using SparseArrays
using LinearAlgebra
using MAT
using Printf
using TimerOutputs
# using MKL
# using Profile
# using ProfileView

using Loraine

const to = TimerOutput()

# READING the input file in Matlab format

file = matopen("/Users/michal/Dropbox/michal/POEMA/IP/ip-for-low-rank-sdp/d.mat")
d = read(file, "d")

c = get(d, "c", 1)
d["c"] = Vector{Float64}(c[:])
d["b_const"] = 0.0 

# OPTIONS FOR myIP
options = Dict{String,Float64}()
options["kit"]          = 0         # kit = 0 for direct solver; kit = 1 for CG
options["tol_cg"]       = 1e-2      # tolerance for CG solver [1e-2]
options["tol_cg_up"]    = 0.5       # tolerance update [0.5]
options["tol_cg_min"]   = 1e-6      # minimal tolerance for CG solver [1e-6]
options["eDIMACS"]      = 1e-24      # epsilon for DIMACS error stopping criterion [1e-5]

options["preconditioner"]  = 1    # 0...no; 1...H_tilde with SMW; 2...H_hat; 3...H_tilde inverse; 4...hybrid
options["erank"] = 1    # estimated rank()

options["aamat"] = 2    # 0 ... A^TA; 1 ... diag(A^TA); 2 ... identity

# Use fig_ev=1 only for small problems!!! switches to preconditioner = 3 !!!
options["fig_ev"]   = 0  # 0...no figure; 1.1.figure with eigenvalues of H; H_til; etc in every IP iteration

options["verb"]     = 1   # 2..full output; 1..short output; 0..no output
options["initpoint"]    = 1     # 0..Loraine heuristics; 1..SDPT3 heuristics
options["maxit"]    = 500
options["datarank"] = 0

## CALLING Loraine

loraine(d,options)

# end