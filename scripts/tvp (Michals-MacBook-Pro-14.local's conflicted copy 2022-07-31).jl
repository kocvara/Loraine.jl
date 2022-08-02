
module tvp

using SparseArrays
using LinearAlgebra
using MAT
using Printf
using TimerOutputs
using MKL
using Profile
using ProfileView


include("../src/Loraine.jl")
using .Loraine

const to = TimerOutput()

# READING the input file in Matlab format

# file = matopen("/Users/michal/Dropbox/michal/POEMA/IP/ip-for-low-rank-sdp/d.mat")
# file = matopen("examples/control1.poema")
# file = matopen("examples/theta1.poema")
file = matopen("examples/vib3.poema")
# file = matopen("examples/trto3.poema")
d = read(file, "d")

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
options["timing"]   = 1   # 1..print timing; 0..don't
options["maxit"]    = 200

## CALLING Loraine

loraine(d,options)

end