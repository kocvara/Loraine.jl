""""Solve a semidefinite optimization problem in SDPA input format by Loraine using JuMP 
(or any other SDP code linked in JuMP)"""

using JuMP
import Loraine 
import CSDP
# import SCS
using Mosek
using MosekTools
# import Hypatia
using Dualization

# Select your semidefinite optimization problem in SDPA input format
# model=read_from_file("/Users/michal/Dropbox/michal/sdplib/Hans/trto3.dat-s")
model=read_from_file("/Users/michal/Dropbox/michal/sdplib/gpp250-1.dat-s")
# model=read_from_file("/Users/michal/Dropbox/michal/POEMA/IP/ip-for-low-rank-sdp/database/problems/SDPA/tru9e.dat-s")
# model=read_from_file("examples/data/vib9.dat-s")

set_optimizer(model, Loraine.Optimizer)

# Loraine options

MOI.set(model, MOI.RawOptimizerAttribute("kit"), 0)
MOI.set(model, MOI.RawOptimizerAttribute("tol_cg"), 1.0e-2)
MOI.set(model, MOI.RawOptimizerAttribute("tol_cg_min"), 1.0e-6)
MOI.set(model, MOI.RawOptimizerAttribute("eDIMACS"), 1e-5)
MOI.set(model, MOI.RawOptimizerAttribute("preconditioner"), 4)
MOI.set(model, MOI.RawOptimizerAttribute("erank"), 4)
MOI.set(model, MOI.RawOptimizerAttribute("aamat"), 2)
MOI.set(model, MOI.RawOptimizerAttribute("verb"), 2)
MOI.set(model, MOI.RawOptimizerAttribute("initpoint"), 0)
MOI.set(model, MOI.RawOptimizerAttribute("maxit"), 50)

optimize!(model)

# Mosek (CSDP, etc) for a comparison
# Mosek must solve the dualized problem to be efficient

# set_optimizer(model, Mosek.Optimizer)
# dual_model = dualize(model, Mosek.Optimizer)
# optimize!(dual_model)

# termination_status(model)