""""Solve a semidefinite optimization problem in SDPA input format by Loraine using JuMP 
(or any other SDP code linked in JuMP)"""

using JuMP
import Loraine 
# import CSDP
# import SCS
# using Mosek
# using MosekTools
# import Hypatia
using Dualization
using MultiFloats

function read_sdpa(
    filename::String,
    T::Type = Float64,
)
    src = MOI.FileFormats.SDPA.Model{T}()
    MOI.read_from_file(src, filename)
    model = GenericModel{T}()
    MOI.copy_to(model, src)
    return model
end

# model = JuMP.GenericModel{Float64x2}(Loraine.Optimizer)

# Select your semidefinite optimization problem in SDPA input format
# model=read_from_file("/Users/michal/Dropbox/michal/sdplib/Hans/trto2.dat-s")
# model=read_from_file("/Users/michal/Dropbox/michal/sdplib/control3.dat-s")
# model=read_from_file("/Users/michal/Dropbox/michal/POEMA/IP/ip-for-low-rank-sdp/database/problems/SDPA/vib5.dat-s")
# model=read_from_file("/Users/michal/Dropbox/michal/j/k.dat-s")

# model=read_sdpa("examples/data/theta1.dat-s")
model=read_from_file(joinpath(dirname(@__DIR__), "examples/data/theta1.dat-s"))
# model=read_from_file("examples/data/theta1.dat-s")
# model=read_from_file("examples/data/maxG11.dat-s") #use with "datarank = -1"

set_optimizer(model, Loraine.Optimizer{Float64})
# set_optimizer(model, Loraine.Optimizer{Float64x2})

# Loraine options

set_attribute(model, "kit", 0)
set_attribute(model, "tol_cg", 1.0e-2)
set_attribute(model, "tol_cg_min", 1.0e-6)
set_attribute(model, "eDIMACS", 1e-6)
set_attribute(model, "preconditioner", 1)
set_attribute(model, "erank", 1)
set_attribute(model, "aamat", 2)
set_attribute(model, "verb", 1)
set_attribute(model, "datarank", 0)
set_attribute(model, "initpoint", 1)
set_attribute(model, "maxit", 100)
set_attribute(model, "datasparsity", 8)

# set_optimizer(model, CSDP.Optimizer)
optimize!(model)

# solution_summary(model)
using Test
@test objective_value(model) ≈ 23 rtol = 1e-6
# # value.(X)

# Mosek (CSDP, etc) for a comparison
# Mosek must solve the dualized problem to be efficient

# set_optimizer(model, CSDP.Optimizer)
# dual_model = dualize(model, CSDP.Optimizer)
# @time optimize!(dual_model)

# termination_status(model)
