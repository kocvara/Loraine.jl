#=
Copyright (c) 2023 Soodeh Habibi, Michal Kocvara, Michael Stingl and co-authors

Loraine.jl is a Julia package developed for H2020 ITN POEMA (http://poema-network.eu) 
and is distributed under the GNU General Public License 3.0. 
=#

module Loraine

export loraine
using SparseArrays
using LinearAlgebra
using Printf
using TimerOutputs
using FameSVD
# using MKLSparse
# using MKL

import MathOptInterface as MOI
import LowRankOpt as LRO
struct Optimizer{T}
    dummy::T
end
function Optimizer{T}() where {T}
    model = LRO.Optimizer{T}()
    MOI.set(
        model,
        MOI.RawOptimizerAttribute("solver"),
        Solvers.Solver{T},
    )
    return model
end
Optimizer() = Optimizer{Float64}()

#modules
include("Solvers.jl")
using .Solvers

include("kron_etc.jl")
include("initial_point.jl")
include("predictor_corrector.jl")
include("prepare_W.jl")

"""
    loraine(filename::AbstractString, options::Dict = Dict{String,Any}(); T::Type = Float64)

Read the semidefinite program in SDPA format from `filename`, solve it with
Loraine in the arithmetic `T` and return the optimizer, which can then be
queried with `MOI.get`, e.g. with `MOI.ObjectiveValue()`.

Each entry of `options` is set as a `MOI.RawOptimizerAttribute`, see
[Options](@ref) for the list of available options.

```julia
model = loraine("theta1.dat-s", Dict("kit" => 1))
MOI.get(model, MOI.ObjectiveValue())
```
"""
function loraine(
    filename::AbstractString,
    options::Dict = Dict{String,Any}();
    T::Type = Float64,
)
    src = MOI.FileFormats.SDPA.Model{T}()
    MOI.read_from_file(src, filename)
    model = MOI.instantiate(Optimizer{T}; with_bridge_type = T)
    for (name, value) in options
        MOI.set(model, MOI.RawOptimizerAttribute(name), value)
    end
    MOI.copy_to(model, src)
    MOI.optimize!(model)
    return model
end

end #module
