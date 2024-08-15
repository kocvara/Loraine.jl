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

#modules
include("Solvers.jl")
using .Solvers

include("kron_etc.jl")
include("makeBBBB.jl")
include("initial_point.jl")
include("predictor_corrector.jl")
include("prepare_W.jl")
include("MOI_wrapper.jl")

function loraine(d, options::Dict)

    verb   = Int64(get(options, "verb", 1))
    timing = Int64(get(options, "timing", 1))
    kit    = Int64(get(options, "kit", 1))
    drank    = Int64(get(options, "datarank", 1))
    κ = Int64(get(dest.options,"datasparsity",1))
    if verb > 0
        t1 = time()
        # @printf("\n *** Loraine.jl v0.1 ***\n")
        # @printf(" *** Initialisation STARTS\n")
    end

    ```PREPARE MODEL```
    model = prepare_model_data(d,drank,κ)

    ```LOAD MODEL```
    solver, halpha = load(model,options)

    tottime = time() - t1
    # if verb > 0
    #     @printf(" *** Preprocessing finished in %8.2f seconds\n", tottime)
    # end

    solver.to = TimerOutput()
    t1 = time()

    # if verb > 0
    #     @printf(" *** IP STARTS\n")
    #     if verb < 2
    #         if kit == 0
    #             @printf(" it        obj         error     CPU/it\n")
    #         else
    #             @printf(" it        obj         error     cg_iter   CPU/it\n")
    #         end
    #     else
    #         if kit == 0
    #             @printf(" it        obj         error      err1      err2      err3      err4      err5      err6     CPU/it\n")
    #         else
    #             @printf(" it        obj         error      err1      err2      err3      err4      err5      err6     cg_pre cg_cor  CPU/it\n")
    #         end
    #     end
    # end

    ```SOLVE```
    @timeit solver.to "solver" begin
    solve(solver::MySolver,halpha::Halpha)
    end

    tottime = time() - t1

    if verb > 0
        # if kit == 1
        #     @printf(" *** Total CG iterations: %8.0d \n", solver.cg_iter_tot)
        # end
        # @printf(" *** Optimal solution found in %8.2f seconds\n", tottime)
    end
    
    if timing > 0
        show(solver.to)
    end
    @printf("\n")

end

end #module
