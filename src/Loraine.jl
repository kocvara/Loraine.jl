module Loraine

export loraine
using SparseArrays
using LinearAlgebra
using Printf
using TimerOutputs
# using MKLSparse
using MKL

#modules
include("Model.jl")
include("Solvers.jl")
include("Precon.jl")
using .Model
using .Solvers
using .Precon

include("kron_etc.jl")
include("makeBBBB.jl")
include("initial_point.jl")
include("predictor_corrector.jl")
include("prepare_W.jl")

function loraine(d,options)

    verb = Int64(get(options, "verb", 1))
    kit  = Int64(get(options, "kit", 1))
    if verb > 0
        t1 = time()
        @printf("\n *** Loraine.jl v0.1 ***\n")
        @printf(" *** Initialisation STARTS\n")
    end

    model = prepare_model_data(d)

    solver, halpha = load(model,options)

    tottime = time() - t1
    if verb > 0
        @printf(" *** Preprocessing finished in %8.2f seconds\n", tottime)
    end

    solver.to = TimerOutput()
    t1 = time()

    if verb > 0
        @printf(" *** IP STARTS\n")
        if verb < 2
            if kit == 0
                @printf(" it        obj         error     CPU/it\n")
            else
                @printf(" it        obj         error     cg_iter   CPU/it\n")
            end
        else
            if kit == 0
                @printf(" it        obj         error      err1      err2      err3      err4      err5      err6     CPU/it\n")
            else
                @printf(" it        obj         error      err1      err2      err3      err4      err5      err6     cg_pre cg_cor  CPU/it\n")
            end
        end
    end
    @timeit solver.to "solver" begin
    solve(solver::MySolver,halpha::Halpha)
    end

    tottime = time() - t1

    if verb > 0
        @printf(" *** Total CG iterations: %8.0d \n", solver.cg_iter_tot)
        @printf(" *** Optimal solution found in %8.2f seconds\n", tottime)
    end
    
    show(solver.to)
    @printf("\n")

end

end #module