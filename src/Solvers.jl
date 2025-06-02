module Solvers

export  MySolver, solve, load, MyA, Halpha

using SparseArrays
using LinearAlgebra
import Statistics: mean
using Printf
using TimerOutputs
using MultiFloats
import NLPModels
import SolverCore
import LowRankOpt as LRO
# using MKLSparse
# using MKL

include("kron_etc.jl")

struct FactoredMatrix{T} <: AbstractMatrix{T}
    factor::Matrix{T}
    factor_inv::Matrix{T} # inv(factor)
    matrix::Matrix{T} # factor * factor_inv'
end
Base.size(A::FactoredMatrix) = size(A.matrix)
Base.getindex(A::FactoredMatrix, i, j) = Base.getindex(A.matrix, i, j)

mutable struct MySolver{T,A,B,SB}
    # main options
    kit::Int64
    tol_cg::T
    tol_cg_up::T
    tol_cg_min::T
    eDIMACS::T
    preconditioner::Int64
    erank::Int64
    aamat::Int64
    fig_ev::Int64
    verb::Int64
    datarank::Int64
    initpoint::Int64
    timing::Int64
    maxit::Int64
    datasparsity::Int64

    to::Any

    # model and preprocessed model data
    model::LRO.Model{T,A}
    jtprod_buffer::B
    schur_buffer::SB

    predict::Bool

    # current iterate
    sigma::T
    tau::T
    mu::T
    expon::T
    iter::Int64
    DIMACS_error::T
    cholBBBB

    status::Int

    regcount::Int

    err1::T
    err2::T
    err3::T
    err4::T
    err5::T
    err6::T

    X::LRO.VectorizedSolution{T}
    S
    y
    yold
    delX
    delS
    dely
    Xn
    Sn

    S_lin::Vector{T}
    Si_lin::Vector{T}
    S_lin_inv::Vector{T}
    delX_lin::Vector{T}
    delS_lin::Vector{T}
    Xn_lin::Vector{T}
    Sn_lin::Vector{T}

    D
    W::LRO.ShapedSolution{T,FactoredMatrix{T}}
    Si
    DDsi

    Rp
    Rd
    Rc
    Rd_lin

    cg_iter_pre
    cg_iter_cor
    cg_iter_tot

    alpha
    beta
    alpha_lin
    beta_lin

    itertime
    tottime

    RNT
    RNT_lin

    function MySolver{T}(
        kit::Int64,
        tol_cg::Float64 ,
        tol_cg_up::Float64 ,
        tol_cg_min::Float64 ,
        eDIMACS::Float64 ,
        preconditioner::Int64,
        erank::Int64,
        aamat::Int64,
        fig_ev::Int64,
        verb::Int64,
        datarank::Int64,
        initpoint::Int64,
        timing::Int64,
        maxit::Int64, 
        datasparsity::Int64,
        model::LRO.Model{T,A}
    ) where {T,A}

        jtprod_buffer = LRO.buffer_for_jtprod(model)
        schur_buffer = LRO.buffer_for_schur_complement(model, datasparsity)
        solver = new{T,A,typeof(jtprod_buffer),typeof(schur_buffer)}()
        solver.kit             = kit
        solver.tol_cg          = tol_cg
        solver.tol_cg_up       = tol_cg_up
        solver.tol_cg_min      = tol_cg_min
        solver.eDIMACS         = eDIMACS
        solver.preconditioner  = preconditioner
        solver.erank           = erank
        solver.aamat           = aamat
        solver.fig_ev          = fig_ev
        solver.verb            = verb   
        solver.datarank        = datarank
        solver.initpoint       = initpoint   
        solver.timing          = timing
        solver.maxit           = maxit 
        solver.datasparsity    = datasparsity
        solver.model           = model
        solver.jtprod_buffer   = jtprod_buffer
        solver.schur_buffer    = schur_buffer
        return solver
    end
end

mutable struct Halpha
    kit
    Umat
    Z
    cholS
    AAAATtau
    function Halpha(
        kit::Int64
    )
    halpha = new()
    halpha.kit             = kit
    return halpha
    end
end

struct Solver{T,A,B,SB} <: SolverCore.AbstractOptimizationSolver
    solver::MySolver{T,A,B,SB}
    halpha::Halpha
    stats::SolverCore.GenericExecutionStats{T,Vector{T},Vector{T},Any}
end

function Solver{T}(model::LRO.Model; kws...) where {T}
    options = Dict(kw[1] => kw[2] for kw in kws)
    stats = SolverCore.GenericExecutionStats(model)
    solver, halpha = load(model, options; T)
    solver.X = LRO.VectorizedSolution{T}(stats.solution, model.dim)
    solver.y = stats.multipliers
    Solver(solver, halpha, stats)
end

const STATUS_MAP = [
    :unknown,
    :first_order,
    :infeasible,
    :unbounded,
    :max_iter,
]

function SolverCore.solve!(
    solver::Solver,
    model::NLPModels.AbstractNLPModel; # Same as `solver.model`, we can ignore
    kws...,
)
    for kw in kws
        field = Symbol(kw[1])
        if field == :verbose
            field = :verb # TODO rename to :verbose to follow NLPModels convention
        end
        setproperty!(solver.solver, field, kw[2])
    end
    solver.solver.to = TimerOutput()
    solve(solver.solver, solver.halpha)
    solver.stats.status = STATUS_MAP[solver.solver.status + 1]
    solver.stats.objective = NLPModels.obj(solver.solver.model, solver.solver.X)
    return
end


include("initial_point.jl")
include("prepare_W.jl") 
include("predictor_corrector.jl")

const DEFAULT_OPTIONS = Dict{String,Any}(
    "kit" => 0,
    "tol_cg" => 1.0e-2,
    "tol_cg_up" => 0.5,
    "tol_cg_min" => 1.0e-7,
    "eDIMACS" => 1.0e-7,
    "preconditioner" => 1,
    "erank" => 1,
    "aamat" => 1,
    "fig_ev" => 0,
    "verb" => 1,
    "datarank" => 0,
    "initpoint" => 0,
    "timing" => 1,
    "maxit" => 100,
    "datasparsity" => 8,
)

function load(model, options::Dict; T = Float64)

    kit = Int64(get(options, "kit", 0))
    tol_cg = get(options, "tol_cg", 1.0e-2)
    tol_cg_up = get(options, "tol_cg_up", 0.5)
    tol_cg_min = get(options, "tol_cg_min", 1.0e-7)
    eDIMACS = get(options, "eDIMACS", 1.0e-7)
    preconditioner = Int64(get(options, "preconditioner", 1))
    erank = Int64(get(options, "erank", 1))
    aamat = Int64(get(options, "aamat", 1))
    fig_ev = Int64(get(options, "fig_ev", 0))
    verb = Int64(get(options, "verb", 1))
    datarank = Int64(get(options, "datarank", 0))
    initpoint = Int64(get(options, "initpoint", 0))
    timing = Int64(get(options, "timing", 1))
    maxit = Int64(get(options, "maxit", 100))
    datasparsity = Int64(get(options, "datasparsity", 8))

    solver = MySolver{T}(kit,
        tol_cg,
        tol_cg_up,
        tol_cg_min,
        eDIMACS,
        preconditioner,
        erank,
        aamat,
        fig_ev,
        verb,
        datarank,
        initpoint,
        timing,
        maxit, 
        datasparsity,
        model,
    )

    halpha = Halpha(kit)
    solver.cg_iter_tot = 0

    # Input parameters check
    if kit < 0 || kit > 1
        solver.kit = 0
        @printf(" ---Parameter kit out of range, setting kit = %1d\n", solver.kit)
    end
    if tol_cg < tol_cg_min && kit == 1
        solver.tol_cg = tol_cg_min
        @printf(" ---Parameter tol_cg smaller than tol_cg_min, setting tol_cg = %7.1e\n", solver.tol_cg)
    end
    if tol_cg_min > eDIMACS && kit == 1
        solver.tol_cg_min = eDIMACS
        @printf(" ---Parameter tol_cg_min switched to eDIMACS = %7.1e\n", eDIMACS)
    end
    if kit == 1 && (preconditioner < 0 || preconditioner > 4)
        solver.preconditioner = 1
        @printf(" ---Parameter preconditioner out of range, setting preconditioner = %1d\n", solver.preconditioner)
    end
    if erank < 0 
        solver.erank = 1
        @printf(" ---Parameter erank negative, setting erank = %1d\n", solver.erank)
    end
    if datarank < -1 
        solver.datarank = 0
        @printf(" ---Parameter datarank out of range, setting datarank = %1d\n", solver.datarank)
    end
    if initpoint < 0 || initpoint > 1
        solver.initpoint = 1
        @printf(" ---Parameter kit out of range, setting initpoint = %1d\n", solver.initpoint)
    end

    # @show model.A
    # @show size(model.A,1)
    # if min(size(model.A,1),size(model.A,2)) < 1
    #     solver.status = 0
    #     @show size(model.A)
    #     error("Data A empty")
    # end

    return solver, halpha
end

function solve(solver::MySolver,halpha::Halpha)
    t1 = time()
    if solver.verb > 0
        @printf("\n *** Loraine.jl v0.2.5 ***\n")

        @printf(" Number of variables: %5d\n",LRO.num_constraints(solver.model))
        @printf(" LMI constraints    : %5d\n",LRO.num_matrices(solver.model))
        if LRO.num_matrices(solver.model) > 0
            @printf(" Matrix size(s)     :")
            msizes = LRO.side_dimension.(Ref(solver.model), LRO.matrix_indices(solver.model))
            Printf.format.(Ref(stdout), Ref(Printf.Format("%6d")), msizes);
            @printf("\n")
        end
        @printf(" Linear constraints : %5d\n",LRO.num_scalars(solver.model))
        if solver.kit>0
            @printf(" Preconditioner     : %5d\n",solver.preconditioner)
        else
            @printf(" Preconditioner     :  none, using direct solver\n")
        end

        @printf(" *** IP STARTS\n")
        if solver.verb < 2
            if solver.kit == 0
                @printf(" it        obj         error     CPU/it\n")
            else
                @printf(" it        obj         error     cg_iter   CPU/it\n")
            end
        else
            if solver.kit == 0
                @printf(" it        obj         error      err1      err2      err3      err4      err5      err6     CPU/it\n")
                @printf("                                p-eq-con   p-feas   d-eq-con   d-feas     d-gap  slackness         \n")
                @printf("---------------------------------------------------------------------------------------------------\n")
            else
                @printf(" it        obj         error      err1      err2      err3      err4      err5      err6    cg_pre  cg_cor  CPU/it\n")
            end
        end
    end

    setup_solver(solver::MySolver,halpha::Halpha)

    initial_point(solver)

    while solver.status == 0 

        t2 = time()
        myIPstep(solver,halpha)
        solver.itertime = time()-t2

        solver.tol_cg = max(solver.tol_cg * solver.tol_cg_up, solver.tol_cg_min)

        check_convergence(solver)

        if solver.preconditioner == 4
            #         if (cg_iter2>erank*nlmi*sqrt(n)/1 && iter>sqrt(n)/60)||cg_iter2>100 %for SNL problems
            if (solver.cg_iter_cor / 2 > solver.erank * LRO.num_matrices(solver.model) * sqrt(LRO.num_constraints(solver.model))/20 && solver.iter > sqrt(LRO.num_constraints(solver.model)) / 60) || solver.cg_iter_cor > 100
                solver.preconditioner = 1; solver.aamat = 2; 
                if solver.verb > 0
                    println("Switching to preconditioner 1")
                end
            end
        end

    end

    solver.tottime = time() - t1
    if solver.verb > 0
        if solver.kit == 1
            @printf(" *** Total CG iterations: %8.0d \n", solver.cg_iter_tot)
        end
        if solver.status == 1
            @printf(" *** Optimal solution found in %8.2f seconds\n", solver.tottime)
        end
    end

end

function setup_solver(solver::MySolver{T},halpha::Halpha) where {T}

    solver.S = Matrix{T}[]

    solver.delX = Matrix{T}[]
    solver.delS = Matrix{T}[]

    solver.D = Vector{T}[]
    solver.W = LRO.ShapedSolution{T,FactoredMatrix{T}}(
        zeros(T, LRO.num_scalars(solver.model)),
        map(LRO.matrix_indices(solver.model)) do i
            dim = LRO.side_dimension(solver.model, i)
            FactoredMatrix(zeros(T, dim, dim), zeros(T, dim, dim), zeros(T, dim, dim))
        end,
    )
    solver.Si = Matrix{T}[]
    solver.DDsi = Vector{T}[]

    solver.Rd = Matrix{T}[]
    solver.Rc = Matrix{T}[]

    solver.alpha = zeros(LRO.num_matrices(solver.model))
    solver.beta = zeros(LRO.num_matrices(solver.model))

    solver.Xn = Matrix{T}[]
    solver.Sn = Matrix{T}[]
    solver.RNT = Matrix{T}[]

    solver.regcount = 0
 
    for mat_idx in LRO.matrix_indices(solver.model)
        dim = LRO.side_dimension(solver.model, mat_idx)
        push!(solver.S,zeros(dim, dim))
        push!(solver.delX,zeros(dim, dim))
        push!(solver.delS,zeros(dim, dim))
        push!(solver.D, zeros(dim))
        push!(solver.Si,zeros(dim, dim))
        push!(solver.DDsi,zeros(dim))
        push!(solver.Rd,zeros(dim, dim))
        push!(solver.Rc,zeros(dim, dim))
        push!(solver.Xn,zeros(dim, dim))
        push!(solver.Sn,zeros(dim, dim))
        push!(solver.RNT,zeros(dim, dim))
    end

    halpha.Umat = Matrix{T}[]
    halpha.Z = Matrix{T}[]
    halpha.AAAATtau = SparseMatrixCSC{T}[]

    for mat_idx in LRO.matrix_indices(solver.model)
        dim = LRO.side_dimension(solver.model, mat_idx)
        push!(halpha.Umat,zeros(dim, solver.erank))
        push!(halpha.Z,zeros(dim, dim))
        # tmp = Matrix(I(dim))
        # push!(halpha.cholS,cholesky(tmp))
        push!(halpha.AAAATtau,spzeros(LRO.num_constraints(solver.model), LRO.num_constraints(solver.model)))
    end

    if solver.kit == 1
        if LRO.num_matrices(solver.model) == 0
            if solver.verb > 0
                println("WARNING: Switching to a direct solver, no LMIs")
            end
            solver.kit = 0
        elseif LRO.num_matrices(solver.model) > 0 && solver.erank >= maximum(Base.Fix1(LRO.side_dimension, solver.model), LRO.matrix_indices(solver.model)) - 1
            if solver.verb > 0
                println("WARNING: Switching to a direct solver, erank bigger than matrix size")
            end
            solver.kit = 0
        end
    end

end

function myIPstep(solver::MySolver{T},halpha::Halpha) where {T}
    mmm = Matrix{T}(undef, LRO.num_constraints(solver.model), LRO.num_constraints(solver.model))
    solver.iter += 1
    if solver.iter > solver.maxit
        solver.status = 4
        if solver.verb > 0
            println("WARNING: Stopped by iteration limit (stopping status = 4)")
        end
    end
    solver.cg_iter_pre = 0
    solver.cg_iter_cor = 0
    
    find_mu(solver)

    # @timeit solver.to "prepare W" begin
    prepare_W(solver)
    # end

    ## predictor
    @timeit solver.to "predictor" begin
    predictor(solver,halpha::Halpha)
    end

    sigma_update(solver)

    ## corrector
    @timeit solver.to "corrector" begin
    corrector(solver,halpha)
    end

end

function find_mu(solver)
    mu = btrace(LRO.num_matrices(solver.model), solver.X, solver.S)

    if LRO.num_scalars(solver.model) > 0
        mu = mu + dot(solver.X[LRO.ScalarIndex], solver.S_lin)
    end
    solver.mu = mu / (LRO.num_scalars(solver.model) + sum(Base.Fix1(LRO.side_dimension, solver.model), LRO.matrix_indices(solver.model), init = 0))
    return solver.mu
end

function check_convergence(solver)

    # DIMACS error evaluation
    b_den = 1 + norm(LRO.cons_constant(solver.model))
    dobj = LRO.dual_obj(solver.model, solver.y)
    solver.err1 = norm(solver.Rp) / b_den
    solver.err2, solver.err3, solver.err4, solver.err5, solver.err6 = 0., 0., 0., 0., 0.
    if LRO.num_matrices(solver.model) > 0
        for mat_idx = LRO.matrix_indices(solver.model)
            i = mat_idx.value
            C_den = 1 + norm(NLPModels.grad(solver.model, mat_idx))
            solver.err2 = solver.err2 + max(0, -eigmin(solver.X[mat_idx]) / b_den)
            solver.err3 = solver.err3 + norm(solver.Rd[i], 2) / C_den
            solver.err4 = solver.err4 + max(0, -eigmin(solver.S[i]) / C_den)
            # err5 = err5 + (vecC[i]"*vec(X[i])-b'*y)/(1+abs(vecC[i]'*vec(X[i]))+abs(b"*y))
            solver.err6 = solver.err6 + dot(solver.S[i], solver.X[mat_idx]) / (1 + abs(NLPModels.obj(solver.model, solver.X[mat_idx], mat_idx)) + abs(dobj))
        end
    end

    pobj = NLPModels.obj(solver.model, solver.X)
    solver.err5 = (dobj - pobj) / (1 + abs(NLPModels.obj(solver.model, solver.X, LRO.MatrixIndex)) + abs(dobj))
    if LRO.num_scalars(solver.model) > 0
        solver.err2 += max(0, -minimum(solver.X[LRO.ScalarIndex]) / b_den)
        solver.err3 += norm(solver.Rd_lin) / (1 + norm(NLPModels.grad(solver.model, LRO.ScalarIndex)))
        solver.err4 += max(0, -minimum(solver.S_lin) / (1 + norm(NLPModels.grad(solver.model, LRO.ScalarIndex))))
        solver.err6 += dot(solver.S_lin', solver.X[LRO.ScalarIndex]) / (1 + abs(NLPModels.obj(solver.model, solver.X, LRO.ScalarIndex)) + abs(dobj))
    end

    DIMACS_error = solver.err2 + solver.err3 + solver.err4 + abs(solver.err5) + solver.err6
    if LRO.num_matrices(solver.model) > 0
        DIMACS_error += solver.err1
    end
    if solver.verb > 0 && solver.status == 0 
        #@sprintf("%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %8.0d %9.0d %8.1e %6.0d %8.2f\n', iter, y[1:ddnvar]"*ddc[:], DIMACS_error, err1, err2, err3, err4, err5, err6, cg_iter1, cg_iter2, eq_norm, arank, titi)
        # @printf("%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %8.0d %9.0d %6.0d\n", iter, dot(y, ctmp'), DIMACS_error, err1, err2, err3, err4, err5, err6, cg_iter1, cg_iter2, cg_iter2)
        if solver.verb > 1
            if solver.kit == 0
                @printf("%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %8.2f\n", solver.iter, dobj, DIMACS_error, solver.err1, solver.err2, solver.err3, solver.err4, solver.err5, solver.err6,solver.itertime)
            else
                @printf("%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %7.0d %7.0d %8.2f\n", solver.iter, dobj, DIMACS_error, solver.err1, solver.err2, solver.err3, solver.err4, solver.err5, solver.err6, solver.cg_iter_pre, solver.cg_iter_cor,solver.itertime)
            end    
    else
            if solver.kit == 0
                @printf("%3.0d %16.8e %9.2e %8.2f\n", solver.iter, dobj, DIMACS_error, solver.itertime)
            else
                @printf("%3.0d %16.8e %9.2e %9.0d %8.2f\n", solver.iter, dobj, DIMACS_error, solver.cg_iter_pre + solver.cg_iter_cor, solver.itertime)
            end
        end
    end

    if DIMACS_error < solver.eDIMACS
        solver.status = 1
        if solver.verb > 0
            println("Primal objective: ", dobj)
            println("Dual objective:   ", pobj)
        end
    end

    if DIMACS_error > 1e55 
        solver.status = 2
        if solver.verb > 0
            println("WARNING: Problem probably infeasible (stopping status = 2)")
        end
    elseif DIMACS_error > 1e55 || abs(dobj) > 1e55
        solver.status = 3
        if solver.verb > 0
            println("WARNING: Problem probably unbounded or infeasible (stopping status = 3)")
        end
    end

end

```Functions for the iterative solver follow```

struct MyA{T,A,B}
    W::LRO.ShapedSolution{T,FactoredMatrix{T}}
    model::LRO.Model{T,A}
    jtprod_buffer::B
    to::TimerOutputs.TimerOutput
end

function (t::MyA)(Ax::Vector, x::Vector)
    @timeit t.to "Ax" begin
        LRO.eval_schur_complement!(t.jtprod_buffer, Ax, t.model, t.W, x)
    end
end

struct MyM_no
    to::TimerOutputs.TimerOutput
end

function (t::MyM_no)(Mx::Vector{T}, x::Vector{T}) where {T}
    copy!(Mx,x)
end

function Prec_for_CG_beta(solver,halpha)    
    
    nlmi = LRO.num_matrices(solver.model)
    kk = solver.erank .* ones(Int64,nlmi,1)  
    nvar = LRO.num_constraints(solver.model)
        
    ntot=0
    if nlmi > 0
        for i in LRO.matrix_indices(solver.model)
            ntot = ntot + size(solver.W[i],1)
        end
    end
    
    halpha.AAAATtau = zeros(nvar)
    if nlmi > 0
        for i in LRO.matrix_indices(solver.model)
            ilmi = i.value
            n = size(solver.W[i],1);
            k = kk[ilmi];
            F = eigen(solver.W[i]); 
            lambdaf = F.values 
            lambda_s = lambdaf[1:n-k]

            if solver.aamat == 0
                ttau = 1.0 * minimum(lambda_s)
            else
                ttau = (minimum(lambda_s) + mean(lambda_s))/2.0 - 1.0e-14
            end
            if solver.aamat < 3
                ZZZ = ones(nvar)
            else
                ZZZ = spzeros(nvar,1)
            end
            
            halpha.AAAATtau += ttau^2 .* ZZZ
        end
        if LRO.num_scalars(solver.model) > 0
            halpha.AAAATtau .+= diag(schur_complement(solver.model, solver.X[LRO.ScalarIndex] .* solver.S_lin_inv, ScalarIndex))
        end
    end
end

struct MyM_beta{T,A}
    model::LRO.Model{T,A}
    AAAATtau
end

function (t::MyM_beta)(Mx::Vector{T}, x::Vector{T}) where {T}
    copy!(Mx, x ./ t.AAAATtau)
end

function Prec_for_CG_tilS_prep(solver::MySolver{T},halpha) where {T} 
    
    @timeit solver.to "prec" begin
    nlmi = LRO.num_matrices(solver.model)
    kk = solver.erank .* ones(Int64,nlmi)
    # kk[2] = 3
    # halpha.Z = SparseMatrixCSC{T}[]
    halpha.Z = Matrix{T}[]

    nvar = LRO.num_constraints(solver.model)
    
    halpha.AAAATtau = spzeros(nvar,nvar)
    
    ntot=0
    if nlmi > 0
        for i in LRO.matrix_indices(solver.model)
            ntot = ntot + size(solver.W[i],1)
        end
    end
    sizeS=0
    if nlmi > 0
        for i in LRO.matrix_indices(solver.model)
            sizeS += kk[i.value] * size(solver.W[i],1)
        end
    end
    
    lbt = 1; lbs=1;
    if nlmi > 0
        for i in LRO.matrix_indices(solver.model)
            ilmi = i.value
            n = size(solver.W[i],1)
            k = kk[ilmi]
            
            F = eigen(Float64.(solver.W[i]))
            vectf = F.vectors
            lambdaf = F.values 

            vect_l = vectf[:,n-k+1:n]
            lambda_l = lambdaf[n-k+1:n]
            vect_s = vectf[:,1:n-k]
            lambda_s = lambdaf[1:n-k]

            if solver.aamat == 0
                ttau = 1.0 * minimum(lambda_s)
            else
                ttau = (minimum(lambda_s) + mean(lambda_s))/2 - 1.0e-14
            end
            
            halpha.Umat[ilmi] = sqrt.(spdiagm(lambda_l) - ttau .* I(k))
            halpha.Umat[ilmi] = vect_l * halpha.Umat[ilmi]
            
            # @timeit solver.to "prec1" begin
            W0 = [vect_s vect_l]*[spdiagm(lambda_s[:]) spzeros(n-k,k); spzeros(k,n-k) ttau * I(k)] * [vect_s vect_l]'
            # end

            # @timeit solver.to "prec2" begin
            W0 = (W0 + W0') ./ 2
            Ztmp = cholesky(2 .* W0 + halpha.Umat[ilmi] * halpha.Umat[ilmi]')
            push!(halpha.Z,Ztmp.L)
            # end
            
            if solver.aamat < 3
                ZZZ = spdiagm(ones(nvar))
            else
                ZZZ = spzeros(nvar,nvar)
            end
            halpha.AAAATtau .+= ttau^2 .* ZZZ
        end
    end
    
    if LRO.num_scalars(solver.model) > 0
        halpha.AAAATtau .+= schur_complement(solver.model, solver.X[LRO.ScalarIndex] .* solver.S_lin_inv, ScalarIndex)
    end
    
    k = kk[1]
    if k > 1 #slow formula
        # @timeit solver.to "prec3" begin
        t = zeros(nvar, k*ntot)
        if nlmi > 0
            for mat_idx in LRO.matrix_indices(solver.model)
                ilmi = mat_idx.value
                n = size(solver.W[mat_idx],1)
                k = kk[ilmi]
                TT = kron(halpha.Umat[ilmi],halpha.Z[ilmi])
                t[1:nvar,lbt:lbt+k*n-1] .= jac(solver.model, mat_idx)' * TT
                lbt = lbt + k*n
            end
        end
        # end
        
        @timeit solver.to "prec4" begin
        S = t' * (halpha.AAAATtau\t) 
        end 
    else #fast formula
        AAAATtau_d = spdiagm(sqrt.(1 ./ diag(halpha.AAAATtau)));

        # @timeit solver.to "prec3" begin
        # t = zeros(nvar, k*ntot)
        # if nlmi > 0
        #     for ilmi = 1:nlmi
        #         if kk[ilmi] == 0
        #             continue 
        #         end
        #         n = size(solver.W[ilmi],1) 
        #         k = kk[ilmi] 
        #         AAs = AAAATtau_d * solver.model.AA[ilmi]
        #         ii_, jj_, aa_ = findnz(AAs)
        #         qq_ = floor.(Int64,(jj_ .- 1) ./ n) .+ 1
        #         pp_ = mod.(jj_ .- 1, n) .+ 1
        #         UU = halpha.Umat[ilmi][qq_]
        #         aau = aa_ .* UU
        #         AU = sparse(ii_,pp_,aau,nvar,n)
        #         if nlmi>1
        #             t[1:nvar,lbt:lbt+k*n-1] .= AU * halpha.Z[ilmi]
        #         else
        #             t .= AU * halpha.Z[1]
        #         end
        #         lbt = lbt + k*n
        #     end 
        # end
        # S .= (t' * t)
        # # mul!(S,t',t)
        # end

        S, lbt = prec_alpha_S!(solver,halpha,AAAATtau_d,kk,ntot,lbt,sizeS)
    end
    
    # Schur complement for the SMW formula
    S = (S + S') ./ 2 + I(size(S,1))
    halpha.cholS = cholesky(S)

    end
       
end

struct MyM{T,A,B}
    model::LRO.Model{T,A}
    jtprod_buffer::B
    AAAATtau
    Umat
    Z
    cholS
end

function prec_alpha_S!(solver::MySolver{T},halpha,AAAATtau_d,kk,didi,lbt,sizeS) where {T}
    @timeit solver.to "prec3" begin
    S = Matrix{T}(undef,sizeS,sizeS)
    nvar = LRO.num_constraints(solver.model)
    t = Matrix{T}(undef,nvar,kk[1]*didi)
    if LRO.num_matrices(solver.model) > 0
        for mat_idx in LRO.matrix_indices(solver.model)
            ilmi = mat_idx.value
            if kk[ilmi] == 0
                continue 
            end
            n = size(solver.W[mat_idx],1) 
            k = kk[ilmi] 

            @timeit solver.to "prec30" begin
                # We can reuse the buffer for different `i`
                # because we directly apply the multiplication
                # with `Umat`.
                AU = reduce(vcat, [
                    (LRO.jtprod!(
                        solver.jtprod_buffer[ilmi],
                        solver.model,
                        mat_idx,
                        AAAATtau_d[i,:],
                    ) * halpha.Umat[ilmi])'
                    for i in axes(AAAATtau_d, 1)
                ])
            end
            if LRO.num_matrices(solver.model) > 1
                # @timeit solver.to "prec32" begin
                didi1 = size(solver.W[mat_idx],1)
                ttmp = Matrix{T}(undef,nvar,kk[ilmi]*didi1)
                mul!(ttmp, AU, halpha.Z[ilmi])
                t[1:nvar,lbt:lbt+k*n-1] = ttmp
                # end
            else
                # @timeit solver.to "prec32" begin
                mul!(t, AU, halpha.Z[1])
                # end
            end
            lbt = lbt + k*n
        end 
    end
    @timeit solver.to "prec33" begin
    mul!(S , t', t)
    end
end

return S, lbt
end

function (t::MyM)(Mx::Vector{T}, x::Vector{T}) where {T}

    nvar = size(x,1)
    nlmi = LRO.num_matrices(t.model)

    yy2 = zeros(nvar,1)
    y33 = zeros(T,0)

    AAAAinvx = t.AAAATtau\x

    if nlmi > 0
        for mat_idx in LRO.matrix_indices(t.model)
            ilmi = mat_idx.value
            y22 = -LRO.jtprod!(t.jtprod_buffer[ilmi], t.model, mat_idx, AAAAinvx)
            y33 = [y33; vec(t.Z[ilmi]' * y22 * t.Umat[ilmi])]
        end
    end
    
    y33 = t.cholS \ y33

    ii = 0
    if nlmi > 0
        for mat_idx in LRO.matrix_indices(t.model)
            ilmi = mat_idx.value
            n = size(t.Umat[ilmi],1)
            k = size(t.Umat[ilmi],2)
            yy = zeros(n*n)
            for i = 1:k
                xx = t.Z[ilmi] * y33[ii+1:ii+n]
                yy .+= kron(t.Umat[ilmi][:,i],xx)
                ii += n
            end
            LRO.add_jprod!(t.model, mat_idx, yy, yy2)
        end
    end

    yyy2 = t.AAAATtau \ yy2

    copy!(Mx,(AAAAinvx - yyy2)[:])

end

end #module
