module Solvers


export MySolver, solve, load, MyA, Halpha

using SparseArrays
using LinearAlgebra
import Statistics: mean
using Printf
using TimerOutputs
using MultiFloats
using ConjugateGradients
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
# Sparse/dense multiplication requires native indices on 32-bit Julia.
_native_sparse_indices(A) = A
function _native_sparse_indices(A::SparseMatrixCSC{T}) where {T}
    return convert(SparseMatrixCSC{T,Int}, A)
end
# Multiply through the dense `matrix` field so we hit BLAS instead of the
# generic `AbstractMatrix` (getindex-based) fallback.
Base.:*(A::FactoredMatrix, B::AbstractMatrix) = A.matrix * _native_sparse_indices(B)
Base.:*(A::AbstractMatrix, B::FactoredMatrix) = _native_sparse_indices(A) * B.matrix
Base.:*(A::FactoredMatrix, B::FactoredMatrix) = A.matrix * B.matrix
function LinearAlgebra.mul!(
    C::AbstractMatrix,
    A::FactoredMatrix,
    B::AbstractVecOrMat,
    α::Number,
    β::Number,
)
    return LinearAlgebra.mul!(C, A.matrix, _native_sparse_indices(B), α, β)
end
function LinearAlgebra.mul!(
    C::AbstractMatrix,
    A::AbstractVecOrMat,
    B::FactoredMatrix,
    α::Number,
    β::Number,
)
    return LinearAlgebra.mul!(C, _native_sparse_indices(A), B.matrix, α, β)
end
function LinearAlgebra.mul!(
    C::AbstractMatrix,
    A::FactoredMatrix,
    B::FactoredMatrix,
    α::Number,
    β::Number,
)
    return LinearAlgebra.mul!(C, A.matrix, B.matrix, α, β)
end

mutable struct MySolver{T,M}
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
    initpoint::Int64
    timing::Int64
    maxit::Int64
    datasparsity::Int64

    to::TimerOutput

    # model and preprocessed model data
    model::M

    predict::Bool

    # current iterate
    sigma::T
    tau::T
    mu::T
    expon::T
    iter::Int64
    DIMACS_error::T
    BBBB::Matrix{T}
    chol_work::Matrix{T} # reusable storage for the dense Schur factorization
    cholBBBB::Cholesky{T,Matrix{T}}

    status::Int

    regcount::Int

    err::NTuple{6,T}

    X::LRO.VectorizedConeVector{T}
    S::LRO.VectorizedConeVector{T}
    y::Vector{T}
    yold::Vector{T}
    delX::Vector{Matrix{T}}
    delS::Vector{Matrix{T}}
    dely::Vector{T}
    Xn::Vector{Matrix{T}}
    Sn::Vector{Matrix{T}}

    Si_lin::Vector{T}
    S_lin_inv::Vector{T}
    delX_lin::Vector{T}
    delS_lin::Vector{T}
    Xn_lin::Vector{T}
    Sn_lin::Vector{T}

    D::Vector{Vector{T}}
    W::LRO.ShapedConeVector{T,FactoredMatrix{T}}
    Si::Vector{Matrix{T}}
    DDsi::Vector{Vector{T}}

    Rp::Vector{T}
    Rd::LRO.VectorizedConeVector{T}

    cg_iter_pre::Int64
    cg_iter_cor::Int64
    cg_iter_tot::Int64
    cg_data::CGData{T}

    alpha::Vector{T}
    beta::Vector{T}
    alpha_lin::T
    beta_lin::T

    itertime::Any
    tottime::Float64

    RNT::Vector{Matrix{T}}
    RNT_lin::Vector{T}

    y_buffer::Vector{T} # Bufer of the same size as `y`
    # Per-block dim×dim scratch reused by `my_kron!` and the matrix products in
    # `predictor`/`corrector`/`find_step`, avoiding temporary matrix products.
    kron_tmp::Vector{Matrix{T}}
    eigen_workspaces::Vector{LRO.EigenWorkspace{T}}
    kron_tmp2::Vector{Matrix{T}}
    # Per-block dim×2 scratch for the products of `ScaledDirection`
    step_tmp::Vector{Matrix{T}}
    # Newton RHS buffer; reused as block scratch after its Jacobian product.
    sol_buffer::LRO.VectorizedConeVector{T}

    function MySolver{T}(
        kit::Int64,
        tol_cg::Float64,
        tol_cg_up::Float64,
        tol_cg_min::Float64,
        eDIMACS::Float64,
        preconditioner::Int64,
        erank::Int64,
        aamat::Int64,
        fig_ev::Int64,
        verb::Int64,
        initpoint::Int64,
        timing::Int64,
        maxit::Int64,
        datasparsity::Int64,
        model::LRO.DualModel{T,A},
    ) where {T,A}

        buffered_model = LRO.BufferedDualModel(model, datasparsity)
        solver = new{T,typeof(buffered_model)}()
        solver.kit = kit
        solver.tol_cg = tol_cg
        solver.tol_cg_up = tol_cg_up
        solver.tol_cg_min = tol_cg_min
        solver.eDIMACS = eDIMACS
        solver.preconditioner = preconditioner
        solver.erank = erank
        solver.aamat = aamat
        solver.fig_ev = fig_ev
        solver.verb = verb
        solver.initpoint = initpoint
        solver.timing = timing
        solver.maxit = maxit
        solver.datasparsity = datasparsity
        solver.model = buffered_model
        return solver
    end
end

struct RankOneBlock{T}
    AU::SparseMatrixCSC{T,Int}
    contraction::SparseMatrixCSC{T,Int}
end

mutable struct MyM{T,M,F,C}
    const model::M
    AAAATtau_fact::F
    const Umat::Vector{Matrix{T}}
    const Z::Vector{Matrix{T}}
    cholS::C
    const AAAAinvx::Vector{T}
    const yy2::Vector{T}
    const yyy2::Vector{T}
    const y33::Vector{T}
    const AyU::Vector{Matrix{T}}
    const chunk::Vector{Matrix{T}}
    const xx::Vector{Vector{T}}
    const yy::Vector{Matrix{T}}
end

mutable struct Halpha{T,M,F}
    kit::Int64
    Umat::Vector{Matrix{T}}
    Z::Vector{Matrix{T}}
    cholS::Cholesky{T,Matrix{T}}
    AAAATtau::Union{Nothing,Matrix{T},SparseMatrixCSC{T,Int}}
    AAAATtau_diag::Vector{T}
    AAAATtau_fact::F
    cg_preconditioner::Union{Nothing,MyM{T,M,F,Cholesky{T,Matrix{T}}}}
    cg_assembly::Union{Nothing,NTuple{3,Matrix{T}}}
    cg_rank_one::Union{Nothing,Vector{RankOneBlock{T}}}

    function Halpha{T,M,F}(kit::Int64) where {T,M,F}
        return new{T,M,F}(
            kit,
            Matrix{T}[],
            Matrix{T}[],
            Cholesky(zeros(T, 0, 0), 'U', 0),
            nothing,
            T[],
            nothing,
            nothing,
            nothing,
            nothing,
        )
    end
end

function Halpha(solver::MySolver{T,M}) where {T,M}
    F = Union{Nothing,Diagonal{T,Vector{T}}}
    if LRO.num_scalars(solver.model) > 0
        factor = T == Float64 ? SparseArrays.CHOLMOD.Factor{T,Int} : Cholesky{T,Matrix{T}}
        F = Union{F,factor}
    end
    return Halpha{T,M,F}(solver.kit)
end

struct Solver{T,M,H} <: SolverCore.AbstractOptimizationSolver
    model::LRO.Model{T}
    solver::MySolver{T,M}
    halpha::H
    stats::SolverCore.GenericExecutionStats{T,Vector{T},Vector{T},Any}
end

function Solver{T}(src::Union{LRO.Model,LRO.DualModel}; kws...) where {T}
    model = LRO.primal_model(src)
    dual = LRO.dual_model(model)
    options = Dict(kw[1] => kw[2] for kw in kws)
    stats = SolverCore.GenericExecutionStats(model)
    solver, halpha = load(dual, options; T)
    solver.X = LRO.VectorizedConeVector(zeros(T, dual.meta.nvar), dual.dim)
    solver.y = stats.solution
    return Solver(model, solver, halpha, stats)
end

function LRO.MOI.get(solver::Solver, ::LRO.DualSolution)
    return LRO.VectorizedConeVector(-solver.stats.multipliers, solver.model.dim)
end

const STATUS_MAP = [
    :unknown,
    :first_order,
    :infeasible, # An unbounded cone-variable dual certifies primal infeasibility.
    :unbounded, # An infeasible cone-variable dual may certify primal unboundedness.
    :max_iter,
    :exception,
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
        if field == :datasparsity && kw[2] != solver.solver.datasparsity
            # The Schur buffer depends on `datasparsity`.
            model = solver.solver.model
            model.schur_buffer = LRO.buffer_for_schur_complement(model.model, kw[2])
        end
        setproperty!(solver.solver, field, kw[2])
    end
    solver.solver.to = TimerOutput()
    solve(solver.solver, solver.halpha)
    solver.stats.status = STATUS_MAP[solver.solver.status+1]
    SolverCore.set_objective!(solver.stats, NLPModels.obj(solver.model, solver.solver.y))
    SolverCore.set_solver_specific!(
        solver.stats,
        :dual_objective,
        -NLPModels.obj(solver.solver.model, solver.solver.X),
    )
    SolverCore.set_solution!(solver.stats, solver.solver.y)
    SolverCore.set_constraint_multipliers!(
        solver.stats,
        -LRO.cone_multipliers(solver.model, solver.solver.X),
    )
    solver.stats.elapsed_time = solver.solver.tottime
    solver.stats.iter = solver.solver.iter
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
    initpoint = Int64(get(options, "initpoint", 0))
    timing = Int64(get(options, "timing", 1))
    maxit = Int64(get(options, "maxit", 100))
    datasparsity = Int64(get(options, "datasparsity", 8))

    solver = MySolver{T}(
        kit,
        tol_cg,
        tol_cg_up,
        tol_cg_min,
        eDIMACS,
        preconditioner,
        erank,
        aamat,
        fig_ev,
        verb,
        initpoint,
        timing,
        maxit,
        datasparsity,
        model,
    )

    halpha = Halpha(solver)
    solver.cg_iter_tot = 0

    # Input parameters check
    if kit < 0 || kit > 1
        solver.kit = 0
        @printf(" ---Parameter kit out of range, setting kit = %1d\n", solver.kit)
    end
    if tol_cg < tol_cg_min && kit == 1
        solver.tol_cg = tol_cg_min
        @printf(
            " ---Parameter tol_cg smaller than tol_cg_min, setting tol_cg = %7.1e\n",
            solver.tol_cg
        )
    end
    if tol_cg_min > eDIMACS && kit == 1
        solver.tol_cg_min = eDIMACS
        @printf(" ---Parameter tol_cg_min switched to eDIMACS = %7.1e\n", eDIMACS)
    end
    if kit == 1 && (preconditioner < 0 || preconditioner > 4)
        solver.preconditioner = 1
        @printf(
            " ---Parameter preconditioner out of range, setting preconditioner = %1d\n",
            solver.preconditioner
        )
    end
    if erank < 0
        solver.erank = 1
        @printf(" ---Parameter erank negative, setting erank = %1d\n", solver.erank)
    end
    if initpoint < 0 || initpoint > 1
        solver.initpoint = 1
        @printf(
            " ---Parameter kit out of range, setting initpoint = %1d\n",
            solver.initpoint
        )
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

function solve(solver::MySolver, halpha::Halpha)
    t1 = time()
    if solver.verb > 0
        @printf(
            "\n *** Loraine.jl v%s ***\n",
            string(pkgversion(parentmodule(@__MODULE__)))
        )

        @printf(" Number of variables: %5d\n", solver.model.meta.ncon)
        @printf(" LMI constraints    : %5d\n", LRO.num_matrices(solver.model))
        if LRO.num_matrices(solver.model) > 0
            @printf(" Matrix size(s)     :")
            msizes =
                LRO.side_dimension.(Ref(solver.model), LRO.matrix_indices(solver.model))
            Printf.format.(Ref(stdout), Ref(Printf.Format("%6d")), msizes)
            @printf("\n")
        end
        @printf(" Linear constraints : %5d\n", LRO.num_scalars(solver.model))
        if solver.kit>0
            @printf(" Preconditioner     : %5d\n", solver.preconditioner)
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
                @printf(
                    " it        obj         error      err1      err2      err3      err4      err5      err6     CPU/it\n"
                )
                @printf(
                    "                                p-eq-con   p-feas   d-eq-con   d-feas     d-gap  slackness         \n"
                )
                @printf(
                    "---------------------------------------------------------------------------------------------------\n"
                )
            else
                @printf(
                    " it        obj         error      err1      err2      err3      err4      err5      err6    cg_pre  cg_cor  CPU/it\n"
                )
            end
        end
    end

    setup_solver(solver::MySolver, halpha::Halpha)

    initial_point(solver)

    while solver.status == 0

        t2 = time()
        myIPstep(solver, halpha)
        solver.itertime = time()-t2

        solver.tol_cg = max(solver.tol_cg * solver.tol_cg_up, solver.tol_cg_min)

        check_convergence(solver)

        if solver.preconditioner == 4
            #         if (cg_iter2>erank*nlmi*sqrt(n)/1 && iter>sqrt(n)/60)||cg_iter2>100 %for SNL problems
            if (
                solver.cg_iter_cor / 2 >
                solver.erank *
                LRO.num_matrices(solver.model) *
                sqrt(solver.model.meta.ncon)/20 &&
                solver.iter > sqrt(solver.model.meta.ncon) / 60
            ) || solver.cg_iter_cor > 100
                solver.preconditioner = 1
                solver.aamat = 2
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

function setup_solver(solver::MySolver{T}, halpha::Halpha) where {T}

    solver.S = similar(solver.X)
    solver.Rd = similar(solver.X)
    solver.Rp = zeros(T, solver.model.meta.ncon)
    solver.y_buffer = similar(solver.Rp)
    solver.dely = similar(solver.Rp)
    solver.yold = similar(solver.y)
    solver.cg_data = CGData(length(solver.y), T)

    solver.delX = Matrix{T}[]
    solver.delS = Matrix{T}[]

    solver.D = Vector{T}[]
    solver.W = LRO.ShapedConeVector{T,FactoredMatrix{T}}(
        zeros(T, LRO.num_scalars(solver.model)),
        map(LRO.matrix_indices(solver.model)) do i
            dim = LRO.side_dimension(solver.model, i)
            FactoredMatrix(zeros(T, dim, dim), zeros(T, dim, dim), zeros(T, dim, dim))
        end,
    )
    solver.Si = Matrix{T}[]
    solver.DDsi = Vector{T}[]

    solver.alpha = zeros(LRO.num_matrices(solver.model))
    solver.beta = zeros(LRO.num_matrices(solver.model))

    solver.Xn = Matrix{T}[]
    solver.Sn = Matrix{T}[]
    solver.RNT = Matrix{T}[]

    solver.kron_tmp = Matrix{T}[]
    solver.kron_tmp2 = Matrix{T}[]
    solver.step_tmp = Matrix{T}[]
    solver.eigen_workspaces = LRO.EigenWorkspace{T}[]
    solver.sol_buffer = similar(solver.X)

    solver.regcount = 0

    solver.delS_lin = zeros(T, LRO.num_scalars(solver.model))
    solver.Si_lin = similar(solver.delS_lin)
    solver.delX_lin = similar(solver.delS_lin)
    solver.Xn_lin = similar(solver.delS_lin)
    solver.Sn_lin = similar(solver.delS_lin)
    solver.RNT_lin = similar(solver.delS_lin)
    for mat_idx in LRO.matrix_indices(solver.model)
        dim = LRO.side_dimension(solver.model, mat_idx)
        push!(solver.delX, zeros(T, dim, dim))
        push!(solver.delS, zeros(T, dim, dim))
        push!(solver.D, zeros(T, dim))
        push!(solver.Si, zeros(T, dim, dim))
        push!(solver.DDsi, zeros(T, dim))
        push!(solver.Xn, zeros(T, dim, dim))
        push!(solver.Sn, zeros(T, dim, dim))
        push!(solver.RNT, zeros(T, dim, dim))
        push!(solver.kron_tmp, zeros(T, dim, dim))
        push!(solver.kron_tmp2, zeros(T, dim, dim))
        push!(solver.step_tmp, zeros(T, dim, 2))
        push!(solver.eigen_workspaces, LRO.EigenWorkspace(solver.kron_tmp[end]))
    end

    halpha.Umat = Matrix{T}[]
    halpha.cg_preconditioner = nothing
    halpha.cg_assembly = nothing
    halpha.cg_rank_one = nothing
    halpha.Z = Matrix{T}[]
    ncon = solver.model.meta.ncon
    halpha.AAAATtau = nothing
    halpha.AAAATtau_diag = zeros(T, ncon)
    halpha.AAAATtau_fact = nothing

    for mat_idx in LRO.matrix_indices(solver.model)
        dim = LRO.side_dimension(solver.model, mat_idx)
        push!(halpha.Umat, zeros(T, dim, solver.erank))
        push!(halpha.Z, zeros(T, dim, dim))
        # tmp = Matrix(I(dim))
        # push!(halpha.cholS,cholesky(tmp))
    end

    if solver.kit == 1
        if LRO.num_matrices(solver.model) == 0
            if solver.verb > 0
                @warn("Switching to a direct solver, no LMIs")
            end
            solver.kit = 0
        elseif LRO.num_matrices(solver.model) > 0 &&
               solver.erank >=
               maximum(
            Base.Fix1(LRO.side_dimension, solver.model),
            LRO.matrix_indices(solver.model),
        ) - 1
            if solver.verb > 0
                @warn("Switching to a direct solver, erank bigger than matrix size")
            end
            solver.kit = 0
        end
    end

    if solver.kit == 0   # if direct solver; compute the Hessian matrix
        solver.BBBB = zeros(T, ncon, ncon)
        solver.chol_work = zeros(T, ncon, ncon)
    end
    if solver.kit == 1 && solver.preconditioner in (1, 4)
        buffers = cg_assembly!(halpha, T, ncon, sum(length, halpha.Umat))
        init_AAAATtau!(halpha, solver.model, T)
        # Placeholder factors are replaced before the first CG application.
        halpha.cholS = Cholesky(buffers[2], 'U', 0)
        halpha.cg_preconditioner = make_preconditioner(halpha, solver.model)
        if solver.erank == 1 && solver.preconditioner == 1
            cg_rank_one!(halpha, solver.model)
        end
    end

end

function myIPstep(solver::MySolver{T}, halpha::Halpha) where {T}
    solver.iter += 1
    if solver.iter > solver.maxit
        solver.status = 4
        if solver.verb > 0
            @warn("Stopped by iteration limit (stopping status = 4)")
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
        predictor(solver, halpha::Halpha)
    end

    sigma_update(solver)

    ## corrector
    @timeit solver.to "corrector" begin
        corrector(solver, halpha)
    end

end

function find_mu(solver)
    solver.mu =
        dot(solver.X, solver.S) / (
            LRO.num_scalars(solver.model) + sum(
                Base.Fix1(LRO.side_dimension, solver.model),
                LRO.matrix_indices(solver.model),
                init = 0,
            )
        )
    return solver.mu
end

function check_convergence(solver)

    # DIMACS error evaluation
    pobj = NLPModels.obj(solver.model, solver.X)
    dobj = LRO.dual_obj(solver.model, solver.y)
    solver.err = LRO.errors(
        solver.model,
        solver.X;
        y = solver.y,
        primal_err = solver.Rp,
        dual_slack = solver.S,
        dual_err = solver.Rd,
        pobj = pobj,
        dobj = dobj,
        workspace = solver.eigen_workspaces,
    )

    # `err[5]` may be negative so we need `abs`
    DIMACS_error = sum(abs, solver.err)
    if solver.verb > 0 && solver.status == 0
        #@sprintf("%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %8.0d %9.0d %8.1e %6.0d %8.2f\n', iter, y[1:ddnvar]"*ddc[:], DIMACS_error, err[1], err[2], err[3], err[4], err[5], err[6], cg_iter1, cg_iter2, eq_norm, arank, titi)
        # @printf("%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %8.0d %9.0d %6.0d\n", iter, dot(y, ctmp'), DIMACS_error, err[1], err[2], err[3], err[4], err[5], err[6], cg_iter1, cg_iter2, cg_iter2)
        if solver.verb > 1
            if solver.kit == 0
                @printf(
                    "%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %8.2f\n",
                    solver.iter,
                    dobj,
                    DIMACS_error,
                    solver.err[1],
                    solver.err[2],
                    solver.err[3],
                    solver.err[4],
                    solver.err[5],
                    solver.err[6],
                    solver.itertime
                )
            else
                @printf(
                    "%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %7.0d %7.0d %8.2f\n",
                    solver.iter,
                    dobj,
                    DIMACS_error,
                    solver.err[1],
                    solver.err[2],
                    solver.err[3],
                    solver.err[4],
                    solver.err[5],
                    solver.err[6],
                    solver.cg_iter_pre,
                    solver.cg_iter_cor,
                    solver.itertime
                )
            end
        else
            if solver.kit == 0
                @printf(
                    "%3.0d %16.8e %9.2e %8.2f\n",
                    solver.iter,
                    dobj,
                    DIMACS_error,
                    solver.itertime
                )
            else
                @printf(
                    "%3.0d %16.8e %9.2e %9.0d %8.2f\n",
                    solver.iter,
                    dobj,
                    DIMACS_error,
                    solver.cg_iter_pre + solver.cg_iter_cor,
                    solver.itertime
                )
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

    if pobj < -1e55
        solver.status = 2
        if solver.verb > 0
            @warn("Problem probably infeasible (stopping status = 2)")
        end
    elseif dobj > 1e55
        solver.status = 3
        if solver.verb > 0
            @warn("Problem probably unbounded (stopping status = 3)")
        end
    elseif any(isnan, solver.err)
        solver.status = 5
        if solver.verb > 0
            @warn("Got NaN (stopping status = 5)")
        end
    end

end

```Functions for the iterative solver follow```

struct MyA{T,M}
    W::LRO.ShapedConeVector{T,FactoredMatrix{T}}
    model::M
    to::TimerOutputs.TimerOutput
end

function (t::MyA)(Ax::Vector, x::Vector)
    @timeit t.to "Ax" begin
        LRO.eval_schur_complement!(t.model, t.W, x, Ax)
    end
end

struct MyM_no
    to::TimerOutputs.TimerOutput
end

function (t::MyM_no)(Mx::Vector{T}, x::Vector{T}) where {T}
    copy!(Mx, x)
end

function Prec_for_CG_beta(solver::MySolver{T}, halpha) where {T}

    nlmi = LRO.num_matrices(solver.model)
    kk = solver.erank .* ones(Int64, nlmi, 1)
    nvar = solver.model.meta.ncon

    ntot=0
    if nlmi > 0
        for i in LRO.matrix_indices(solver.model)
            ntot = ntot + size(solver.W[i], 1)
        end
    end

    fill!(halpha.AAAATtau_diag, zero(T))
    if nlmi > 0
        for i in LRO.matrix_indices(solver.model)
            ilmi = i.value
            n = size(solver.W[i], 1)
            k = kk[ilmi]
            F = eigen(solver.W[i])
            lambdaf = F.values
            lambda_s = lambdaf[1:(n-k)]

            if isempty(lambda_s)
                # The selected rank covers this block (e.g. a 1x1 PSD block).
                # Keep a positive baseline; for 1x1 this gives W0 = W and U = 0.
                ttau = minimum(lambdaf)
            elseif solver.aamat == 0
                ttau = minimum(lambda_s)
            else
                ttau = (minimum(lambda_s) + mean(lambda_s)) / 2 - T(1e-14)
            end
            if solver.aamat < 3
                halpha.AAAATtau_diag .+= ttau^2
            end
        end
        if LRO.num_scalars(solver.model) > 0
            tmp = zeros(eltype(halpha.AAAATtau_diag), nvar, nvar)
            LRO.add_schur_complement!(
                solver.model,
                solver.X[LRO.ScalarIndex] .* solver.S_lin_inv,
                LRO.ScalarIndex,
                tmp,
            )
            halpha.AAAATtau_diag .+= diag(tmp)
        end
    end
end

struct MyM_beta{M,V}
    model::M
    AAAATtau::V
end

function (t::MyM_beta)(Mx::Vector{T}, x::Vector{T}) where {T}
    @. Mx = x / t.AAAATtau
end

function cg_assembly!(halpha, ::Type{T}, nvar, sizeS) where {T}
    buffers = halpha.cg_assembly
    if isnothing(buffers) || size(buffers[1]) != (nvar, sizeS)
        buffers = (zeros(T, nvar, sizeS), zeros(T, sizeS, sizeS), zeros(T, nvar, sizeS))
        halpha.cg_assembly = buffers
    end
    return buffers
end

# The pattern of A_j*u is independent of u. Cache its sparse rows and the
# contraction that fills their nonzeros, avoiding sparse assembly each iteration.


function RankOneBlock(model, i)
    J = model.jprod_buffer[i.value]
    n = Int(LRO.side_dimension(model, i))
    nvar = Int(model.meta.ncon)
    entries, constraints, coefficients = findnz(J)
    rows = mod.(entries .- 1, n) .+ 1
    cols = div.(entries .- 1, n) .+ 1
    AU = sparse(Int.(constraints), Int.(rows), ones(eltype(J), length(entries)), nvar, n)
    # Map each A_j[p,q] to the stored slot for (j,p) in AU.
    slots = SparseMatrixCSC{Int,Int}(AU)
    nonzeros(slots) .= eachindex(nonzeros(slots))
    targets = getindex.(Ref(slots), constraints, rows)
    contraction = sparse(targets, Int.(cols), coefficients, nnz(AU), n)
    return RankOneBlock(AU, contraction)
end

function cg_rank_one!(halpha, model)
    blocks = halpha.cg_rank_one
    if isnothing(blocks)
        blocks = [RankOneBlock(model, i) for i in LRO.matrix_indices(model)]
        halpha.cg_rank_one = blocks
    end
    return blocks
end

function rank_one_product!(t, block::RankOneBlock, U, Z, weights)
    values = nonzeros(block.AU)
    mul!(values, block.contraction, view(U, :, 1))
    row_weights = view(weights, rowvals(block.AU))
    @. values *= -row_weights
    mul!(t, block.AU, Z)
    return t
end

# W = W0 + U*U', so the Cholesky factor for the preconditioner is
# chol(2W0 + U*U') = chol(2W - U*U'). Reuse the spectral and factor buffers.
function cg_scaling!(U, Z, workspace, W, aamat)
    values, vectors = LRO.eigen!(workspace, copyto!(workspace.scratch, W); vectors = true)
    n, k = size(U)
    small = view(values, 1:(n-k))
    tau = if isempty(small)
        minimum(values)
    elseif aamat == 0
        minimum(small)
    else
        (minimum(small) + mean(small)) / 2 - 1.0e-14
    end
    large = (n-k+1):n
    U .= view(vectors, :, large) .* sqrt.(transpose(view(values, large)) .- tau)
    mul!(Z, U, U', -one(eltype(Z)), zero(eltype(Z)))
    @. Z += 2 * W
    LinearAlgebra.hermitianpart!(Z)
    cholesky!(Hermitian(Z, :L))
    tril!(Z)
    return tau
end

# `AAAATtau = baseline * I + C_lin * Diagonal(X_lin ./ S_lin) * C_lin'` where
# `C_lin` is the Jacobian of the scalar constraints. When each scalar
# constraint involves at most one entry of `y`, e.g., bounds as in the `vib`
# and `tru` problems, it is diagonal and `AAAATtau` is `nothing` while
# `AAAATtau_fact` is the `Diagonal`. Otherwise, `AAAATtau` is sparse with the
# sparsity pattern of `C_lin * C_lin'` and `AAAATtau_fact` is its sparse
# Cholesky factorization, whose symbolic analysis is reused.
# When this pattern is dense, a dense `Matrix` would make the factorization
# faster but not the triangular solves of each CG iteration so it does not seem
# worth it for now. CHOLMOD only supports `Float64` so we use a dense `Matrix`
# for other types.
function init_AAAATtau!(halpha, model, ::Type{T}) where {T}
    if halpha.AAAATtau isa AbstractMatrix{T} ||
       (isnothing(halpha.AAAATtau) && halpha.AAAATtau_fact isa Diagonal{T})
        return
    end
    C = LRO.jac(model.model, LRO.ScalarIndex)
    if all(k -> length(nzrange(C, k)) <= 1, axes(C, 2))
        halpha.AAAATtau = nothing
        halpha.AAAATtau_fact = Diagonal(zeros(T, model.meta.ncon))
        return
    end
    # The values are positive so no entry of the pattern cancels out
    pattern = SparseMatrixCSC(size(C)..., C.colptr, C.rowval, ones(T, nnz(C)))
    H = _native_sparse_indices(pattern * pattern' + I)
    halpha.AAAATtau = T == Float64 ? H : Matrix(H)
    halpha.AAAATtau_fact = nothing
    return
end

_zero_entries!(H::SparseMatrixCSC) = fill!(nonzeros(H), zero(eltype(H)))
_zero_entries!(H::Matrix) = fill!(H, zero(eltype(H)))

_cholesky!(::Nothing, H) = cholesky(Hermitian(H))
_cholesky!(F::SparseArrays.CHOLMOD.Factor, H) = cholesky!(F, Hermitian(H))
function _cholesky!(F::Cholesky, H::Matrix)
    copyto!(F.factors, H)
    cholesky!(Hermitian(F.factors))
    return F
end

function fill_AAAATtau!(halpha, solver, baseline)
    C = LRO.jac(solver.model.model, LRO.ScalarIndex)
    rows, values = rowvals(C), nonzeros(C)
    X = solver.X[LRO.ScalarIndex]
    H = halpha.AAAATtau
    if isnothing(H)
        d = halpha.AAAATtau_fact.diag
        fill!(d, baseline)
        for k in axes(C, 2)
            w = X[k] * solver.S_lin_inv[k]
            for p in nzrange(C, k)
                d[rows[p]] += w * abs2(values[p])
            end
        end
        return
    end
    _zero_entries!(H)
    for i in axes(H, 1)
        H[i, i] = baseline
    end
    for k in axes(C, 2)
        w = X[k] * solver.S_lin_inv[k]
        for p in nzrange(C, k), q in nzrange(C, k)
            H[rows[q], rows[p]] += w * values[q] * conj(values[p])
        end
    end
    halpha.AAAATtau_fact = _cholesky!(halpha.AAAATtau_fact, H)
    return
end

function preconditioner_weights!(weights, base::AbstractVector)
    weights .= sqrt.(1 ./ base)
    return
end

function preconditioner_weights!(weights, base)
    indices = view(CartesianIndices(base), diagind(base))
    weights .= sqrt.(1 ./ view(base, indices))
    return
end

function factor_preconditioner!(halpha, S)
    # Checked factorization either succeeds with upper storage and info = 0,
    # or throws. Keep the wrapper already referencing this buffer on success.
    cholesky!(Hermitian(S))
    factor = halpha.cholS
    if factor.factors !== S || factor.uplo != 'U' || factor.info != 0
        halpha.cholS = Cholesky(S, 'U', 0)
    end
    return
end

function Prec_for_CG_tilS_prep(solver::MySolver{T}, halpha) where {T}

    @timeit solver.to "prec" begin
        nvar = solver.model.meta.ncon

        init_AAAATtau!(halpha, solver.model, T)
        baseline = zero(T)
        sizeS = sum(length, halpha.Umat; init = 0)

        lbt = 1
        if LRO.num_matrices(solver.model) > 0
            for i in LRO.matrix_indices(solver.model)
                ilmi = i.value
                ttau = cg_scaling!(
                    halpha.Umat[ilmi],
                    halpha.Z[ilmi],
                    solver.eigen_workspaces[ilmi],
                    solver.W[i].matrix,
                    solver.aamat,
                )

                if solver.aamat < 3
                    baseline += ttau^2
                end
            end
        end

        # Factorize once here; `AAAATtau` is constant for the whole predictor/corrector
        # CG solve, so this avoids re-factorizing it on every single CG iteration.
        fill_AAAATtau!(halpha, solver, baseline)

        if solver.erank > 1 #slow formula
            # @timeit solver.to "prec3" begin
            t, S, solved_t = cg_assembly!(halpha, T, nvar, sizeS)
            if LRO.num_matrices(solver.model) > 0
                for mat_idx in LRO.matrix_indices(solver.model)
                    ilmi = mat_idx.value
                    n = size(solver.W[mat_idx], 1)
                    k = size(halpha.Umat[ilmi], 2)
                    TT = kron(halpha.Umat[ilmi], halpha.Z[ilmi])
                    mul!(
                        view(t, :, Int(lbt):Int(lbt+k*n-1)),
                        _native_sparse_indices(solver.model.jprod_buffer[ilmi])',
                        TT,
                    )
                    lbt = lbt + k*n
                end
            end
            # end

            @timeit solver.to "prec4" begin
                ldiv!(solved_t, halpha.AAAATtau_fact, t)
                mul!(S, t', solved_t)
            end
        else #fast formula
            # Reuse the application's scratch; y_buffer holds the live RHS.
            weights = halpha.cg_preconditioner.AAAAinvx
            preconditioner_weights!(
                weights,
                isnothing(halpha.AAAATtau) ? halpha.AAAATtau_fact.diag : halpha.AAAATtau,
            )

            S = prec_alpha_S!(solver, halpha, weights)
        end

        # Schur complement for the SMW formula
        LinearAlgebra.hermitianpart!(S)
        indices = view(CartesianIndices(S), diagind(S))
        view(S, indices) .+= one(T)
        # The previous preconditioner is dead; its factor may share this buffer.
        factor_preconditioner!(halpha, S)

    end

end

# One workspace per predictor/corrector CG solve, reused by every application.


function MyM(model, factor, U::Vector{Matrix{T}}, Z, chol, workspace = nothing) where {T}
    return MyM{T,typeof(model),typeof(factor),typeof(chol)}(
        model,
        factor,
        U,
        Z,
        chol,
        workspace,
    )
end

function make_preconditioner(halpha::Halpha{T,M,F}, model::M) where {T,M,F}
    return MyM{T,M,F,Cholesky{T,Matrix{T}}}(
        model,
        halpha.AAAATtau_fact,
        halpha.Umat,
        halpha.Z,
        halpha.cholS,
        halpha.cg_preconditioner,
    )
end

function MyM{T,M,F,C}(
    model,
    AAAATtau_fact,
    Umat::Vector{Matrix{T}},
    Z::Vector{Matrix{T}},
    cholS,
    workspace = nothing,
) where {T,M,F,C}
    if workspace isa MyM{T,M,F,C} &&
       workspace.model === model &&
       workspace.Umat === Umat &&
       workspace.Z === Z
        workspace.AAAATtau_fact = AAAATtau_fact
        workspace.cholS = cholS
        return workspace
    end
    if !isnothing(workspace)
        # Refresh the factors each iteration, keeping the application buffers.
        return MyM{T,M,F,C}(
            model,
            AAAATtau_fact,
            Umat,
            Z,
            cholS,
            workspace.AAAAinvx,
            workspace.yy2,
            workspace.yyy2,
            workspace.y33,
            workspace.AyU,
            workspace.chunk,
            workspace.xx,
            workspace.yy,
        )
    end
    nvar = model.meta.ncon
    sizeS = sum(length, Umat; init = 0)
    return MyM{T,M,F,C}(
        model,
        AAAATtau_fact,
        Umat,
        Z,
        cholS,
        zeros(T, nvar),
        zeros(T, nvar),
        zeros(T, nvar),
        zeros(T, sizeS),
        [zeros(T, size(u)) for u in Umat],
        [zeros(T, size(u)) for u in Umat],
        [zeros(T, size(u, 1)) for u in Umat],
        [zeros(T, size(z)) for z in Z],
    )
end

function prec_alpha_S!(solver::MySolver{T}, halpha, weights) where {T}
    @timeit solver.to "prec3" begin
        sizeS = sum(length, halpha.Umat; init = 0)
        t, S, _ = cg_assembly!(halpha, T, solver.model.meta.ncon, sizeS)
        blocks = cg_rank_one!(halpha, solver.model)
        offset = 0
        for i in LRO.matrix_indices(solver.model)
            n = size(halpha.Z[i.value], 1)
            # Native ranges keep SparseArrays multiplication valid on 32-bit.
            columns = Int(offset+1):Int(offset+n)
            rank_one_product!(
                view(t, :, columns),
                blocks[i.value],
                halpha.Umat[i.value],
                halpha.Z[i.value],
                weights,
            )
            offset += n
        end
        mul!(S, t', t)
    end
    return S
end

function (t::MyM)(Mx::Vector{T}, x::Vector{T}) where {T}
    ldiv!(t.AAAAinvx, t.AAAATtau_fact, x)
    offset = 0
    for mat_idx in LRO.matrix_indices(t.model)
        i = mat_idx.value
        y22 = LRO.unsafe_jtprod(t.model, t.AAAAinvx, mat_idx)
        # U is thin: preserve the cheaper right-first multiplication order.
        mul!(t.AyU[i], y22, t.Umat[i])
        mul!(t.chunk[i], t.Z[i]', t.AyU[i])
        copyto!(t.y33, offset + 1, t.chunk[i], 1, length(t.chunk[i]))
        offset += length(t.chunk[i])
    end
    ldiv!(t.cholS, t.y33)

    fill!(t.yy2, zero(T))
    offset = 0
    for mat_idx in LRO.matrix_indices(t.model)
        i = mat_idx.value
        n, k = size(t.Umat[i])
        fill!(t.yy[i], zero(T))
        for j = 1:k
            mul!(t.xx[i], t.Z[i], view(t.y33, (offset+1):(offset+n)))
            # Accumulate the outer product without allocating a Kronecker product.
            for col = 1:n, row = 1:n
                t.yy[i][row, col] += t.xx[i][row] * t.Umat[i][col, j]
            end
            offset += n
        end
        LRO.add_jprod!(t.model, t.yy[i], t.yy2, mat_idx)
    end
    ldiv!(t.yyy2, t.AAAATtau_fact, t.yy2)
    @. Mx = t.AAAAinvx - t.yyy2
    return Mx
end

end #module
