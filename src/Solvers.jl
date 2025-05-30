module Solvers

export  MySolver, solve, load, MyA, Halpha

using SparseArrays
using LinearAlgebra
import Statistics: mean
using Printf
using TimerOutputs
using MultiFloats
# using MKLSparse
# using MKL

include("kron_etc.jl")
include("makeBBBB.jl")
include("model.jl")

mutable struct MySolver{T}
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
    model::MyModel

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

    X
    S
    y
    yold
    delX
    delS
    dely
    Xn
    Sn

    X_lin
    S_lin
    Si_lin
    S_lin_inv
    delX_lin
    delS_lin
    Xn_lin
    Sn_lin

    D
    G
    Gi
    W
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
        model::MyModel
        ) where {T}

        solver = new{T}()
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
    datasparsity = Int64(get(options, "maxit", 8))

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
        MyModel(model.A,
            model.AA,
            model.B,
            model.C,
            model.nzA,
            model.sigmaA,
            model.qA,
            model.b,
            model.b_const,
            model.d_lin,
            model.C_lin,
            model.n,
            model.msizes,
            model.nlin,
            model.nlmi
        )
    )

    halpha = Halpha(kit)
    solver.cg_iter_tot = 0

    if verb > 0
        t1 = time()
        @printf("\n *** Loraine.jl v0.2.5 ***\n")
        @printf(" *** Initialisation STARTS\n")
    end

    if verb > 0
        @printf(" Number of variables: %5d\n",model.n)
        @printf(" LMI constraints    : %5d\n",model.nlmi)
        if model.nlmi>0
            @printf(" Matrix size(s)     :")
            Printf.format.(Ref(stdout), Ref(Printf.Format("%6d")), model.msizes);
            @printf("\n")
        end
        @printf(" Linear constraints : %5d\n",model.nlin)
        if solver.kit>0
            @printf(" Preconditioner     : %5d\n",preconditioner)
        else
            @printf(" Preconditioner     :  none, using direct solver\n")
        end
    end

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
            if (solver.cg_iter_cor / 2 > solver.erank * solver.model.nlmi * sqrt(solver.model.n)/20 && solver.iter > sqrt(solver.model.n) / 60) || solver.cg_iter_cor > 100
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

    solver.X = Matrix{T}[]
    solver.S = Matrix{T}[]
    solver.y = Vector{T}[]

    solver.delX = Matrix{T}[]
    solver.delS = Matrix{T}[]

    solver.D = Vector{T}[]
    solver.G = Matrix{T}[]
    solver.Gi = Matrix{T}[]
    solver.W = Matrix{T}[]
    solver.Si = Matrix{T}[]
    solver.DDsi = Vector{T}[]

    solver.Rd = Matrix{T}[]
    solver.Rc = Matrix{T}[]

    solver.alpha = zeros(solver.model.nlmi)
    solver.beta = zeros(solver.model.nlmi)

    solver.Xn = Matrix{T}[]
    solver.Sn = Matrix{T}[]
    solver.RNT = Matrix{T}[]

    solver.regcount = 0
 
    for i = 1:solver.model.nlmi
        push!(solver.X,zeros(solver.model.msizes[i], solver.model.msizes[i]))
        push!(solver.S,zeros(solver.model.msizes[i], solver.model.msizes[i]))
        push!(solver.delX,zeros(solver.model.msizes[i], solver.model.msizes[i]))
        push!(solver.delS,zeros(solver.model.msizes[i], solver.model.msizes[i]))
        push!(solver.D, zeros(solver.model.msizes[i]))
        push!(solver.G,zeros(solver.model.msizes[i],  solver.model.msizes[i]))
        push!(solver.Gi,zeros(solver.model.msizes[i], solver.model.msizes[i]))
        push!(solver.W,zeros(solver.model.msizes[i],  solver.model.msizes[i]))
        push!(solver.Si,zeros(solver.model.msizes[i], solver.model.msizes[i]))
        push!(solver.DDsi,zeros(solver.model.msizes[i]))
        push!(solver.Rd,zeros(solver.model.msizes[i], solver.model.msizes[i]))
        push!(solver.Rc,zeros(solver.model.msizes[i], solver.model.msizes[i]))
        push!(solver.Xn,zeros(solver.model.msizes[i], solver.model.msizes[i]))
        push!(solver.Sn,zeros(solver.model.msizes[i], solver.model.msizes[i]))
        push!(solver.RNT,zeros(solver.model.msizes[i], solver.model.msizes[i]))
    end

    halpha.Umat = Matrix{T}[]
    halpha.Z = Matrix{T}[]
    halpha.AAAATtau = SparseMatrixCSC{T}[]

    for i = 1:solver.model.nlmi
        push!(halpha.Umat,zeros(solver.model.msizes[i], solver.erank))
        push!(halpha.Z,zeros(solver.model.msizes[i], solver.model.msizes[i]))
        # tmp = Matrix(I(solver.model.msizes[i]))
        # push!(halpha.cholS,cholesky(tmp))
        push!(halpha.AAAATtau,spzeros(solver.model.n, solver.model.n))
    end

    if solver.kit == 1
        if solver.model.nlmi == 0
            if solver.verb > 0
                println("WARNING: Switching to a direct solver, no LMIs")
            end
            solver.kit = 0
        elseif solver.model.nlmi > 0 && solver.erank >= maximum(solver.model.msizes) - 1
            if solver.verb > 0
                println("WARNING: Switching to a direct solver, erank bigger than matrix size")
            end
            solver.kit = 0
        end
    end

    # when datarank was set to -1 and conversion failed, we switch to datarank = 0
    if ~isempty(solver.model.B)
        if solver.model.nlmi > 0
            for ilmi = 1:solver.model.nlmi
                if nnz(solver.model.B[ilmi]) == 0
                    solver.datarank = 0
                end
            end
        end
    end

end

function myIPstep(solver::MySolver{T},halpha::Halpha) where {T}
    mmm = Matrix{T}(undef, solver.model.n, solver.model.n)
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
    trXS = 0
    if solver.model.nlmi > 0
        for i = 1:solver.model.nlmi
            trXS = trXS + sum(sum(solver.X[i] .* solver.S[i]))
        end
    end 
    mu = trXS

    if solver.model.nlin > 0
        mu = mu + tr(solver.X_lin' * solver.S_lin)
    end
    solver.mu = mu / (sum(solver.model.msizes) + solver.model.nlin)
    return solver.mu
end

function check_convergence(solver)

    # DIMACS error evaluation
    solver.err1 = norm(solver.Rp) / (1 + norm(solver.model.b))
    (solver.err2,solver.err3,solver.err4,solver.err5,solver.err6) = [0.,0.,0.,0.,0.]
    if solver.model.nlmi > 0
        for i = 1:solver.model.nlmi
            solver.err2 = solver.err2 + max(0, -eigmin(solver.X[i]) / (1 + norm(solver.model.b)))
            solver.err3 = solver.err3 + norm(solver.Rd[i], 2) / (1 + norm(solver.model.C[i]))
            solver.err4 = solver.err4 + max(0, -eigmin(solver.S[i]) / (1 + norm(solver.model.C[i])))
            # err5 = err5 + (vecC[i]"*vec(X[i])-b'*y)/(1+abs(vecC[i]'*vec(X[i]))+abs(b"*y))
            solver.err6 = solver.err6 + (vec(solver.S[i]))' * vec(solver.X[i]) / (1 + abs(vec(solver.model.C[i])' * vec(solver.X[i])) + abs(dot(solver.model.b', solver.y)))
        end
    end

    solver.err5 = (btrace(solver.model.nlmi, solver.model.C, solver.X) - dot(solver.model.b', solver.y)) / (1 + abs(btrace(solver.model.nlmi, solver.model.C, solver.X)) + abs(dot(solver.model.b', solver.y)))
    if solver.model.nlin > 0
        solver.err2 = solver.err2 + max(0, -minimum(solver.X_lin) / (1 + norm(solver.model.b)))
        solver.err3 = solver.err3 + norm(solver.Rd_lin) / (1 + norm(solver.model.d_lin))
        solver.err4 = solver.err4 + max(0, -minimum(solver.S_lin) / (1 + norm(solver.model.d_lin)))
        solver.err5 = (btrace(solver.model.nlmi, solver.model.C, solver.X) + dot(solver.model.d_lin', solver.X_lin) - dot(solver.model.b',solver.y)) / (1 + abs(btrace(solver.model.nlmi, solver.model.C, solver.X)) + abs(dot(solver.model.b', solver.y)))
        solver.err6 = solver.err6 + dot(solver.S_lin' , solver.X_lin) / (1 + abs(dot(solver.model.d_lin', solver.X_lin)) + abs(dot(solver.model.b', solver.y)))
    end

    if solver.model.nlmi > 0
        DIMACS_error = solver.err1 + solver.err2 + solver.err3 + solver.err4 + abs(solver.err5) + solver.err6
    else
        DIMACS_error = solver.err2 + solver.err3 + solver.err4 + abs(solver.err5) + solver.err6
    end
    if solver.verb > 0 && solver.status == 0 
        #@sprintf("%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %8.0d %9.0d %8.1e %6.0d %8.2f\n', iter, y[1:ddnvar]"*ddc[:], DIMACS_error, err1, err2, err3, err4, err5, err6, cg_iter1, cg_iter2, eq_norm, arank, titi)
        # @printf("%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %8.0d %9.0d %6.0d\n", iter, dot(y, ctmp'), DIMACS_error, err1, err2, err3, err4, err5, err6, cg_iter1, cg_iter2, cg_iter2)
        if solver.verb > 1
            if solver.kit == 0
                @printf("%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %8.2f\n", solver.iter, -dot(solver.y, solver.model.b') + solver.model.b_const, DIMACS_error, solver.err1, solver.err2, solver.err3, solver.err4, solver.err5, solver.err6,solver.itertime)
            else
                @printf("%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %7.0d %7.0d %8.2f\n", solver.iter, -dot(solver.y, solver.model.b') + solver.model.b_const, DIMACS_error, solver.err1, solver.err2, solver.err3, solver.err4, solver.err5, solver.err6, solver.cg_iter_pre, solver.cg_iter_cor,solver.itertime)
            end    
    else
            if solver.kit == 0
                @printf("%3.0d %16.8e %9.2e %8.2f\n", solver.iter, -dot(solver.y, solver.model.b') + solver.model.b_const, DIMACS_error, solver.itertime)
            else
                @printf("%3.0d %16.8e %9.2e %9.0d %8.2f\n", solver.iter, -dot(solver.y, solver.model.b') + solver.model.b_const, DIMACS_error, solver.cg_iter_pre + solver.cg_iter_cor, solver.itertime)
            end
        end
    end

    if DIMACS_error < solver.eDIMACS
        solver.status = 1
        solver.y = solver.y
        if solver.verb > 0
            println("Primal objective: ", -dot(solver.y, solver.model.b') + solver.model.b_const)
            if solver.model.nlin > 0
                println("Dual objective:   ", -btrace(solver.model.nlmi, solver.model.C, solver.X) - dot(solver.model.d_lin', solver.X_lin))
            else
                println("Dual objective:   ", -btrace(solver.model.nlmi, solver.model.C, solver.X) )
            end
            end
    end

    if DIMACS_error > 1e55 
        solver.status = 2
        if solver.verb > 0
            println("WARNING: Problem probably infeasible (stopping status = 2)")
        end
    elseif DIMACS_error > 1e55 || abs(dot(solver.y, solver.model.b')) > 1e55
        solver.status = 3
        if solver.verb > 0
            println("WARNING: Problem probably unbounded or infeasible (stopping status = 3)")
        end
    end

end

```Functions for the iterative solver follow```

struct MyA{T}
    W::Vector{Matrix{T}}
    AA::Vector{SparseArrays.SparseMatrixCSC{Float64,Int}}
    nlin::Int64
    C_lin::SparseArrays.SparseMatrixCSC{Float64,Int64}
    X_lin
    S_lin_inv
    to::TimerOutputs.TimerOutput
end

function (t::MyA)(Ax::Vector{T}, x::Vector{T}) where {T}
    @timeit t.to "Ax" begin
    nlmi = length(t.AA)
    m = size(t.AA[1],1)
    ax1 = zeros(m,1)
    if nlmi > 0
        for ilmi = 1:nlmi
            waxwtmp = Matrix{T}(undef,size(t.W[ilmi]))
            waxw = Matrix{T}(undef,size(t.W[ilmi]))
            # @timeit t.to "Ax1" begin
            ax = Vector{T}(undef,size(t.AA[ilmi],2))
            # end
            # @timeit t.to "Ax2" begin
            mul!(ax, transpose(t.AA[ilmi]), x)
            # ax = transpose(t.AA[ilmi]) * x
            # end
            # @timeit t.to "Ax3" begin
            # waxw .= t.W[ilmi] * mat(ax) * t.W[ilmi]
            mul!(waxwtmp,t.W[ilmi], mat(ax))
            mul!(waxw, waxwtmp, t.W[ilmi])
            # end
            # @timeit t.to "Ax4" begin
            ax1 .+= t.AA[ilmi] * waxw[:]
            # end
        end
    end
    if t.nlin>0
        ax1 .+= t.C_lin * ((t.X_lin .* t.S_lin_inv) .* (t.C_lin' * x))
    end

    mul!(Ax,I(m),ax1[:])
    end
end

struct MyM_no
    to::TimerOutputs.TimerOutput
end

function (t::MyM_no)(Mx::Vector{T}, x::Vector{T}) where {T}
    copy!(Mx,x)
end

function Prec_for_CG_beta(solver,halpha)    
    
    nlmi = solver.model.nlmi
    kk = solver.erank .* ones(Int64,nlmi,1)  
    nvar = solver.model.n
        
    ntot=0
    if nlmi > 0
        for ilmi=1:nlmi
            ntot = ntot + size(solver.W[ilmi],1)
        end
    end
    
    halpha.AAAATtau = zeros(nvar)
    if nlmi > 0
        for ilmi = 1:nlmi
            n = size(solver.W[ilmi],1);
            k = kk[ilmi];
            F = eigen(solver.W[ilmi]); 
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
        if solver.model.nlin > 0
            halpha.AAAATtau .+= diag(solver.model.C_lin * spdiagm((solver.X_lin .* solver.S_lin_inv)[:]) * solver.model.C_lin')
        end
    end
end

struct MyM_beta
    AA
    AAAATtau
end

function (t::MyM_beta)(Mx::Vector{T}, x::Vector{T}) where {T}
    copy!(Mx, x ./ t.AAAATtau)
end

function Prec_for_CG_tilS_prep(solver::MySolver{T},halpha) where {T} 
    
    @timeit solver.to "prec" begin
    nlmi = solver.model.nlmi
    kk = solver.erank .* ones(Int64,nlmi,1)
    # kk[2] = 3
    # halpha.Z = SparseMatrixCSC{T}[]
    halpha.Z = Matrix{T}[]

    nvar = solver.model.n
    
    halpha.AAAATtau = spzeros(nvar,nvar)
    
    ntot=0
    if nlmi > 0
        for ilmi=1:nlmi
            ntot = ntot + size(solver.W[ilmi],1)
        end
    end
    sizeS=0
    if nlmi > 0
        for ilmi=1:nlmi
            sizeS += kk[ilmi] * size(solver.W[ilmi],1)
        end
    end
    
    lbt = 1; lbs=1;
    if nlmi > 0
        for ilmi = 1:nlmi
            n = size(solver.W[ilmi],1);
            k = kk[ilmi];
            
            F = eigen(Float64.(solver.W[ilmi])); 
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
    
    if solver.model.nlin > 0
        halpha.AAAATtau .+= solver.model.C_lin * spdiagm((solver.X_lin .* solver.S_lin_inv)[:]) * solver.model.C_lin'
    end
    
    didi = 0
    for ilmi = 1:nlmi
        didi += size(solver.W[ilmi],1)
    end
    k = kk[1]
    if k > 1 #slow formula
        # @timeit solver.to "prec3" begin
        t = zeros(nvar, k*didi)
        if nlmi > 0
            for ilmi = 1:nlmi
                n = size(solver.W[ilmi],1)
                k = kk[ilmi]
                TT = kron(halpha.Umat[ilmi],halpha.Z[ilmi])
                t[1:nvar,lbt:lbt+k*n-1] .= solver.model.AA[ilmi] * TT
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
        # t = zeros(nvar, k*didi)
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

        S, lbt = prec_alpha_S!(solver,halpha,AAAATtau_d,kk,didi,lbt,sizeS)
    end
    
    # Schur complement for the SMW formula
    S = (S + S') ./ 2 + I(size(S,1))
    halpha.cholS = cholesky(S)

    end
       
end

struct MyM
    AA
    AAAATtau
    Umat
    Z
    cholS
end

function prec_alpha_S!(solver::MySolver{T},halpha,AAAATtau_d,kk,didi,lbt,sizeS) where {T}
    @timeit solver.to "prec3" begin
    S = Matrix{T}(undef,sizeS,sizeS)
    nvar = solver.model.n
    t = Matrix{T}(undef,nvar,kk[1]*didi)
    if solver.model.nlmi > 0
        for ilmi = 1:solver.model.nlmi
            if kk[ilmi] == 0
                continue 
            end
            n = size(solver.W[ilmi],1) 
            k = kk[ilmi] 

            @timeit solver.to "prec30" begin
            AAs = AAAATtau_d * solver.model.AA[ilmi]
            end
            # @timeit solver.to "prec31" begin
            ii_, jj_, aa_ = findnz(AAs)
            qq_ = floor.(Int64,(jj_ .- 1) ./ n) .+ 1
            pp_ = mod.(jj_ .- 1, n) .+ 1
            aau = Vector{T}(undef,length(aa_))
            aau .= aa_ .* halpha.Umat[ilmi][qq_]
            AU = sparse(ii_,pp_,aau,nvar,n)
            # end
            if solver.model.nlmi>1
                # @timeit solver.to "prec32" begin
                didi1 = size(solver.W[ilmi],1)
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
    nlmi = length(t.AA)

    yy2 = zeros(nvar,1)
    y33 = zeros(T,0)

    AAAAinvx = t.AAAATtau\x

    if nlmi > 0
        for ilmi = 1:nlmi
            y22 = t.AA[ilmi]' * AAAAinvx
            y33 = [y33; vec(t.Z[ilmi]' * mat(y22) * t.Umat[ilmi])]
        end
    end
    
    y33 = t.cholS \ y33

    ii = 0
    if nlmi > 0
        for ilmi = 1:nlmi
            n = size(t.Umat[ilmi],1)
            k = size(t.Umat[ilmi],2)
            yy = zeros(n*n)
            for i = 1:k
                xx = t.Z[ilmi] * y33[ii+1:ii+n]
                yy .+= kron(t.Umat[ilmi][:,i],xx)
                ii += n
            end
            yy2 .+= t.AA[ilmi] * yy
        end
    end

    yyy2 = t.AAAATtau \ yy2

    copy!(Mx,(AAAAinvx - yyy2)[:])

end

end #module
