module Solvers

export  MySolver, solve, load, MyA, Halpha

using SparseArrays
using LinearAlgebra
using Statistics
using Printf
using TimerOutputs
# using MKLSparse
using MKL

include("kron_etc.jl")
include("makeBBBB.jl")
include("model.jl")

mutable struct MySolver
    # main options
    kit::Int64
    tol_cg::Float64
    tol_cg_up::Float64
    tol_cg_min::Float64
    eDIMACS::Float64
    preconditioner::Int64
    erank::Int64
    aamat::Int64
    fig_ev::Int64
    verb::Int64
    timing::Int64
    maxit::Int64

    to::Any

    # model and preprocessed model data
    model::MyModel

    predict::Bool

    # current iterate
    sigma::Float64
    tau::Float64
    mu::Float64
    expon::Float64
    iter::Int64
    DIMACS_error::Float64
    cholBBBB

    status::Int

    err1::Float64
    err2::Float64
    err3::Float64
    err4::Float64
    err5::Float64
    err6::Float64

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

    cg_iter
    cg_iter_tot

    alpha
    beta
    alpha_lin
    beta_lin

    itertime

    RNT
    RNT_lin

    function MySolver(
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
        timing::Int64,
        maxit::Int64, 
        model::MyModel
        )

        solver = new()
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
        solver.timing          = timing   
        solver.maxit           = maxit 
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
    "tol_cg_min" => 1.0e-6,
    "eDIMACS" => 1e-7,
    "preconditioner" => 1,
    "erank" => 1,
    "aamat" => 1,
    "fig_ev" => 0,
    "verb" => 1,
    "timing" => 1,
    "maxit" => 20,
)

function load(model, options::Dict)

    kit = Int64(get(options, "kit", 0))
    tol_cg = get(options, "tol_cg", 1.0e-2)
    tol_cg_up = get(options, "tol_cg_up", 0.5)
    tol_cg_min = get(options, "tol_cg_min", 1.0e-6)
    eDIMACS = get(options, "eDIMACS", 1e-7)
    preconditioner = Int64(get(options, "preconditioner", 1))
    erank = Int64(get(options, "erank", 1))
    aamat = Int64(get(options, "aamat", 1))
    fig_ev = Int64(get(options, "fig_ev", 0))
    verb = Int64(get(options, "verb", 1))
    timing = Int64(get(options, "timing", 1))
    maxit = Int64(get(options, "maxit", 20))

    solver = MySolver(kit,
        tol_cg,
        tol_cg_up,
        tol_cg_min,
        eDIMACS,
        preconditioner,
        erank,
        aamat,
        fig_ev,
        verb,
        timing,
        maxit, 
        MyModel(model.A,
            model.AA,
            model.myA,
            model.C,
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
        @printf("\n *** Loraine.jl v0.1 ***\n")
        @printf(" *** Initialisation STARTS\n")
    end


    if verb > 0
        @printf(" Number of variables: %5d\n",model.n)
        @printf(" LMI constraints    : %5d\n",model.nlmi)
        if model.nlmi>1
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
            else
                @printf(" it        obj         error      err1      err2      err3      err4      err5      err6     cg_pre cg_cor  CPU/it\n")
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

    end

    tottime = time() - t1
    if solver.verb > 0
        if solver.kit == 1
            @printf(" *** Total CG iterations: %8.0d \n", solver.cg_iter_tot)
        end
        @printf(" *** Optimal solution found in %8.2f seconds\n", tottime)
    end

end

function setup_solver(solver::MySolver,halpha::Halpha)

    solver.X = Matrix{Float64}[]
    solver.S = Matrix{Float64}[]
    solver.y = Vector{Float64}[]

    solver.delX = Matrix{Float64}[]
    solver.delS = Matrix{Float64}[]

    solver.D = Vector{Float64}[]
    solver.G = Matrix{Float64}[]
    solver.Gi = Matrix{Float64}[]
    solver.W = Matrix{Float64}[]
    solver.Si = Matrix{Float64}[]
    solver.DDsi = Vector{Float64}[]

    solver.Rd = Matrix{Float64}[]
    solver.Rc = Matrix{Float64}[]

    solver.alpha = zeros(solver.model.nlmi)
    solver.beta = zeros(solver.model.nlmi)

    solver.Xn = Matrix{Float64}[]
    solver.Sn = Matrix{Float64}[]
    solver.RNT = Matrix{Float64}[]
 
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

    halpha.Umat = Matrix{Float64}[]
    halpha.Z = Matrix{Float64}[]
    halpha.AAAATtau = SparseMatrixCSC{Float64}[]

    @show solver.model.A
    @show solver.model.b
    @show solver.model.b_const
    @show solver.model.C
    @show solver.model.d_lin
    @show solver.model.C_lin

    println(size(solver.model.d_lin))

    # model = new()
    # model.A = A
    # model.AA = AA
    # model.myA = myA
    # model.C = C
    # model.b = b
    # model.d_lin = d_lin
    # model.C_lin = C_lin
    # model.n = n
    # model.msizes = msizes
    # model.nlin = nlin
    # model.nlmi = nlmi

    for i = 1:solver.model.nlmi
        push!(halpha.Umat,zeros(solver.model.msizes[i], solver.erank))
        push!(halpha.Z,zeros(solver.model.msizes[i], solver.model.msizes[i]))
        # tmp = Matrix(I(solver.model.msizes[i]))
        # push!(halpha.cholS,cholesky(tmp))
        push!(halpha.AAAATtau,spzeros(solver.model.n, solver.model.n))
    end

    if solver.kit == 1
        if solver.model.nlmi == 0
            println("WARNING: Switching to a direct solver, no LMIs")
            solver.kit = 0
        elseif solver.model.nlmi > 0 && solver.erank >= maximum(solver.model.msizes) - 1
            println("WARNING: Switching to a direct solver, erank bigger than matrix size")
            solver.kit = 0
        end
    end

end


function myIPstep(solver::MySolver,halpha::Halpha)
    solver.iter += 1
    if solver.iter > solver.maxit
        solver.status = 2
    end
    solver.cg_iter = 0
    
    find_mu(solver)

    # @timeit solver.to "prepare W" begin
    prepare_W(solver)
    # end

    ## predictor
    @timeit solver.to "predictor" begin
    predictor(solver::MySolver,halpha::Halpha)
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
    # @show solver.err1

    if solver.model.nlmi > 0
        DIMACS_error = solver.err1 + solver.err2 + solver.err3 + solver.err4 + abs(solver.err5) + solver.err6
    else
        DIMACS_error = solver.err2 + solver.err3 + solver.err4 + abs(solver.err5) + solver.err6
    end
    if solver.verb > 0
        #@sprintf("%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %8.0d %9.0d %8.1e %6.0d %8.2f\n', iter, y[1:ddnvar]"*ddc[:], DIMACS_error, err1, err2, err3, err4, err5, err6, cg_iter1, cg_iter2, eq_norm, arank, titi)
        # @printf("%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %8.0d %9.0d %6.0d\n", iter, dot(y, ctmp'), DIMACS_error, err1, err2, err3, err4, err5, err6, cg_iter1, cg_iter2, cg_iter2)
        if solver.verb > 1
        @printf("%3.0d %16.8e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.0d %8.2f\n", solver.iter, -dot(solver.y, solver.model.b'), DIMACS_error, solver.err1, solver.err2, solver.err3, solver.err4, solver.err5, solver.err6, solver.cg_iter, solver.itertime)
        else
            if solver.kit == 0
                @printf("%3.0d %16.8e %9.2e %8.2f\n", solver.iter, -dot(solver.y, solver.model.b') + solver.model.b_const, DIMACS_error, solver.itertime)
            else
                @printf("%3.0d %16.8e %9.2e %9.0d %8.2f\n", solver.iter, -dot(solver.y, solver.model.b') + solver.model.b_const, DIMACS_error, solver.cg_iter, solver.itertime)
            end
        end
    end

    if DIMACS_error < solver.eDIMACS
        solver.status = 1
        solver.y = solver.y
    end

    if DIMACS_error > 1e50 || abs(dot(solver.y, solver.model.b')) > 1e50
        solver.status = 2
        @warn("Problem probably unbounded or infeasible (stopping status = 2)")
    end

end

```Functions for the iterative solver follow```

struct MyA
    W::Vector{Matrix{Float64}}
    AA::Vector{SparseArrays.SparseMatrixCSC{Float64}}
    nlin::Int64
    C_lin::SparseArrays.SparseMatrixCSC{Float64, Int64}
    X_lin
    S_lin_inv
    to::TimerOutputs.TimerOutput
end

function (t::MyA)(Ax::Vector{Float64}, x::Vector{Float64})
    @timeit t.to "Ax" begin
    nlmi = length(t.AA)
    m = size(t.AA[1],1)
    ax1 = zeros(size(t.AA[1],1))
    if nlmi > 0
        for ilmi = 1:nlmi
            @timeit t.to "Ax1" begin
            # ax = zeros(size(t.AA[ilmi],2))
            end
            @timeit t.to "Ax2" begin
            # mul!(ax, transpose(t.AA[ilmi]), x)
            ax = transpose(t.AA[ilmi]) * x
            end
            @timeit t.to "Ax3" begin
            waxw = t.W[ilmi] * mat(ax) * t.W[ilmi]
            end
            @timeit t.to "Ax4" begin
            ax1 += t.AA[ilmi] * waxw[:]
            end
            # mul!(Ax,t.AA[1],waxw)
        end
    end
    if t.nlin>0
        ax1 = ax1 + t.C_lin * ((t.X_lin .* t.S_lin_inv) .* (t.C_lin' * x))
    end

    mul!(Ax,I(m),ax1[:])
    end
end


function Prec_for_CG_tilS_prep(solver,halpha)    
    
    @timeit solver.to "prec" begin
    nlmi = solver.model.nlmi
    kk = solver.erank .* ones(Int64,nlmi,1)
    # kk[2] = 3
    
    nvar = size(solver.model.AA[1],1)
    
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
    S = zeros(sizeS,sizeS)
    
    lbt = 1; lbs=1;
    if nlmi > 0
        for ilmi = 1:nlmi
            n = size(solver.W[ilmi],1);
            k = kk[ilmi];
            
            F = eigen(solver.W[ilmi]); 
            vectf = F.vectors
            lambdaf = F.values 

            vect_l = vectf[:,n-k+1:n]
            lambda_l = lambdaf[n-k+1:n]
            vect_s = vectf[:,1:n-k]
            lambda_s = lambdaf[1:n-k]

            if solver.aamat==0
                ttau = 1.0*minimum(lambda_s)
            else
                ttau = (minimum(lambda_s) + mean(lambda_s))/2 - 1.0e-14
            end
            
            halpha.Umat[ilmi] = sqrt.(spdiagm(lambda_l) - ttau .* I(k))
            halpha.Umat[ilmi] = vect_l * halpha.Umat[ilmi]
            # m = size(halpha.Umat[ilmi],1);
            
            @timeit solver.to "prec1" begin
            W0 = [vect_s vect_l]*[spdiagm(lambda_s[:]) spzeros(n-k,k); spzeros(k,n-k) ttau * I(k)] * [vect_s vect_l]'
            end

            @timeit solver.to "prec2" begin
            Z = cholesky(2 .* Hermitian(W0) + halpha.Umat[ilmi] * halpha.Umat[ilmi]')
            halpha.Z[ilmi] = Z.L
            end
            
            # switch aamat
                # case 0
                    # ZZZ = solver.model.AAAAT{ilmi};
                # case 1
                    # ZZZ = spdiags(diag(AAAAT{ilmi}),0,nvar,nvar);
                if solver.aamat < 3
                    ZZZ = spdiagm(ones(nvar))
                else
                    ZZZ = spzeros(nvar,nvar)
                end
            # end
            
            halpha.AAAATtau .+= ttau^2 .* ZZZ
        end
    end
    
    if solver.model.nlin > 0
        halpha.AAAATtau .+= solver.model.C_lin * spdiagm((solver.X_lin .* solver.S_lin_inv)[:]) * solver.model.C_lin';
    end
    
    didi = 0
    for ilmi = 1:nlmi
        didi += size(solver.W[ilmi],1)
    end
    k = kk[1]
    if k > 1 #slow formula
        @timeit solver.to "prec3" begin
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
        end
        
        @timeit solver.to "prec4" begin
        S = t' * (halpha.AAAATtau\t) 
        end 
    else #fast formula
        AAAATtau_d = spdiagm(sqrt.(1 ./ diag(halpha.AAAATtau)));

        t = zeros(nvar, k*didi)
        if nlmi > 0
            for ilmi = 1:nlmi
                if kk[ilmi] == 0
                    continue 
                end
                n = size(solver.W[ilmi],1) 
                k = kk[ilmi] 
                AAs = AAAATtau_d * solver.model.AA[ilmi]
                ii_, jj_, aa_ = findnz(AAs)
                qq_ = floor.(Int64,(jj_ .- 1) ./ n) .+ 1
                pp_ = mod.(jj_ .- 1, n) .+ 1
                UU = halpha.Umat[ilmi][qq_]
                aau = aa_ .* UU
                AU = sparse(ii_,pp_,aau,nvar,n)
                if nlmi>1
                    t[1:nvar,lbt:lbt+k*n-1] .= AU * halpha.Z[ilmi]
                else
                    t .= AU * halpha.Z[1]
                end
                lbt = lbt + k*n
            end 
        end
        S .= (t' * t)
    end

    
    # Schur complement for the SMW formula

    S = Hermitian(S) + I(size(S,1))
    
    halpha.cholS = cholesky(S)
    # if flag>0
    #     icount = 0;
    #     while flag>0
    #         S = S + ttau.*eye(sizeS);
    #         [L,flag] = chol(S,'lower');
    #         icount = icount + 1;
    #         if icount>1000
    #             error('Schur complement cannot be made positive definite')
    #             return
    #         end
    #     end
    # end

    end
       
end

struct MyM
    AA
    AAAATtau
    Umat
    Z
    L
end

function (t::MyM)(Mx::Vector{Float64}, x::Vector{Float64})

    nvar = size(x,1)
    nlmi = length(t.AA)

    yy2 = zeros(nvar,1)
    # y3 = []
    y33 = zeros(Float64,0)

    # titi = time()
    AAAAinvx = t.AAAATtau\x

    # mul!(Mx,I(nvar),x)

    if nlmi > 0
        for ilmi = 1:nlmi
            y22 = t.AA[ilmi]' * AAAAinvx
            y33 = [y33; vec(t.Z[ilmi]' * mat(y22) * t.Umat[ilmi])]
        end
    end
    
    y33 = t.L \ y33

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

    # mx1 = AAAAinvx - yyy2
    # mul!(Mx,I(nvar),mx1[:])
    copy!(Mx,(AAAAinvx - yyy2)[:])
    # copy!(Mx,x)

    # print(time()-titi,"\n")

end

end #module
