
using ConjugateGradients
using GenericLinearAlgebra

function predictor(solver::MySolver{T},halpha::Halpha) where {T}
    
    solver.predict = true
    solver.Rp = cons(solver.model, solver.X_lin, solver.X)

    for mat_idx = matrix_indices(solver.model)
        i = mat_idx.value
        solver.Rd[i] .= dual_cons(solver.model, mat_idx, solver.y, solver.S)
        solver.Rc[i] .= solver.sigma .* solver.mu .* Matrix(I, length(solver.D[i]), 1) - solver.D[i] .^ 2
    end

    if num_scalars(solver.model) > 0
        solver.Rd_lin = dual_cons(solver.model, ScalarIndex, solver.y, solver.S_lin)
        Rc_lin = solver.sigma * solver.mu .* ones(num_scalars(solver.model), 1) - solver.X_lin .* solver.S_lin
    end

    w = solver.X_lin .* solver.S_lin_inv
    if solver.kit == 0   # if direct solver; compute the Hessian matrix
        BBBB = schur_complement(solver.model, w, solver.W, solver.G, solver.datarank)
    end
    # end

    # RHS for the Hessian equation
    h = solver.Rp + jprod(
        solver.model,
        isempty(w) ? w : spdiagm(w) * solver.Rd_lin + solver.X_lin,
        [solver.W[i] * (solver.Rd[i] + solver.S[i]) * solver.W[i] for i in 1:num_matrices(solver.model)],
    )

    # solving the linear system()
    if solver.kit == 0   # direct solver
    #     @timeit solver.to "backslash" begin
        if ishermitian(BBBB)
            try
                cholBBBB1, cholBBBB2 = cholesky(BBBB)
                solver.cholBBBB = cholBBBB1
            catch err
                if solver.verb > 0
                    println("Matrix H not positive definite, trying to regularize")
                end
                icount = 0
                solver.regcount += 1
                if solver.regcount > 5
                    if solver.verb > 0
                        println("WARNING: too many regularizations of H, giving up")
                    end
                    solver.cholBBBB = I(size(BBBB, 1))
                    solver.status = 3
                    return
                end
                while isposdef(BBBB) == false
                    BBBB = BBBB + 1e-4 .* I(size(BBBB, 1))
                    icount = icount + 1
                    if icount > 1000
                        if solver.verb > 0
                            println("WARNING: H cannot be made positive definite, giving up")
                        end
                        solver.cholBBBB = I(size(BBBB, 1))
                        solver.status = 3
                        return
                    end
                end
                solver.cholBBBB = cholesky(BBBB)
            else
                solver.cholBBBB = copy(solver.cholBBBB)
            end
            solver.dely = solver.cholBBBB \ h
            solver.dely = solver.cholBBBB' \ (solver.cholBBBB \ h)
            # delyy = solver.dely
        else
            @warn("System matrix not Hermitian, stopping Loraine")
            solver.maxit = 1e10
            solver.status = 2
            solver.cholBBBB = 0
        end
        # # Iterative refinement
        # resid = h - BBBB * solver.dely;
        # # @show norm(resid - (h[solver.cholBBBB.p] - solver.cholBBBB.L * solver.cholBBBB.U * solver.dely))
        # if norm(resid)/(1+norm(h)) > 1e-15
        #     coco = 1
        #     while coco <= 200
        #         deldely = solver.cholBBBB \ resid
        #         w = BBBB * deldely;
        #         alphaIR = resid' * w / (w' * w)
        #         solver.dely = solver.dely + alphaIR * deldely
        #         resid = resid - alphaIR * w
        #         coco = coco + 1
        #         if norm(resid)/(1+norm(h)) < 1e-50
        #             break
        #         end
        #     end
        # end
        # # @show norm(delyy-solver.dely)

    #     end
    else
        A = MyA(w, solver.W, solver.model, solver.to)
        if solver.preconditioner == 0
            M = MyM_no(solver.to)
        elseif solver.preconditioner == 1
            Prec_for_CG_tilS_prep(solver,halpha)  
            M = MyM(solver.model, halpha.AAAATtau, halpha.Umat, halpha.Z, halpha.cholS)
        elseif solver.preconditioner == 2 || solver.preconditioner == 4
            Prec_for_CG_beta(solver,halpha)  
            M = MyM_beta(solver.model, halpha.AAAATtau)
        end

        # @timeit solver.to "CG predictor" begin
        # ConjugateGradients.jl needs `tol` to be `Float64`,
        # maybe we can fix this in that package but in the mean time, we just
        # convert the tolerance to `Float64`
        solver.dely, exit_code, num_iters = cg(A, h[:]; tol = Float64(solver.tol_cg), maxIter = 10000, precon = M)
        # end

        # print(num_iters, exit_code)
        solver.cg_iter_pre += num_iters
        solver.cg_iter_tot += num_iters
    end

    # @timeit solver.to "find step predictor" begin
    find_step(solver)
    # end

end

function sigma_update(solver::MySolver{T}) where {T}
    step_pred = min(minimum([solver.alpha; solver.alpha_lin]), minimum([solver.beta; solver.beta_lin]))
    if (solver.mu .> 1e-6)
        if (step_pred .< 1 / sqrt(3))
                expon_used = 1.0
        else
                expon_used = max(solver.expon, T(3) * step_pred^2)
        end
    else
            expon_used = max(1, min(solver.expon, T(3) * step_pred^2))
    end
    if btrace(num_matrices(solver.model), solver.Xn, solver.Sn) .< 0
        solver.sigma = T(0.8)
    else
        if num_matrices(solver.model) > 0
            tmp1 = btrace(num_matrices(solver.model), solver.Xn, solver.Sn)
        else
            tmp1 = 0
        end
        if num_scalars(solver.model) > 0
                tmp2 = dot(solver.Xn_lin', solver.Sn_lin)
        else
                tmp2 = 0
        end
        tmp12 = (tmp1 + tmp2) / (num_scalars(solver.model) + sum(Base.Fix1(side_dimension, solver.model), matrix_indices(solver.model), init = 0))
        tmp12 = convert(Float64, tmp12)
        mu = Float64(solver.mu)
        solver.sigma = min(1.0, ((tmp12) / mu) ^ Float64(expon_used))
    end

    return solver.sigma
end   

function corrector(solver,halpha)
    solver.predict = false
    if num_scalars(solver.model) > 0
        tmp = (solver.delX_lin .* solver.delS_lin) .* (solver.Si_lin) - (solver.sigma * solver.mu) .* (solver.Si_lin)
        x = spdiagm((solver.X_lin .* solver.Si_lin)[:]) * solver.Rd_lin + solver.X_lin + tmp
    else
        x = Float64[]
    end
    X = map(matrix_indices(solver.model)) do mat_idx
        i = mat_idx.value
        K = my_kron(
            solver.G[i],
            solver.G[i],
            solver.G[i]' * solver.Rd[i] * solver.G[i] + spdiagm(solver.D[i]) - Diagonal((solver.sigma * solver.mu) ./ solver.D[i]) - solver.RNT[i],
        )
        K
    end
    h = solver.Rp + jprod(solver.model, x, X)

    # solving the linear system()
    if solver.kit == 0   # direct solver
    # @timeit to "corrector backsl" begin
        # solver.cholBBBB = cholesky(BBBB)
        # solver.dely = solver.cholBBBB \ h
        solver.dely = solver.cholBBBB' \ (solver.cholBBBB \ h)
        # # Iterative refinement
        # # resid = h - BBBB * solver.dely;
        # resid = h - solver.cholBBBB * solver.cholBBBB' * solver.dely
        # # resid = h[solver.cholBBBB.p] - solver.cholBBBB.L * solver.cholBBBB.U * solver.dely
        # if norm(resid)/(1+norm(h)) > 1e-15
        #     coco = 1
        #     while coco <= 200
        #         deldely = solver.cholBBBB \ resid
        #         # w = BBBB * deldely;
        #         w = solver.cholBBBB.L * solver.cholBBBB.U * deldely
        #         # w = solver.cholBBBB.L * solver.cholBBBB.U * deldely
        #         # w[solver.cholBBBB.p] = w
        #         alphaIR = resid' * w / (w' * w)
        #         solver.dely = solver.dely + alphaIR .* deldely
        #         resid = resid - alphaIR .* w
        #         coco = coco + 1
        #         # @show norm(resid)/(1+norm(h)) 
        #         if norm(resid)/(1+norm(h)) < 1e-50
        #             # @show norm(resid)/(1+norm(h)) 
        #             # @show coco
        #             break
        #         end
        #     end
        # end
    else
        A = MyA(solver.X_lin .* solver.S_lin_inv, solver.W, solver.model, solver.to)
        if solver.preconditioner == 0
            M = MyM_no(solver.to)
        elseif solver.preconditioner == 1
            M = MyM(solver.model, halpha.AAAATtau, halpha.Umat, halpha.Z, halpha.cholS)
        else
            M = MyM_beta(solver.model, halpha.AAAATtau)
        end

        @timeit solver.to "CG corrector" begin
        solver.dely, exit_code, num_iters = cg(A, h[:]; tol = Float64(solver.tol_cg), maxIter = 10000, precon = M)
        end
        solver.cg_iter_cor += num_iters
        solver.cg_iter_tot += num_iters
    end
    # end

    # find delX, delS
    @timeit solver.to "find step corrector" begin
    find_step(solver)
    end
end

function find_step(solver::MySolver{T}) where {T}
    if num_matrices(solver.model) > 0
        for mat_idx in matrix_indices(solver.model)
            i = mat_idx.value
            @timeit solver.to "find_step_A" begin
            solver.delS[i] .= solver.Rd[i] .+ jtprod(solver.model, mat_idx, solver.dely)
            Ξ = vec(my_kron(solver.W[i], solver.W[i], solver.delS[i]))
            if solver.predict
                solver.delX[i] .= mat(-solver.X[i][:] .- Ξ)
            else
                solver.delX[i] .= mat(((solver.sigma * solver.mu) .* solver.Si[i] .- solver.X[i])[:] .- Ξ .+ vec(my_kron(solver.G[i], solver.G[i], solver.RNT[i])))
            end
            end

            # determining steplength to stay feasible
            @timeit solver.to "find_step_B" begin
            delSb = solver.G[i]' * solver.delS[i] * solver.G[i]
            delXb = solver.Gi[i] * solver.delX[i] * solver.Gi[i]'
            end

            @timeit solver.to "find_step_C" begin
            XXX = solver.DDsi[i]' .* delXb .* solver.DDsi[i]
            XXX .= (XXX .+ XXX') ./ 2
            end
            @timeit solver.to "find_step_D" begin
            mimiX = eigmin(T.(XXX))
            end
            if mimiX .> -1e-6
                solver.alpha[i] = 0.99
            else
                solver.alpha[i] = min(1, -solver.tau / mimiX)
            end

            @timeit solver.to "find_step_C" begin
            XXX = solver.DDsi[i]' .* delSb .* solver.DDsi[i]
            XXX .= (XXX .+ XXX') ./ 2
            end
            @timeit solver.to "find_step_D" begin
            mimiS = eigmin(T.(XXX))
            end
            if mimiS .> -1e-6
                solver.beta[i] = 0.99
            else
                solver.beta[i] = min(1, -solver.tau / mimiS)
            end
        end
    end

    if num_scalars(solver.model) > 0
        find_step_lin(solver)
    else
        solver.alpha_lin = 1
        solver.beta_lin = 1
    end    

    if solver.predict
        # solution update
        if num_matrices(solver.model) > 0
            for mat_idx in matrix_indices(solver.model)
                i = mat_idx.value
                solver.Xn[i] = solver.X[i] + solver.alpha[i] .* solver.delX[i]
                solver.Sn[i] = solver.S[i] + solver.beta[i] .* solver.delS[i]
                dim = side_dimension(solver.model, mat_idx)
                deed = solver.D[i] * ones(dim)' + ones(side_dimension(solver.model, mat_idx)) * solver.D[i]'
                solver.RNT[i] = -(solver.Gi[i] * solver.delX[i] * solver.delS[i] * solver.G[i] + solver.G[i]' * solver.delS[i] * solver.delX[i] * solver.Gi[i]') ./ deed 
            end
        end
    else
        solver.yold = solver.y
        solver.y = solver.y + minimum([solver.beta; solver.beta_lin]) * solver.dely
        if num_matrices(solver.model) > 0
            for i = 1:num_matrices(solver.model)
                solver.X[i] = solver.X[i] + minimum([solver.alpha; solver.alpha_lin]) .* solver.delX[i]
                solver.X[i] = (solver.X[i] + solver.X[i]') ./ 2
                solver.S[i] = solver.S[i] + minimum([solver.beta; solver.beta_lin]) .* solver.delS[i]
                solver.S[i] = (solver.S[i] + solver.S[i]') ./ 2
            end       
        end
    end  

    return
end


function find_step_lin(solver)
    solver.delS_lin = solver.Rd_lin + jtprod(solver.model, ScalarIndex, solver.dely)
    if solver.predict
        solver.delX_lin = -solver.X_lin - (solver.X_lin) .* (solver.Si_lin) .* solver.delS_lin
    else
        solver.delX_lin = -solver.X_lin - (solver.X_lin) .* (solver.Si_lin) .* solver.delS_lin + (solver.sigma * solver.mu) .* (solver.Si_lin) + solver.RNT_lin
    end
    mimiX_lin = minimum(solver.delX_lin ./ solver.X_lin)
    if mimiX_lin .> -1e-6
        solver.alpha_lin = 0.99
    else
        solver.alpha_lin = min(1, -solver.tau / mimiX_lin)
    end
    mimiS_lin = minimum(solver.delS_lin ./ solver.S_lin)
    if mimiS_lin .> -1e-6
        solver.beta_lin = 0.99
    else
        solver.beta_lin = min(1, -solver.tau / mimiS_lin)
    end

    if solver.predict
        # solution update
        solver.Xn_lin = solver.X_lin + solver.alpha_lin .* solver.delX_lin
        solver.Sn_lin = solver.S_lin + solver.beta_lin .* solver.delS_lin

        solver.RNT_lin = -(solver.delX_lin .* solver.delS_lin) .* solver.Si_lin
    else
        # @show solver.X_lin
        # @show mimiX_lin
        solver.X_lin = solver.X_lin + minimum([solver.alpha; solver.alpha_lin]) .* solver.delX_lin
        solver.S_lin = solver.S_lin + minimum([solver.beta; solver.beta_lin]) .* solver.delS_lin
        solver.S_lin_inv = 1 ./ solver.S_lin
    end
    
    return 
end
