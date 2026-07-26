
using ConjugateGradients
using GenericLinearAlgebra

function predictor(solver::MySolver{T},halpha::Halpha) where {T}

    solver.predict = true
    copyto!(solver.Rp, solver.model.b)

    if solver.model.nlmi > 0
        for i = 1:solver.model.nlmi
            mul!(solver.Rp, solver.model.AA[i], vec(solver.X[i]), -1.0, 1.0)
        end
        Threads.@threads for i = 1:solver.model.nlmi
            mul!(solver.work_m2[i], solver.model.AA[i]', solver.y)
            mat!(solver.work_mm[i], solver.work_m2[i])
            solver.Rd[i] .= solver.model.C[i] .- solver.S[i] .- solver.work_mm[i]
        end
    end

    if solver.model.nlin > 0
        mul!(solver.Rp, solver.model.C_lin, solver.X_lin, -1.0, 1.0)
        solver.Rd_lin = solver.model.d_lin - solver.S_lin - solver.model.C_lin' * solver.y
    end

    if solver.kit == 0   # if direct solver; compute the Hessian matrix
        if solver.model.nlmi > 0
            if solver.datarank == -1
                BBBB = makeBBBB_rank1(solver.model.n, solver.model.nlmi, solver.model.B, solver.G, solver.to)
            else
                BBBB = makeBBBBs(solver.model.n, solver.model.nlmi, solver.model.A, solver.model.AA, solver.W, solver.to, solver.model.qA, solver.model.sigmaA)
            end
        else
            BBBB = zeros(T, solver.model.n, solver.model.n)
        end
        if solver.model.nlin > 0
            BBBB .+= solver.model.C_lin * Diagonal(solver.X_lin .* solver.Si_lin) * solver.model.C_lin'
        end
        BBBB = Hermitian(BBBB, :L)
    end
    # end

    if solver.model.nlmi > 0
        h = makeRHS(solver.model.nlmi,solver.model.AA,solver.W,solver.S,solver.Rp,solver.Rd)
    else
        h = copy(solver.Rp)
    end
    if solver.model.nlin > 0
        h .+= solver.model.C_lin * (Diagonal(solver.X_lin .* solver.Si_lin) * solver.Rd_lin + solver.X_lin)
    end

    # solving the linear system()
    if solver.kit == 0   # direct solver
    #     @timeit solver.to "backslash" begin
        if ishermitian(BBBB)
            try
                cholBBBB1, _ = cholesky(BBBB)
                solver.cholBBBB = cholBBBB1
            catch
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
            solver.dely = solver.cholBBBB' \ (solver.cholBBBB \ h)
            # delyy = solver.dely
        else
            @warn("System matrix not Hermitian, stopping Loraine")
            solver.maxit = 1e10
            solver.status = 2
            solver.cholBBBB = 0
        end
    #     end
    else
        A = MyA(solver.W,solver.model.AA,solver.model.nlin,solver.model.C_lin,solver.X_lin,solver.S_lin_inv,solver.to)
        if solver.preconditioner == 0
            M = MyM_no(solver.to)
        elseif solver.preconditioner == 1
            Prec_for_CG_tilS_prep(solver,halpha)
            M = MyM(solver.model.AA, halpha.AAAATtau_fact, halpha.Umat, halpha.Z, halpha.cholS)
        elseif solver.preconditioner == 2 || solver.preconditioner == 4
            Prec_for_CG_beta(solver,halpha)
            M = MyM_beta(solver.model.AA, halpha.AAAATtau)
        end

        # @timeit solver.to "CG predictor" begin
        # ConjugateGradients.jl needs `tol` to be `Float64`,
        # maybe we can fix this in that package but in the mean time, we just
        # convert the tolerance to `Float64`
        solver.dely, exit_code, num_iters = cg(A, h[:]; tol = Float64(solver.tol_cg), maxIter = Int64(10000), precon = M)
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
    step_pred = min(
        minimum(solver.alpha; init = zero(T)),
        solver.alpha_lin,
        minimum(solver.beta; init = zero(T)),
        solver.beta_lin,
    )
    if (solver.mu > 1e-6)
        if (step_pred < 1 / sqrt(3))
                expon_used = 1.0
        else
                expon_used = max(solver.expon, T(3) * step_pred^2)
        end
    else
            expon_used = max(1, min(solver.expon, T(3) * step_pred^2))
    end
    trXnSn = btrace(solver.model.nlmi, solver.Xn, solver.Sn)
    if trXnSn < 0
        solver.sigma = T(0.8)
    else
        if solver.model.nlmi > 0
            tmp1 = trXnSn
        else
            tmp1 = 0
        end
        if solver.model.nlin > 0
                tmp2 = dot(solver.Xn_lin', solver.Sn_lin)
        else
                tmp2 = 0
        end
        tmp12 = (tmp1 + tmp2) / (sum(solver.model.msizes) + solver.model.nlin)
        tmp12 = convert(Float64, tmp12)
        mu = Float64(solver.mu)
        solver.sigma = min(1.0, ((tmp12) / mu) ^ Float64(expon_used))
    end

    return solver.sigma
end

function corrector(solver,halpha)
    solver.predict = false
    h = copy(solver.Rp)
    if solver.model.nlmi > 0
        for i = 1:solver.model.nlmi
            mul!(h, solver.model.AA[i], my_kron(solver.G[i], solver.G[i], (solver.G[i]' * solver.Rd[i] * solver.G[i] + Diagonal(@. solver.D[i] - (solver.sigma * solver.mu) / solver.D[i]) - solver.RNT[i])), true, true)
        end
    end
    if solver.model.nlin > 0
        tmp = (solver.delX_lin .* solver.delS_lin) .* (solver.Si_lin) - (solver.sigma * solver.mu) .* (solver.Si_lin)
        h .+= solver.model.C_lin * (Diagonal(solver.X_lin .* solver.Si_lin) * solver.Rd_lin + solver.X_lin + tmp)
    end

    # solving the linear system()
    if solver.kit == 0   # direct solver
    # @timeit to "corrector backsl" begin
        solver.dely = solver.cholBBBB' \ (solver.cholBBBB \ h)
    else
        A = MyA(solver.W,solver.model.AA,solver.model.nlin,solver.model.C_lin,solver.X_lin,solver.S_lin_inv,solver.to)
        if solver.preconditioner == 0
            M = MyM_no(solver.to)
        elseif solver.preconditioner == 1
            M = MyM(solver.model.AA, halpha.AAAATtau_fact, halpha.Umat, halpha.Z, halpha.cholS)
        else
            M = MyM_beta(solver.model.AA, halpha.AAAATtau)
        end

        @timeit solver.to "CG corrector" begin
            solver.dely, exit_code, num_iters = cg(A, h[:]; tol = Float64(solver.tol_cg), maxIter = Int64(10000), precon = M)
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
    if solver.model.nlmi > 0
        @timeit solver.to "find_step" begin
        for i = 1:solver.model.nlmi
            mul!(solver.work_m2[i], solver.model.AA[i]', solver.dely)
            mat!(solver.work_mm[i], solver.work_m2[i])
            solver.delS[i] .= solver.Rd[i] .- solver.work_mm[i]
            Ξ = my_kron(solver.W[i], solver.W[i], solver.delS[i])
            if solver.predict
                solver.delX[i] .= mat(-vec(solver.X[i]) .- Ξ)
            else
                solver.delX[i] .= mat(((solver.sigma * solver.mu) .* solver.Si[i] .- solver.X[i])[:] .- Ξ .+ my_kron(solver.G[i], solver.G[i], solver.RNT[i]))
            end

            delSb = solver.G[i]' * solver.delS[i] * solver.G[i]
            delXb = solver.Gi[i] * solver.delX[i] * solver.Gi[i]'

            XXX = @. solver.DDsi[i]' * delXb * solver.DDsi[i]
            XXX .= (XXX .+ XXX') ./ 2
            mimiX = minimum(eigvals(Symmetric(XXX)))
            solver.alpha[i] = mimiX > -1e-6 ? T(0.99) : min(T(1), -solver.tau / mimiX)

            @. XXX = solver.DDsi[i]' * delSb * solver.DDsi[i]
            XXX .= (XXX .+ XXX') ./ 2
            mimiS = minimum(eigvals(Symmetric(XXX)))
            solver.beta[i] = mimiS > -1e-6 ? T(0.99) : min(T(1), -solver.tau / mimiS)
        end
        end
    end

    if solver.model.nlin > 0
        find_step_lin(solver)
    else
        solver.alpha_lin = 1
        solver.beta_lin = 1
    end

    if solver.predict
        if solver.model.nlmi > 0
            Threads.@threads for i = 1:solver.model.nlmi
                @. solver.Xn[i] = solver.X[i] + solver.alpha[i] * solver.delX[i]
                @. solver.Sn[i] = solver.S[i] + solver.beta[i] * solver.delS[i]
                deed = solver.D[i] .+ solver.D[i]'
                solver.RNT[i] .= -(solver.Gi[i] * solver.delX[i] * solver.delS[i] * solver.G[i] + solver.G[i]' * solver.delS[i] * solver.delX[i] * solver.Gi[i]') ./ deed
            end
        end
    else
        solver.yold = copy(solver.y)
        beta_step = min(minimum(solver.beta), solver.beta_lin)
        LinearAlgebra.axpy!(beta_step, solver.dely, solver.y)
        if solver.model.nlmi > 0
            alpha_step = min(minimum(solver.alpha), solver.alpha_lin)
            Threads.@threads for i = 1:solver.model.nlmi
                @. solver.X[i] += alpha_step * solver.delX[i]
                solver.X[i] .= (solver.X[i] .+ solver.X[i]') ./ 2
                @. solver.S[i] += beta_step * solver.delS[i]
                solver.S[i] .= (solver.S[i] .+ solver.S[i]') ./ 2
            end
        end
    end

    return
end


function find_step_lin(solver)
    solver.delS_lin = solver.Rd_lin - solver.model.C_lin' * solver.dely
    if solver.predict
        solver.delX_lin = -solver.X_lin - (solver.X_lin) .* (solver.Si_lin) .* solver.delS_lin
    else
        solver.delX_lin = -solver.X_lin - (solver.X_lin) .* (solver.Si_lin) .* solver.delS_lin + (solver.sigma * solver.mu) .* (solver.Si_lin) + solver.RNT_lin
    end
    mimiX_lin = minimum(solver.delX_lin ./ solver.X_lin)
    if mimiX_lin > -1e-6
        solver.alpha_lin = 0.99
    else
        solver.alpha_lin = min(1, -solver.tau / mimiX_lin)
    end
    mimiS_lin = minimum(solver.delS_lin ./ solver.S_lin)
    if mimiS_lin > -1e-6
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
        solver.X_lin = solver.X_lin + min(minimum(solver.alpha), solver.alpha_lin) .* solver.delX_lin
        solver.S_lin = solver.S_lin + min(minimum(solver.beta), solver.beta_lin) .* solver.delS_lin
        solver.S_lin_inv = 1 ./ solver.S_lin
    end

    return
end
