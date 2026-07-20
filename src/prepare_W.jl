using TimerOutputs
using FameSVD
using MultiFloats

function try_cholesky(solver, X, name::String)
    try
        return cholesky(X)
    catch
        if solver.verb > 0
            println("Matrix $name not positive definite, trying to regularize")
        end
        icount = 0
        while isposdef(X) == false
            X .+= 1e-5 .* I(size(X, 1))
            icount += 1
            if icount > 1000
                if solver.verb > 0
                    @warn("$name cannot be made positive definite, giving up")
                end
                solver.status = 4
                return I(size(X, 1))
            end
        end
        return cholesky(X)
    end
end

function prepare_W(solver::MySolver{T}) where {T}

    # @timeit solver.to "prpr" begin
        solver.W[LRO.ScalarIndex] .= solver.X[LRO.ScalarIndex] .* solver.S_lin_inv
        for mat_idx = LRO.matrix_indices(solver.model)
            i = mat_idx.value
            # @timeit to "prpr1" begin
                Ctmp = try_cholesky(solver, solver.X[mat_idx], "X")
                CtmpS = try_cholesky(solver, solver.S[mat_idx], "S")
                # Ctmp = cholesky(solver.X[mat_idx])
                # CtmpS = cholesky(solver.S[mat_idx])
            @timeit solver.to "prep W SVD" begin
                CCtmp = Matrix{T}(undef,size(CtmpS.L,1),size(CtmpS.L,1))
                mul!(CCtmp, (CtmpS.L)' , Ctmp.L)
                @timeit solver.to "prep W SVD svd" begin
                if T == Float64
                    U, Dtmp, V = fsvd(T.(CCtmp))
                else
                    U, Dtmp, V = svd(T.(CCtmp))
                end
                end
            end

            # @show minimum(Dtmp)
            solver.D[i] = copy(Dtmp)
            Di2 = try
                Diagonal(1.0 ./ sqrt.(Dtmp))
            catch err
                @warn("Numerical difficulties, giving up")
                    solver.status = 4
                Diagonal(I(size(solver.Dtmp, 1)))
            end

            # @timeit to "prpr3a" begin
                W = solver.W[mat_idx]
                W.factor .= Ctmp.L * V * Di2
                W.factor_inv .= inv(W.factor)
                LinearAlgebra.mul!(W.matrix, W.factor, W.factor')
            # end
            # @timeit to "prpr3" begin
            # end
            # @timeit to "prpr4" begin
                # solver.Si[i] = inv(solver.S[i])
                solver.Si[i] = (CtmpS.L)' \ ((CtmpS.L) \ (I(size(solver.Si[i],1))))  # S[i] inverse
                # DDtmp = (CtmpS.U * solver.G[i])
                # DDtmp = DDtmp' * DDtmp
                DDtmp = solver.W[mat_idx].factor' * solver.S[mat_idx] * solver.W[mat_idx].factor
                DDtmp = (DDtmp + DDtmp') ./ 2.0
                try
                    solver.DDsi[i] = (1.0 ./ sqrt.(diag(DDtmp,0)))
                catch err
                    @warn("Numerical difficulties, giving up")
                    solver.DDsi[i] = diag(I(size(DDtmp, 1)))
                    solver.status = 4
                    return
                else
                    solver.DDsi[i] = copy(solver.DDsi[i])
                end
            # end
        end
        if LRO.num_scalars(solver.model) > 0
            solver.Si_lin = inv.(solver.S[LRO.ScalarIndex])
        else
            solver.Si_lin = []
        end
        # end

    return solver.D, solver.W, solver.Si, solver.DDsi, solver.Si_lin

end
