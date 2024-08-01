using TimerOutputs
using FameSVD
using MultiFloats
# using MKL


function _try_cholesky(solver, X, i::Integer, name::String)
    try
        return cholesky(X[i])
    catch
        if solver.verb > 0
            println("Matrix $name not positive definite, trying to regularize")
        end
        icount = 0
        while isposdef(X[i]) == false
            X[i] += 1e-5 .* I(size(X[i], 1))
            icount += 1
            if icount > 1000
                if solver.verb > 0
                    println("WARNING: $name cannot be made positive definite, giving up")
                end
                solver.status = 4
                return I(size(X[i], 1))
            end
        end
        return cholesky(X[i])
    end
end

function prepare_W(solver::MySolver{T}) where {T}

    # @timeit solver.to "prpr" begin
        for i = 1:solver.model.nlmi
            # @timeit to "prpr1" begin
                # Ctmp = _try_cholesky(solver, solver.X, i, "X")
                # CtmpS = _try_cholesky(solver, solver.S, i, "S")
                Ctmp = cholesky(solver.X[i])
                CtmpS = cholesky(solver.S[i])
            @timeit solver.to "prep W SVD" begin
                CCtmp = Matrix{T}(undef,size(CtmpS.L,1),size(CtmpS.L,1))
                mul!(CCtmp, (CtmpS.L)' , Ctmp.L)
                @timeit solver.to "prep W SVD svd" begin
                U, Dtmp, V = svd(CCtmp)
                end
            end

            # @show minimum(Dtmp)
            solver.D[i] = copy(Dtmp)
            Di2 = try
                Diagonal(1.0 ./ sqrt.(Dtmp))
            catch err
                println("WARNING: Numerical difficulties, giving up")
                    solver.status = 4
                Diagonal(I(size(solver.Dtmp, 1)))
            end

            # @timeit to "prpr3a" begin
                solver.G[i] = Ctmp.L * V * Di2
            # end
            # @timeit to "prpr3" begin
                solver.Gi[i] = inv(solver.G[i])
                solver.W[i] =  solver.G[i] * solver.G[i]'
            # end
            # @timeit to "prpr4" begin
                # solver.Si[i] = inv(solver.S[i])
                solver.Si[i] = (CtmpS.L)' \ ((CtmpS.L) \ (I(size(solver.Si[i],1))))  # S[i] inverse
                # DDtmp = (CtmpS.U * solver.G[i])
                # DDtmp = DDtmp' * DDtmp
                DDtmp = solver.G[i]' * solver.S[i] * solver.G[i]
                DDtmp = (DDtmp + DDtmp') ./ 2.0
                # @show minimum(diag(DDtmp,0))
                try
                    solver.DDsi[i] = (1.0 ./ sqrt.(diag(DDtmp,0)))
                catch err
                    println("WARNING: Numerical difficulties, giving up")
                    solver.DDsi[i] = diag(I(size(DDtmp, 1)))
                    solver.status = 4
                    return
                else
                    solver.DDsi[i] = copy(solver.DDsi[i])
                end    
            # end
        end
        if solver.model.nlin > 0
            solver.Si_lin = 1.0 ./ solver.S_lin
        else
            solver.Si_lin = []
        end
        # end

    return solver.D, solver.G, solver.Gi, solver.W, solver.Si, solver.DDsi, solver.Si_lin

end
