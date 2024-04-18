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

function prepare_W(solver)

    # @timeit solver.to "prpr" begin
        for i = 1:solver.model.nlmi
            # @timeit to "prpr1" begin
                # Ctmp = _try_cholesky(solver, solver.X, i, "X")
                # CtmpS = _try_cholesky(solver, solver.S, i, "S")
                X1 = Float64x2.(solver.X[i])
                Ctmp1,Ctmp2 = cholesky(X1)
                CtmpL = Float64.(Ctmp1)
                # Ctmp = cholesky(solver.X[i])
                S1 = Float64x2.(solver.S[i])
                Ctmp1,Ctmp2 = cholesky(S1)
                CtmpSL = Float64.(Ctmp1)
                # CtmpS = cholesky(solver.S[i])
            @timeit solver.to "prep W SVD" begin
                CCtmp = Matrix{Float64}(undef,size(CtmpSL,1),size(CtmpSL,1))
                mul!(CCtmp, (CtmpSL)' , CtmpL)
                @timeit solver.to "prep W SVD svd" begin
                CCtmp1=Float64.(CCtmp)
                U1, Dtmp1, V1 = svd(CCtmp1)
                U = Float64.(U1)
                V = Float64.(V1)
                Dtmp = Float64.(Dtmp1)
                end
            end

            # @show minimum(Dtmp)
            solver.D[i] = copy(Dtmp)
            Di2 = try
                Diagonal(1 ./ sqrt.(Dtmp))
            catch err
                println("WARNING: Numerical difficulties, giving up")
                    solver.status = 4
                Diagonal(I(size(solver.Dtmp, 1)))
            end

            # @timeit to "prpr3a" begin
                solver.G[i] = CtmpL * V * Di2
            # end
            # @timeit to "prpr3" begin
                solver.Gi[i] = inv(solver.G[i])
                solver.W[i] =  solver.G[i] * solver.G[i]'
            # end
            # @timeit to "prpr4" begin
                # solver.Si[i] = inv(solver.S[i])
                solver.Si[i] = (CtmpSL)' \ ((CtmpSL) \ (I(size(solver.Si[i],1))))  # S[i] inverse
                # DDtmp = (CtmpS.U * solver.G[i])
                # DDtmp = DDtmp' * DDtmp
                DDtmp = solver.G[i]' * solver.S[i] * solver.G[i]
                DDtmp = (DDtmp + DDtmp') ./ 2
                # @show minimum(diag(DDtmp,0))
                try
                    solver.DDsi[i] = (1 ./ sqrt.(diag(DDtmp,0)))
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
            solver.Si_lin = 1 ./ solver.S_lin
        else
            solver.Si_lin = []
        end
        # end

    return solver.D, solver.G, solver.Gi, solver.W, solver.Si, solver.DDsi, solver.Si_lin

end
