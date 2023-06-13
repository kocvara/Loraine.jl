using TimerOutputs
using MKL

function prepare_W(solver)

    # @timeit to "prpr" begin
        for i = 1:solver.model.nlmi
            # @timeit to "prpr1" begin
                
                try
                    Ctmp = cholesky(solver.X[i])
                catch
                    println("Matrix X not positive definite")
                else
                    Ctmp = copy(Ctmp)
                end
    
            if isposdef(Ctmp) == false
                icount = 0
                while issuccess(Ctmp) == false
                    solver.X[i] = solver.X[i] + 1e-6 .* eye(size(solver.X[i], 1))
                    Ctmp = cholesky(sparse(solver.X[i]), perm=1:size(solver.X[i],1))
                    icount = icount + 1
                    if icount > 1000
                        error("X cannot be made positive definite")
                        return
                    end
                end
            end
            CtmpS = cholesky(solver.S[i])
            if issuccess(CtmpS) == false
                icount = 0
                while issuccess(CtmpS) == false
                    solver.S[i] = solver.S[i] + 1.0e-6 .* eye(size(solver.S[i], 1))
                    CtmpS = cholesky(sparse(solver.S[i]), perm=1:size(solver.S[i],1))
                    icount = icount + 1
                    if icount > 1000
                        error("S cannot be made positive definite")
                        return
                    end
                end
            end
            # end

            @timeit solver.to "prep W SVD" begin
                CCtmp = Matrix{Float64}(undef,size(CtmpS.L,1),size(CtmpS.L,1))
                mul!(CCtmp, (CtmpS.L)' , Ctmp.L)
                # U, Dtmp, V = try
                U, Dtmp, V = svd!(CCtmp)
                # catch
                    # U, Dtmp, V = try
                        # U, Dtmp, V = LAPACK.gesvd!('S','S',CCtmp); V = V'
                    # catch
                        # ()
                    # end
            #     end
            end

            # print(typeof(Dtmp))
            solver.D[i] = copy(Dtmp)
            Di2 = Diagonal(1 ./ sqrt.(Dtmp))
            # @timeit to "prpr3a" begin
                solver.G[i] = Ctmp.L * V * Di2
            # end
            # @timeit to "prpr3" begin
                solver.Gi[i] = inv(solver.G[i])
                solver.W[i] =  solver.G[i] * solver.G[i]'
            # end
            # @timeit to "prpr4" begin
                solver.Si[i] = inv(solver.S[i])
                DDtmp = solver.G[i]' * solver.S[i] * solver.G[i]
                solver.DDsi[i] = (1 ./ sqrt.(diag(DDtmp,0)))
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