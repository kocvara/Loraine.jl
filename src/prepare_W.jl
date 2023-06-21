using TimerOutputs
# using MKL

function prepare_W(solver)

    # @timeit to "prpr" begin
        for i = 1:solver.model.nlmi
            # @timeit to "prpr1" begin
                try
                    Ctmp = cholesky(solver.X[i])
                catch
                    println("Matrix X not positive definite, trying to regularize")
                    icount = 0
                    while isposdef(solver.X[i]) == false
                        solver.X[i] = solver.X[i] + 1e-5 .* I(size(solver.X[i], 1))
                        icount = icount + 1
                        # @show icount
                        if icount > 1000
                            println("WARNING: X cannot be made positive definite, giving up")
                            Ctmp = I(size(solver.X[i], 1))
                            solver.status = 4
                            return
                        end
                    end
                    Ctmp = cholesky(solver.X[i])
                else
                    Ctmp = copy(Ctmp)
                end

                try
                    CtmpS = cholesky(solver.S[i])
                catch
                    println("Matrix S not positive definite, trying to regularize")
                    icount = 0
                    while isposdef(solver.S[i]) == false
                        solver.S[i] = solver.S[i] + 1e-5 .* I(size(solver.S[i], 1))
                        icount = icount + 1
                        # @show icount
                        if icount > 1000
                            println("WARNING: S cannot be made positive definite, giving up")
                            CtmpS = I(size(solver.S[i], 1))
                            solver.status = 4
                            return
                        end
                    end
                    CtmpS = cholesky(solver.S[i])
                else
                    CtmpS = copy(CtmpS)
                end

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
            try
                Di2 = Diagonal(1 ./ sqrt.(Dtmp))
            catch err
                println("WARNING: Numerical difficulties, giving up")
                Di2 = Diagonal(I(size(solver.Dtmp, 1)))
                solver.status = 4
                return
            else 
                Di2 = copy(Di2)
            end

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