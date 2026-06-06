using TimerOutputs
using FameSVD
using MultiFloats

function try_cholesky(solver, X, i::Integer, name::String)
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

    @timeit solver.to "prepare_W" begin
    Threads.@threads for i = 1:solver.model.nlmi
        Ctmp = try_cholesky(solver, solver.X, i, "X")
        CtmpS = try_cholesky(solver, solver.S, i, "S")
        CCtmp = Matrix{T}(undef, size(CtmpS.L, 1), size(CtmpS.L, 1))
        mul!(CCtmp, (CtmpS.L)', Ctmp.L)
        if T == Float64
            _, Dtmp, V = fsvd(CCtmp)
        else
            _, Dtmp, V = svd(CCtmp)
        end

        solver.D[i] = copy(Dtmp)
        Di2 = try
            Diagonal(1.0 ./ sqrt.(Dtmp))
        catch
            println("WARNING: Numerical difficulties, giving up")
            solver.status = 4
            Diagonal(ones(T, length(Dtmp)))
        end

        solver.G[i] = Ctmp.L * V * Di2
        solver.Gi[i] = inv(solver.G[i])
        solver.W[i] = solver.G[i] * solver.G[i]'
        solver.Si[i] = (CtmpS.L)' \ ((CtmpS.L) \ I(size(solver.Si[i], 1)))
        DDtmp = solver.G[i]' * solver.S[i] * solver.G[i]
        DDtmp = @. (DDtmp + DDtmp') / 2.0
        try
            solver.DDsi[i] = (1.0 ./ sqrt.(diag(DDtmp, 0)))
        catch
            println("WARNING: Numerical difficulties, giving up")
            solver.DDsi[i] = diag(I(size(DDtmp, 1)))
            solver.status = 4
        end
    end
    end

    if solver.model.nlin > 0
        solver.Si_lin = 1.0 ./ solver.S_lin
    else
        solver.Si_lin = []
    end

    return solver.D, solver.G, solver.Gi, solver.W, solver.Si, solver.DDsi, solver.Si_lin

end
