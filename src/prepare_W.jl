using TimerOutputs
using MultiFloats

# FameSVD forms both singular-vector matrices, but scaling only needs V.
# Keep its Gram-matrix algorithm and LAPACK tolerance, and reuse Gram storage.
function scaling_svd!(gram, A::Matrix{Float64}, workspace = nothing)
    BLAS.syrk!('U', 'T', 1.0, A, 0.0, gram)
    values, vectors = if isnothing(workspace)
        LAPACK.syevr!('V', 'A', 'U', gram, 0.0, 0.0, 0, 0, 0.0)
    else
        LRO.eigen!(workspace, gram; vectors = true, abstol = 0.0)
    end
    reverse!(values)
    @fastmath values .= sqrt.(values)
    reverse!(vectors; dims = 2)
    return values, vectors
end

function try_cholesky!(solver, workspace, X, name::String)
    try
        return cholesky!(copyto!(workspace, X))
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
                # Keep the identity fallback in the same factorization type.
                return cholesky!(copyto!(workspace, I))
            end
        end
        return cholesky!(copyto!(workspace, X))
    end
end

function prepare_W_block!(solver::MySolver{T}, i) where {T}
    mat_idx = LRO.MatrixIndex(i)
    W = solver.W[mat_idx]
    # Previous scaling matrices are dead here; use them for the factors.
    Ctmp = try_cholesky!(solver, W.matrix, solver.X[mat_idx], "X")
    CtmpS = try_cholesky!(solver, W.factor_inv, solver.S[mat_idx], "S")
    CCtmp = solver.kron_tmp[i]
    mul!(CCtmp, (CtmpS.L)', Ctmp.L)
    if T == Float64
        Dtmp, V = scaling_svd!(solver.kron_tmp2[i], CCtmp, solver.eigen_workspaces[i])
    else
        _, Dtmp, V = svd(CCtmp)
    end

    copyto!(solver.D[i], Dtmp)
    try
        @. solver.DDsi[i] = inv(sqrt(Dtmp))
    catch
        @warn("Numerical difficulties, giving up")
        solver.status = 4
        fill!(solver.DDsi[i], one(T))
    end

    mul!(W.factor, Ctmp.L, V)
    rmul!(W.factor, Diagonal(solver.DDsi[i]))

    # Consume the S factor before overwriting W.factor_inv with its result.
    copyto!(solver.Si[i], I)
    ldiv!(CtmpS.L, solver.Si[i])
    ldiv!(CtmpS.L', solver.Si[i])
    # As `W.factor = L * V * Diagonal(DDsi)` where `L` is the Cholesky factor
    # of `X` and `V` is orthogonal, its inverse is `Diagonal(DDsi)⁻¹ * V' / L`.
    adjoint!(W.factor_inv, V)
    rdiv!(W.factor_inv, Ctmp.L)
    W.factor_inv ./= solver.DDsi[i]
    # `L` is stored in `W.matrix` so it is overwritten last.
    LinearAlgebra.mul!(W.matrix, W.factor, W.factor')
    # `W.factor' * S * W.factor` is `Diagonal(D)` as `CCtmp = U * Diagonal(D) * V'`
    # so there is no need to recompute `DDsi` from its diagonal.
    return
end

function prepare_W(solver::MySolver{T}) where {T}

    @timeit solver.to "prepare_W" begin
        solver.W[LRO.ScalarIndex] .= solver.X[LRO.ScalarIndex] .* solver.S_lin_inv
        nblocks = LRO.num_matrices(solver.model)
        if Threads.nthreads() == 1 || nblocks <= 1
            for i = 1:nblocks
                prepare_W_block!(solver, i)
            end
        else
            Threads.@threads for i = 1:nblocks
                prepare_W_block!(solver, i)
            end
        end
    end

    solver.Si_lin .= inv.(solver.S[LRO.ScalarIndex])

    return solver.D, solver.W, solver.Si, solver.DDsi, solver.Si_lin

end
