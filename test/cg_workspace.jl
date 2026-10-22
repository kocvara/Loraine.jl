module TestCGWorkspace
using Test, LinearAlgebra, SparseArrays, Random
import Loraine
const LRO = Loraine.Solvers.LRO
function check_application(op, P, x)
    out = similar(x)
    op(out, x)
    @test out ≈ P \ x
    if eltype(x) == Float64
        # Mixed sparse/dense buffers currently box MatrixIndex during dispatch.
        # Bound that fixed overhead while forbidding matrix/vector temporaries.
        allowance = isconcretetype(eltype(op.model.jtprod_buffer)) ? 0 : 128
        @test (@allocated op(out, x)) <= allowance
    end
    for rhs in (-x, zero(x), 2x, x)
        op(out, rhs)
        @test out ≈ P \ rhs
    end
end
function check_workspace(::Type{T}, dims, rank) where {T}
    rng = MersenneTwister(17)
    n = 3
    A = [sparse([mod1(j, d)], [mod1(j, d)], T[j], d, d) for d in dims, j = 1:n]
    model = LRO.Model(
        [spzeros(T, d, d) for d in dims],
        A,
        zeros(T, n),
        SparseVector{T,Int64}(spzeros(T, 0)),
        SparseMatrixCSC{T,Int64}(spzeros(T, n, 0)),
        Int64.(dims),
    )
    model = LRO.BufferedModelForSchur(model, 1)
    U = [randn(rng, T, d, rank) / 10 for d in dims]
    Z = [Matrix{T}(I, d, d) + ones(T, d, d) / d for d in dims]
    workspace = nothing
    for zero_block in (false, true)
        zero_block && fill!(U[2], zero(T))
        D = Diagonal(T[2, 3, 4] .* (zero_block ? 2 : 1))
        V = hcat(
            [Matrix(model.jprod_buffer[i])' * kron(U[i], Z[i]) for i in eachindex(dims)]...,
        )
        P = D + V * V'
        S = cholesky(Symmetric(I + V' * (D \ V)))
        op = Loraine.Solvers.MyM(model, factorize(D), U, Z, S, workspace)
        if !isnothing(workspace)
            @test op.AAAAinvx === workspace.AAAAinvx
            @test op.AyU === workspace.AyU
            @test op.yy === workspace.yy
            @test op.y33 === workspace.y33
        end
        workspace = op
        check_application(op, P, randn(rng, T, n))
    end
end
function check_rank_one_preparation(dims)
    rng = MersenneTwister(23)
    nvar = 3
    A = [sparse([mod1(j, d)], [mod1(j, d)], [Float64(j)], d, d) for d in dims, j = 1:nvar]
    model = LRO.Model(
        [spzeros(d, d) for d in dims],
        A,
        zeros(nvar),
        SparseVector{Float64,Int64}(spzeros(0)),
        SparseMatrixCSC{Float64,Int64}(spzeros(nvar, 0)),
        Int64.(dims),
    )
    solver, halpha = Loraine.Solvers.load(model, Dict("verb" => 0))
    solver.to = Loraine.Solvers.TimerOutput()
    # Match the Int64 Jacobian indices seen in the 32-bit CI failure.
    @test all(J -> eltype(rowvals(J)) == Int64, solver.model.jprod_buffer)
    matrices = [Matrix{Float64}(I, d, d) for d in dims]
    solver.W = LRO.ShapedSolution(
        Float64[],
        [Loraine.Solvers.FactoredMatrix(M, M, M) for M in matrices],
    )
    halpha.Umat = [randn(rng, d, 1) for d in dims]
    halpha.Z = [randn(rng, d, d) for d in dims]
    D = spdiagm([2.0, 3.0, 4.0])
    V = hcat(
        [
            Matrix(solver.model.jprod_buffer[i])' * kron(halpha.Umat[i], halpha.Z[i])
            for i in eachindex(dims)
        ]...,
    )
    S, offset = Loraine.Solvers.prec_alpha_S!(
        solver,
        halpha,
        D,
        ones(Int64, length(dims)),
        sum(dims),
        1,
        sum(dims),
    )
    @test S ≈ V' * Matrix(D)^2 * V
    @test offset == sum(dims) + 1
end
@testset "Rank-one preconditioner preparation" begin
    for dims in ([3], [3, 4, 1])
        check_rank_one_preparation(dims)
    end
end

@testset "CG preconditioner workspace" begin
    for T in (Float32, Float64), dims in ([3, 4, 1], [20, 40, 1]), rank in (1, 2)
        @testset "$T dims=$dims rank=$rank" begin
            check_workspace(T, dims, rank)
        end
    end
end
scaling_allocations(U, Z, ws, W, aamat) =
    @allocated Loraine.Solvers.cg_scaling!(U, Z, ws, W, aamat)

@testset "CG scaling reuses buffers and matches spectral reconstruction" begin
    rng = MersenneTwister(29)
    for T in (Float32, Float64), n in (1, 3, 7), k in unique((1, min(2, n), n))
        Q = Matrix(qr(randn(rng, n, n)).Q)
        U, Z = zeros(T, n, k), zeros(T, n, n)
        ws = LRO.EigenWorkspace(zeros(T, n, n))
        for aamat in (0, 1), shift in (0.0, 0.5)
            # Also cover ill-conditioned scalings seen near convergence.
            spectrum = if n > 1 && shift == 0.5
                exp10.(range(0.0, T == Float32 ? 4.0 : 12.0; length = n))
            else
                (1:n) .+ shift
            end
            W = T.(Q * Diagonal(spectrum) * Q')
            original = copy(W)
            # Reference the previous implementation, which reconstructed W0.
            F = eigen(Float64.(W))
            small = F.values[1:(n-k)]
            tau = if isempty(small)
                minimum(F.values)
            elseif aamat == 0
                minimum(small)
            else
                (minimum(small) + sum(small) / length(small)) / 2 - 1e-14
            end
            Uref = F.vectors[:, (n-k+1):n] * Diagonal(sqrt.(F.values[(n-k+1):n] .- tau))
            W0 = F.vectors * Diagonal(vcat(small, fill(tau, k))) * F.vectors'
            expected = 2W0 + Uref * Uref'
            actual_tau = Loraine.Solvers.cg_scaling!(U, Z, ws, W, aamat)
            @test actual_tau ≈ tau atol = 20eps(T) * norm(W)
            @test U * U' ≈ T.(Uref * Uref') atol = 20eps(T) * norm(W)
            @test Z * Z' ≈ T.(expected)
            @test istril(Z)
            @test all(>(0), diag(Z))
            @test W == original
            @test scaling_allocations(U, Z, ws, W, aamat) == 0
        end
    end
end

@testset "Preconditioner assembly reuses setup storage" begin
    LS = Loraine.Solvers
    for rank in (1, 2), scalars in (false, true)
        dims = [3, 4]
        nvar = 3
        A = [
            sparse([mod1(j, d)], [mod1(j, d)], [Float64(j)], d, d) for d in dims, j = 1:nvar
        ]
        model = LRO.Model(
            [spdiagm(ones(d)) for d in dims],
            A,
            zeros(nvar),
            SparseVector{Float64,Int64}(scalars ? sparsevec([1.0, 2.0]) : spzeros(0)),
            scalars ?
            sparse(Int64[1, 2, 2, 3], Int64[1, 1, 2, 2], [1.0, 2.0, 1.0, 3.0], nvar, 2) :
            SparseMatrixCSC{Float64,Int64}(spzeros(nvar, 0)),
            Int64.(dims),
        )
        state = LS.Solver{Float64}(model; kit = 1, erank = rank, verb = 0)
        solver, halpha = state.solver, state.halpha
        solver.kit = 1
        solver.erank = rank
        solver.to = LS.TimerOutput()
        LS.setup_solver(solver, halpha)
        LS.initial_point(solver)
        for idx in LRO.matrix_indices(solver.model)
            d = dims[idx.value]
            solver.X[idx] .= Diagonal(collect(1.0:d))
            solver.S[idx] .= Diagonal(collect(2.0:(d+1)))
        end
        LS.prepare_W(solver)
        buffers = halpha.cg_assembly
        U, Z = copy(halpha.Umat), copy(halpha.Z)
        base = nothing
        for pass = 1:2
            LS.Prec_for_CG_tilS_prep(solver, halpha)
            if scalars
                @test halpha.AAAATtau isa Matrix{Float64}
                @test halpha.AAAATtau_fact isa Cholesky
                @test !isdiag(halpha.AAAATtau)
                @test Matrix(halpha.AAAATtau_fact) ≈ halpha.AAAATtau
                if pass == 1
                    base = halpha.AAAATtau
                else
                    @test halpha.AAAATtau === base
                end
            else
                @test isnothing(halpha.AAAATtau)
                @test halpha.AAAATtau_fact isa Diagonal
            end
            V = hcat(
                [
                    Matrix(solver.model.jprod_buffer[i])' *
                    kron(halpha.Umat[i], halpha.Z[i]) for i in eachindex(dims)
                ]...,
            )
            factor = if scalars && rank == 1
                Diagonal(diag(halpha.AAAATtau))
            else
                halpha.AAAATtau_fact
            end
            expected = I + V' * (factor \ V)
            @test all(a === b for (a, b) in zip(U, halpha.Umat))
            @test all(a === b for (a, b) in zip(Z, halpha.Z))
            @test Matrix(halpha.cholS) ≈ expected
            @test all(a === b for (a, b) in zip(buffers, halpha.cg_assembly))
            @test halpha.cholS.factors === buffers[2]
        end
    end
end

end
