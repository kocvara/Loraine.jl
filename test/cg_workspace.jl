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
    A = [sparse([mod1(j,d)], [mod1(j,d)], T[j], d, d) for d in dims, j in 1:n]
    model = LRO.Model(
        [spzeros(T, d, d) for d in dims],
        A,
        zeros(T, n),
        SparseVector{T,Int64}(spzeros(T, 0)),
        SparseMatrixCSC{T,Int64}(spzeros(T, n, 0)),
        Int64.(dims),
    )
    model = LRO.BufferedModelForSchur(model, 1)
    U = [randn(rng,T,d,rank) / 10 for d in dims]
    Z = [Matrix{T}(I,d,d) + ones(T,d,d) / d for d in dims]
    D = Diagonal(T[2,3,4])
    for zero_block in (false, true)
        zero_block && fill!(U[2], zero(T))
        V = hcat(
            [Matrix(model.jprod_buffer[i])' * kron(U[i], Z[i]) for i in eachindex(dims)]...,
        )
        P = D + V * V'
        S = cholesky(Symmetric(I + V' * (D \ V)))
        op = Loraine.Solvers.MyM(model, factorize(D), U, Z, S)
        check_application(op, P, randn(rng,T,n))
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
    for T in (Float32, Float64), dims in ([3,4,1], [20,40,1]), rank in (1,2)
        @testset "$T dims=$dims rank=$rank" begin
            check_workspace(T, dims, rank)
        end
    end
end
end
