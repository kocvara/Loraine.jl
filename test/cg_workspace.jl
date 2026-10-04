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
    model = LRO.Model([spzeros(T,d,d) for d in dims], A, zeros(T,n), spzeros(T,0), spzeros(T,n,0), dims)
    model = LRO.BufferedModelForSchur(model, 1)
    U = [randn(rng,T,d,rank) / 10 for d in dims]
    Z = [Matrix{T}(I,d,d) + ones(T,d,d) / d for d in dims]
    D = Diagonal(T[2,3,4])
    for zero_block in (false, true)
        zero_block && fill!(U[2], zero(T))
        V = hcat([model.jprod_buffer[i]' * kron(U[i], Z[i]) for i in eachindex(dims)]...)
        P = D + V * V'
        S = cholesky(Symmetric(I + V' * (D \ V)))
        op = Loraine.Solvers.MyM(model, factorize(D), U, Z, S)
        check_application(op, P, randn(rng,T,n))
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
