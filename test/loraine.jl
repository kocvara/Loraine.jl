using Test
using JuMP
using LinearAlgebra
using SparseArrays
using MultiFloats: Float64x2
import MathOptInterface as MOI
import Loraine

const THETA1 = joinpath(dirname(@__DIR__), "examples", "data", "theta1.dat-s")

@testset "FactoredMatrix multiplication" begin
    for T in (Float32, Float64), Ti in (Int32, Int64)
        M = T[1 2 0; -1 3 4; 0 1 2]
        B = SparseMatrixCSC{T,Ti}(sparse(T[2 0 -1; 0 3 0; 4 0 5]))
        F = Loraine.Solvers.FactoredMatrix(M, Matrix{T}(I, 3, 3), M)
        @test F * B ≈ M * Matrix(B)
        @test B * F ≈ Matrix(B) * M
        @test F * F ≈ M * M
        for (left, right, expected) in (
            (F, B, M * Matrix(B)),
            (B, F, Matrix(B) * M),
            (F, F, M * M),
            (F, M, M * M),
            (M, F, M * M),
        )
            C = zeros(T, 3, 3)
            mul!(C, left, right)
            @test C ≈ expected
            fill!(C, T(2))
            mul!(C, left, right, T(3), T(-2))
            @test C ≈ 3expected .- 4
        end
    end
end

@testset "loraine" begin
    @testset "direct solver" begin
        model = Loraine.loraine(THETA1, Dict("verb" => 0))
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(model, MOI.ObjectiveValue()) ≈ 23 rtol = 1e-6
    end

    # The options are forwarded as `MOI.RawOptimizerAttribute`s
    @testset "iterative solver" begin
        model = Loraine.loraine(THETA1, Dict("verb" => 0, "kit" => 1))
        @test MOI.get(model, MOI.RawOptimizerAttribute("kit")) == 1
        @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(model, MOI.ObjectiveValue()) ≈ 23 rtol = 1e-6
    end

    @testset "default options" begin
        model = Loraine.loraine(THETA1)
        @test MOI.get(model, MOI.ObjectiveValue()) ≈ 23 rtol = 1e-6
    end
end

# A 1x1 block has no residual eigenvalues at the default preconditioner rank.
@testset "CG with a 1x1 PSD block" begin
    for preconditioner in (1, 2), aamat in (0, 1), scalar_first in (false, true)
        @testset "preconditioner=$preconditioner aamat=$aamat scalar_first=$scalar_first" begin
            model = Model(Loraine.Optimizer)
            for (key, value) in ("kit" => 1, "verb" => 0, "preconditioner" => preconditioner, "aamat" => aamat)
                set_attribute(model, key, value)
            end
            @variable(model, y[1:3])
            @variable(model, z)
            X = [y[1] -1 -1; -1 y[2] -1; -1 -1 y[3]]
            Z = reshape([z], 1, 1)
            for block in (scalar_first ? (Z, X) : (X, Z))
                @constraint(model, block in PSDCone())
            end
            @objective(model, Min, sum(y) + z)
            optimize!(model)
            @test termination_status(model) == MOI.OPTIMAL
            @test objective_value(model) ≈ 6 atol = 1e-5
        end
    end
end

@testset "Direct solver reuses its Newton direction" begin
    model = Model(Loraine.Optimizer)
    set_silent(model)
    set_attribute(model, "kit", 0)
    @variable(model, X[1:2, 1:2] in PSDCone())
    @constraint(model, X[1, 1] == 1)
    @constraint(model, X[2, 2] == 1)
    @objective(model, Min, 2 * X[1, 2])
    optimize!(model)
    @test termination_status(model) == MOI.OPTIMAL
    @test objective_value(model) ≈ -2 atol = 1e-6
    backend = unsafe_backend(model)
    solver, halpha = backend.solver.solver, backend.solver.halpha
    Loraine.Solvers.setup_solver(solver, halpha)
    Loraine.Solvers.initial_point(solver)
    direction = solver.dely
    workspace = solver.chol_work
    for _ = 1:3
        Loraine.Solvers.myIPstep(solver, halpha)
        @test solver.dely === direction
        @test all(isfinite, direction)
        @test solver.dely !== solver.y_buffer
        @test solver.chol_work === workspace
        @test solver.cholBBBB.factors === workspace
        @test solver.BBBB ≈ Matrix(solver.cholBBBB)
    end
end

# Zero Jacobian columns produce a singular Schur matrix, exercising recovery.
function singular_schur_solver()
    LS = Loraine.Solvers
    LRO = LS.LRO
    model = LRO.Model(
        [sparse(Matrix{Float64}(I, 2, 2))],
        [spzeros(2, 2) for _ = 1:1, _ = 1:2],
        zeros(2),
        SparseVector{Float64,Int64}(spzeros(0)),
        SparseMatrixCSC{Float64,Int64}(spzeros(2, 0)),
        Int64[2],
    )
    state = LS.Solver{Float64}(model; kit = 0, verb = 0)
    solver, halpha = state.solver, state.halpha
    solver.to = LS.TimerOutput()
    LS.setup_solver(solver, halpha)
    LS.initial_point(solver)
    LS.find_mu(solver)
    LS.prepare_W(solver)
    return solver, halpha
end

@testset "Schur Cholesky recovery" begin
    solver, halpha = singular_schur_solver()
    workspace = solver.chol_work
    Loraine.Solvers.predictor(solver, halpha)
    @test solver.status == 0
    @test solver.regcount == 1
    @test solver.BBBB ≈ 1e-4 * Matrix{Float64}(I, 2, 2)
    @test solver.cholBBBB.factors === workspace
    @test Matrix(solver.cholBBBB) ≈ solver.BBBB

    solver, halpha = singular_schur_solver()
    solver.regcount = 5
    workspace = solver.chol_work
    Loraine.Solvers.predictor(solver, halpha)
    @test solver.status == 3
    @test solver.regcount == 6
    @test iszero(solver.BBBB)
    @test solver.cholBBBB.factors === workspace
    @test Matrix(solver.cholBBBB) ≈ Matrix{Float64}(I, 2, 2)
end

@testset "Newton scratch across multiple PSD blocks" begin
    for kit in (0, 1)
        model = Model(Loraine.Optimizer)
        set_silent(model)
        set_attribute(model, "kit", kit)
        blocks = Vector{VariableRef}[]
        for d in (3, 4, 1)
            y = @variable(model, [1:d], base_name = "y_$d")
            X = AffExpr[r == c ? y[r] : -1.0 for r = 1:d, c = 1:d]
            @constraint(model, X in PSDCone())
            push!(blocks, y)
        end
        @objective(model, Min, sum(sum(y) for y in blocks))
        optimize!(model)
        @test termination_status(model) == MOI.OPTIMAL
        @test objective_value(model) ≈ 18 atol = 1e-5
        for y in blocks
            @test value.(y) ≈ fill(length(y) - 1, length(y)) atol = 1e-5
        end
    end
end

function iteration_storage(solver)
    return (
        solver.yold,
        solver.dely,
        solver.cg_data,
        solver.D[1],
        solver.Si[1],
        solver.DDsi[1],
        solver.Si_lin,
        solver.delX_lin,
        solver.Xn_lin,
        solver.Sn_lin,
        solver.RNT_lin,
        solver.S_lin_inv,
    )
end

@testset "Scaling invariants and persistent iteration buffers" begin
    for (T, kit) in ((Float64, 0), (Float64, 1), (Float64x2, 0))
        model = GenericModel{T}(Loraine.Optimizer{T})
        set_silent(model)
        set_attribute(model, "kit", kit)
        @variable(model, y[1:3])
        @variable(model, z >= 1)
        @constraint(model, [y[1] -1 -1; -1 y[2] -1; -1 -1 y[3]] in PSDCone())
        @objective(model, Min, sum(y) + z)
        if T == Float64
            optimize!(model)
            @test termination_status(model) == MOI.OPTIMAL
            @test objective_value(model) ≈ T(7) atol = 1e-6
        else
            # LowRankOpt's eigmin-based convergence check needs LAPACK types.
            # Exercise higher-precision iterations without that check.
            MOI.Utilities.attach_optimizer(backend(model))
            MOI.Utilities.attach_optimizer(backend(model).optimizer.model)
        end
        state = unsafe_backend(model).solver
        solver, halpha = state.solver, state.halpha
        LS = Loraine.Solvers
        solver.to = LS.TimerOutput()
        LS.setup_solver(solver, halpha)
        LS.initial_point(solver)
        buffers = iteration_storage(solver)
        preconditioner = halpha.cg_preconditioner
        LS.myIPstep(solver, halpha)
        @test all(a === b for (a, b) in zip(buffers, iteration_storage(solver)))
        for _ = 1:2
            LS.prepare_W(solver)
            mat_idx = LS.LRO.MatrixIndex(1)
            W = solver.W[mat_idx]
            tolerance = T == Float64 ? 1e-8 : 1e-18
            @test W.matrix * solver.S[mat_idx] * W.matrix ≈ solver.X[mat_idx] atol =
                tolerance
            @test W.factor_inv * W.factor ≈ Matrix{T}(I, 3, 3) atol = tolerance
            @test solver.Si[1] * solver.S[mat_idx] ≈ Matrix{T}(I, 3, 3) atol = tolerance
            LS.myIPstep(solver, halpha)
            @test all(a === b for (a, b) in zip(buffers, iteration_storage(solver)))
            if kit == 1
                @test halpha.cg_preconditioner.AAAAinvx === preconditioner.AAAAinvx
                @test halpha.cg_preconditioner.yy === preconditioner.yy
            end
        end
    end
end

@testset "Scaling Cholesky workspace and recovery" begin
    solver, _ = singular_schur_solver()
    for diagonal in (1.0, -1e-6)
        X = Matrix(Diagonal([diagonal, 2.0]))
        workspace = similar(X)
        factor = Loraine.Solvers.try_cholesky!(solver, workspace, X, "X")
        @test factor.factors === workspace
        @test Matrix(factor) ≈ X
        @test isposdef(X)
        @test solver.status == 0
    end
end

@testset "CG retries a failed preconditioner" begin
    solver, _ = singular_schur_solver()
    A = Diagonal([2.0, 3.0])
    b = [2.0, 6.0]
    operator = (out, x) -> mul!(out, A, x)
    # A zero preconditioner makes CG divide 0 by 0 instead of finding a step.
    failed_preconditioner = (out, x) -> fill!(out, 0.0)
    direction, exit_code, _ =
        Loraine.Solvers.solve_cg(solver, operator, b, failed_preconditioner)
    @test exit_code > 0
    @test all(isfinite, direction)
    @test A * direction ≈ b
    @test b == [2.0, 6.0]
    @test direction === solver.dely
    indefinite_operator = (out, x) -> mul!(out, -A, x)
    @test_throws ErrorException Loraine.Solvers.solve_cg(
        solver,
        indefinite_operator,
        b,
        Loraine.Solvers.MyM_no(solver.to),
    )
end

@testset "Scaling uses only right singular vectors" begin
    LS = Loraine.Solvers
    for n in (1, 3, 7)
        A = [r == c ? 2.0 + r : sin(r + 2c) / 10 for r = 1:n, c = 1:n]
        original = copy(A)
        gram = similar(A)
        values, V = LS.scaling_svd!(gram, A)
        reference = LS.fsvd(A)
        @test values ≈ reference.S
        @test A' * A * V ≈ V * Diagonal(values .^ 2)
        @test V' * V ≈ Matrix{Float64}(I, n, n)
        @test A == original
    end
end
