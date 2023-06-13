using JuMP
import LinearAlgebra
import Plots
import Random
import SCS
import Loraine
import Test

function svd_cholesky(X::AbstractMatrix)
    F = LinearAlgebra.svd(X)
    # We now have `X ≈ `F.U * D² * F.U'` where:
    D = LinearAlgebra.Diagonal(sqrt.(F.S))
    # So `X ≈ U' * U` where `U` is:
    return (F.U * D)'
end

function solve_max_cut_sdp(weights)
    N = size(weights, 1)
    # Calculate the (weighted) Laplacian of the graph: L = D - W.
    L = LinearAlgebra.diagm(0 => weights * ones(N)) - weights
    model = Model(Loraine.Optimizer)
    # set_silent(model)
    MOI.set(model, MOI.RawOptimizerAttribute("kit"), 0)
    @variable(model, X[1:N, 1:N], PSD)
    for i in 1:N
        set_start_value(X[i, i], 1.0)
    end
    @objective(model, Max, 0.25 * LinearAlgebra.dot(L, X))
    @constraint(model, LinearAlgebra.diag(X) .== 1)
    optimize!(model)
    V = svd_cholesky(value(X))
    Random.seed!(N)
    r = rand(N)
    r /= LinearAlgebra.norm(r)
    cut = [LinearAlgebra.dot(r, V[:, i]) > 0 for i in 1:N]
    S = findall(cut)
    T = findall(.!cut)
    println("Solution:")
    println(" (S, T) = ({", join(S, ", "), "}, {", join(T, ", "), "})")
    return S, T
end

S, T = solve_max_cut_sdp([0 1 5 0; 1 0 0 9; 5 0 0 2; 0 9 2 0])