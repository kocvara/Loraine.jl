using JuMP
import LinearAlgebra
import Random
# import SCS
import Loraine
import Test

function example_minimum_distortion()
    model = Model(Loraine.Optimizer)
    # set_silent(model)
    MOI.set(model, MOI.RawOptimizerAttribute("kit"), 0)
    D = [
        0.0 1.0 1.0 1.0
        1.0 0.0 2.0 2.0
        1.0 2.0 0.0 2.0
        1.0 2.0 2.0 0.0
    ]
    @variable(model, c² >= 1.0)
    @variable(model, Q[1:4, 1:4], PSD)
    for i in 1:4, j in (i+1):4
        @constraint(model, D[i, j]^2 <= Q[i, i] + Q[j, j] - 2 * Q[i, j])
        @constraint(model, Q[i, i] + Q[j, j] - 2 * Q[i, j] <= c² * D[i, j]^2)
    end
    fix(Q[1, 1], 0)
    @objective(model, Min, c²)
    optimize!(model)
    Test.@test termination_status(model) == OPTIMAL
    Test.@test primal_status(model) == FEASIBLE_POINT
    Test.@test objective_value(model) ≈ 4 / 3 atol = 1e-4
    # Recover the minimal distorted embedding:
    return value(Q)
end

Q = example_minimum_distortion()
@test Q ≈ [
    0  0  0  0
    0  4 -2 -2
    0 -2  4 -2
    0 -2 -2  4
] / 3 rtol = 1e-6

function plot_distortion(Q)
    X = [zeros(3) sqrt(Q[2:end, 2:end])]
    return Plots.plot(
        X[1, :],
        X[2, :],
        X[3, :];
        seriestype = :mesh3d,
        connections = ([0, 0, 0, 1], [1, 2, 3, 2], [2, 3, 1, 3]),
        legend = false,
        fillalpha = 0.1,
        lw = 3,
        ratio = :equal,
        xlim = (-1.1, 1.1),
        ylim = (-1.1, 1.1),
        zlim = (-1.5, 1.0),
        zticks = -1:1,
        camera = (60, 30),
    )
end

# import Plots
# plot_distortion(Q)
