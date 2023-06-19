using Test
import MathOptInterface as MOI
import Loraine

function tests()
    optimizer = Loraine.Optimizer()
    # MOI.set(optimizer, MOI.RawOptimizerAttribute("eDIMACS"), 1e-6) # comment this to enable output
    MOI.set(optimizer, MOI.Silent(), false) # comment this to enable output
    bridged = MOI.instantiate(Loraine.Optimizer, with_bridge_type = Float64)
    # Fix for `Unable to query the dual of a variable bound that was reformulated using `ZerosBridge`.
    MOI.Bridges.remove_bridge(bridged, MOI.Bridges.Variable.ZerosBridge{Float64})
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        bridged,
    )
    # MOI.set(model, MOI.RawOptimizerAttribute("eDIMACS"), 1e-5)
    MOI.set(model, MOI.RawOptimizerAttribute("kit"), 0)
    MOI.set(model, MOI.RawOptimizerAttribute("initpoint"), 1)
    MOI.set(model, MOI.Silent(), true) # comment this to enable output
    config = MOI.Test.Config(
        atol = 1e-2,
        exclude = Any[
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
            MOI.ConstraintName,
            MOI.VariableName,
            MOI.ObjectiveBound,
            MOI.SolverVersion,
        ],
    )
    MOI.Test.runtests(
        model,
        config,
        include = [
            # No constraints
            r"test_solve_TerminationStatus_DUAL_INFEASIBLE$",

            # Unable to bridge RotatedSecondOrderCone to PSD because the dimension is too small: got 2, expected >= 3
            r"test_conic_SecondOrderCone_INFEASIBLE$",
            r"test_constraint_PrimalStart_DualStart_SecondOrderCone$",

            # MathOptInterface.ITERATION_LIMIT == MathOptInterface.DUAL_INFEASIBLE
            r"test_linear_DUAL_INFEASIBLE_2$",
            r"test_conic_SecondOrderCone_no_initial_bound$",
            r"test_conic_SecondOrderCone_negative_post_bound_2$",
            r"test_conic_SecondOrderCone_negative_post_bound_3$",
        ],
    )
    return
end

@testset "MOI tests" begin
    tests()
end
