using Test
import MathOptInterface as MOI
import Loraine

function tests()
    optimizer = Loraine.Optimizer()
    # MOI.set(optimizer, MOI.RawOptimizerAttribute("eDIMACS"), 1e-6) # comment this to enable output
    MOI.set(optimizer, MOI.Silent(), false) # comment this to enable output
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.instantiate(Loraine.Optimizer, with_bridge_type = Float64),
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
        # include = [
        #     r"test_solve_result_index$",
        #         ]        

        exclude = [
            # One variable but no constraints
            r"test_attribute_RawStatusString$",
            r"test_attribute_SolveTimeSec$",

            # No constraints
            r"test_solve_TerminationStatus_DUAL_INFEASIBLE$",
            r"test_objective_ObjectiveFunction_blank$",

            # Unable to query the dual of a variable bound
            r"test_conic_RotatedSecondOrderCone_INFEASIBLE_2$",
            r"test_conic_RotatedSecondOrderCone_VectorOfVariables$",
            r"test_linear_integration$",
            r"test_quadratic_constraint_GreaterThan$",
            r"test_quadratic_constraint_LessThan$",
            r"test_conic_linear_VectorOfVariables_2$",

            # Infeasible or Unbounded returned, Infeasible only expected
            r"test_conic_NormInfinityCone_INFEASIBLE$",
            r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_EqualTo_lower$",
            r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_EqualTo_upper$",
            r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_GreaterThan$",
            r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_Interval_lower$",
            r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_Interval_upper$",
            r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_LessThan$",
            r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan$",
            r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan_max$",
            r"test_conic_NormOneCone_INFEASIBLE$",

            # Infeasible or Unbounded returned, other expected
            r"test_conic_RotatedSecondOrderCone_INFEASIBLE$",
            r"test_linear_DUAL_INFEASIBLE$",
            r"test_linear_DUAL_INFEASIBLE_2$",
            r"test_linear_INFEASIBLE_2$",
            r"test_conic_RotatedSecondOrderCone_INFEASIBLE$",

            # Unable to bridge RotatedSecondOrderCone to PSD because the dimension is too small: got 2, expected >= 3
            r"test_conic_SecondOrderCone_INFEASIBLE$",
            r"test_constraint_PrimalStart_DualStart_SecondOrderCone$",

            # Evaluated: MathOptInterface.FEASIBLE_POINT == MathOptInterface.NO_SOLUTION
            r"test_solve_result_index$",
    
            # FIXME to investigate
            r"test_conic_SecondOrderCone_negative_post_bound_2$",
            r"test_conic_SecondOrderCone_negative_post_bound_3$",
            r"test_conic_SecondOrderCone_no_initial_bound$",
            r"test_linear_INFEASIBLE$",
            r"test_linear_transform$",

            r"test_infeasible_MAX_SENSE$",               
            r"test_infeasible_MAX_SENSE_offset$",
            r"test_infeasible_MIN_SENSE$",        
            r"test_infeasible_MIN_SENSE_offset$",        
            r"test_infeasible_affine_MAX_SENSE$",        
            r"test_infeasible_affine_MAX_SENSE_offset$", 
            r"test_infeasible_affine_MIN_SENSE$",        
            r"test_infeasible_affine_MIN_SENSE_offset$", 
            r"test_linear_FEASIBILITY_SENSE$",   
            r"test_linear_integration_Interval$",   
        ],
    )
    return
end

@testset "MOI tests" begin
    tests()
end
