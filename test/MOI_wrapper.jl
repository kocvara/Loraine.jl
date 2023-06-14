using Test
import MathOptInterface as MOI
import Loraine
import Hypatia

function tests()
    optimizer = Loraine.Optimizer()
    # MOI.set(optimizer, MOI.RawOptimizerAttribute("eDIMACS"), 1e-6) # comment this to enable output
    MOI.set(optimizer, MOI.Silent(), false) # comment this to enable output
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.instantiate(Loraine.Optimizer, with_bridge_type = Float64),
    )
    # MOI.set(model, MOI.RawOptimizerAttribute("eDIMACS"), 1e-5)
    # MOI.set(model, MOI.RawOptimizerAttribute("kit"), 0)
    # MOI.set(model, MOI.Silent(), true) # comment this to enable output
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
            r"test_modification_set_singlevariable_lessthan$",
        ]

#         exclude = [
#             # One variable but no constraints
#             r"test_attribute_RawStatusString$",
#             r"test_attribute_SolveTimeSec$",

#             # No constraints
#             r"test_solve_TerminationStatus_DUAL_INFEASIBLE$",
#             r"test_objective_ObjectiveFunction_blank$",

#             # Wrong sign of the optimal objective (value OK)
#             r"test_constraint_ScalarAffineFunction_Interval$",
#             r"test_constraint_ScalarAffineFunction_LessThan$",
#             r"test_constraint_ScalarAffineFunction_duplicate$",
#             r"test_constraint_VectorAffineFunction_duplicate$",
#             r"test_conic_GeometricMeanCone_VectorAffineFunction$",
#             r"test_conic_GeometricMeanCone_VectorAffineFunction_2$",
#             r"test_conic_GeometricMeanCone_VectorAffineFunction_3$",
#             r"test_conic_GeometricMeanCone_VectorOfVariables$",
#             r"test_conic_GeometricMeanCone_VectorOfVariables_2$",
#             r"test_conic_GeometricMeanCone_VectorOfVariables_3$",
#             r"test_conic_NormOneCone_VectorAffineFunction$",
#             r"test_conic_NormOneCone_VectorOfVariables$",
#             r"test_conic_NormInfinityCone_VectorAffineFunction$",
#             r"test_conic_NormInfinityCone_VectorOfVariables$",
#             r"test_conic_RootDetConeSquare$",
#             r"test_conic_RootDetConeSquare_VectorAffineFunction$",
#             r"test_conic_RootDetConeSquare_VectorOfVariables$",
#             r"test_conic_RootDetConeTriangle$",
#             r"test_conic_RootDetConeTriangle_VectorAffineFunction$",
#             r"test_conic_RootDetConeTriangle_VectorOfVariables$",
#             r"test_conic_RotatedSecondOrderCone_INFEASIBLE_2$",
#             r"test_conic_RotatedSecondOrderCone_VectorAffineFunction$",
#             r"test_conic_RotatedSecondOrderCone_VectorOfVariables$",
#             r"test_conic_RotatedSecondOrderCone_out_of_order$",
#             r"test_conic_SecondOrderCone_VectorAffineFunction$",
#             r"test_conic_SecondOrderCone_VectorOfVariables$",
#             r"test_linear_VariablePrimalStart_partial$",
#             r"test_linear_add_constraints$",
#             r"test_linear_integration$",
#             r"test_modification_affine_deletion_edge_cases$",
#             r"test_modification_coef_scalaraffine_lessthan$",
#             r"test_modification_const_vectoraffine_nonpos$",
#             r"test_modification_const_vectoraffine_zeros$",
#             r"test_modification_func_scalaraffine_lessthan$",
#             r"test_modification_multirow_vectoraffine_nonpos$",
#             r"test_modification_set_scalaraffine_lessthan$",
#             r"test_quadratic_constraint_GreaterThan$",
#             r"test_quadratic_constraint_LessThan$",
#             r"test_quadratic_constraint_basic$",
#             r"test_quadratic_constraint_integration$",
#             r"test_quadratic_duplicate_terms$",
#             r"test_variable_solve_with_upperbound$",
#             r"test_linear_integration_Interval$",
#             r"test_linear_integration_modification$",

#             # Infeasible or Unbounded returned, Infeasible only expected
#             r"test_conic_NormInfinityCone_INFEASIBLE$",
#             r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_EqualTo_lower$",
#             r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_EqualTo_upper$",
#             r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_GreaterThan$",
#             r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_Interval_lower$",
#             r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_Interval_upper$",
#             r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_LessThan$",
#             r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan$",
#             r"test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_VariableIndex_LessThan_max$",
#             r"test_conic_NormOneCone_INFEASIBLE$",

#             # Infeasible or Unbounded returned, other expected
#             # r"test_conic_RotatedSecondOrderCone_INFEASIBLE$",
#             r"test_linear_DUAL_INFEASIBLE$",
#             r"test_linear_DUAL_INFEASIBLE_2$",
#             r"test_linear_INFEASIBLE_2$",
#             r"test_conic_RotatedSecondOrderCone_INFEASIBLE$",

#             # Unable to bridge RotatedSecondOrderCone to PSD because the dimension is too small: got 2, expected >= 3
#             r"test_conic_SecondOrderCone_INFEASIBLE$",
#             r"test_constraint_PrimalStart_DualStart_SecondOrderCone$",
#             r"test_constraint_qcp_duplicate_diagonal$",
#             r"test_constraint_qcp_duplicate_off_diagonal$",

#             # Unable to query the dual of a variable bound that was reformulated using `ZerosBridge`. 
#             r"test_conic_linear_VectorOfVariables_2$",

#             # Evaluated: MathOptInterface.FEASIBLE_POINT == MathOptInterface.NO_SOLUTION
#             r"test_solve_result_index$",

#             # Evaluated: ≈(-3.9999991059784197, 8.0, ...
#             r"test_linear_integration_delete_variables$",

#             # Evaluated: ≈(-1.9999996384222327, 3.0, ...
#             r"test_quadratic_nonhomogeneous$",
#             r"test_objective_ObjectiveFunction_constant$",
#             r"test_modification_transform_singlevariable_lessthan$",

    
#             # FIXME to investigate
#             r"test_conic_SecondOrderCone_negative_post_bound_2$",
#             r"test_conic_SecondOrderCone_negative_post_bound_3$",
#             r"test_conic_SecondOrderCone_no_initial_bound$",
#             r"test_linear_INFEASIBLE$",
#             r"test_linear_transform$",
#             r"test_modification_coef_scalar_objective$",
#             r"test_modification_const_scalar_objective$",
#             r"test_modification_set_singlevariable_lessthan$",
#         ],
    )
    return
end

@testset "MOI tests" begin
    tests()
end
