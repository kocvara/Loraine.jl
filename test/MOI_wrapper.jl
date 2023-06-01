using Test
import MathOptInterface as MOI
import Loraine

function tests()
    optimizer = Loraine.Optimizer()
    MOI.set(optimizer, MOI.Silent(), true) # comment this to enable output
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.instantiate(Loraine.Optimizer, with_bridge_type = Float64),
    )
    MOI.set(model, MOI.Silent(), true) # comment this to enable output
    config = MOI.Test.Config(
        atol = 1e-2,
        exclude = Any[
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
            MOI.ConstraintName,
            MOI.VariableName,
            MOI.ObjectiveBound,
        ],
    )
    MOI.Test.runtests(
        model,
        config,
        include = String[
            "test_conic_PositiveSemidefiniteConeTriangle_VectorAffineFunction",
        ]
    )
    return
end

tests()
