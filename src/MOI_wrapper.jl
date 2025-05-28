const VAF{T} = MOI.VectorAffineFunction{T}
const PSD = MOI.PositiveSemidefiniteConeTriangle
const NNG = MOI.Nonnegatives

MOI.Utilities.@product_of_sets(NNGCones, NNG)

MOI.Utilities.@product_of_sets(PSDCones, PSD)

MOI.Utilities.@struct_of_constraints_by_set_types(PSDOrNot, PSD, NNG)

const OptimizerCache{T} = MOI.Utilities.GenericModel{
    T,
    MOI.Utilities.ObjectiveContainer{T},
    MOI.Utilities.VariablesContainer{T},
    PSDOrNot{T}{
        MOI.Utilities.MatrixOfConstraints{
            T,
            MOI.Utilities.MutableSparseMatrixCSC{
                T,
                Int64,
                MOI.Utilities.OneBasedIndexing,
            },
            Vector{T},
            PSDCones{T},
        },
        MOI.Utilities.MatrixOfConstraints{
            T,
            MOI.Utilities.MutableSparseMatrixCSC{
                T,
                Int64,
                MOI.Utilities.OneBasedIndexing,
            },
            Vector{T},
            NNGCones{T},
        },
    },
}

mutable struct Optimizer{T} <: MOI.AbstractOptimizer
    solver::Union{Nothing,MySolver{T}}
    halpha::Union{Nothing,Halpha}
    lmi_id::Dict{MOI.ConstraintIndex{VAF{T},PSD},Int64}
    lin_cones::Union{Nothing,NNGCones{T}}
    max_sense::Bool
    objective_constant::T
    silent::Bool
    options::Dict{String,Any}

    function Optimizer{T}() where {T}
        return new{T}(
            nothing,
            nothing,
            Dict{MOI.ConstraintIndex{VAF{T},PSD},Int64}(),
            nothing,
            false,
            0.0,
            false,
            copy(Solvers.DEFAULT_OPTIONS),
        )
    end
end

Optimizer() = Optimizer{Float64}()

function MOI.default_cache(::Optimizer, ::Type{T}) where {T}
    return MOI.Utilities.UniversalFallback(OptimizerCache{T}())
end

function MOI.is_empty(optimizer::Optimizer)
    return isnothing(optimizer.solver)
end

function MOI.empty!(optimizer::Optimizer)
    optimizer.solver = nothing
    optimizer.lin_cones = nothing
    return
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Loraine"

# MOI.RawOptimizerAttribute

function MOI.supports(::Optimizer, param::MOI.RawOptimizerAttribute)
    return haskey(Solvers.DEFAULT_OPTIONS, param.name)
end

function MOI.set(optimizer::Optimizer, param::MOI.RawOptimizerAttribute, value)
    if !MOI.supports(optimizer, param)
        throw(MOI.UnsupportedAttribute(param))
    end
    optimizer.options[param.name] = value
    if !isnothing(optimizer.solver)
        setproperty!(optimizer.solver, Symbol(param.name), value)
    end
    return
end

function MOI.get(optimizer::Optimizer, param::MOI.RawOptimizerAttribute)
    if !MOI.supports(optimizer, param)
        throw(MOI.UnsupportedAttribute(param))
    end
    return optimizer.options[param.name]
end

# MOI.Silent

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(optimizer::Optimizer, ::MOI.Silent, value::Bool)
    optimizer.silent = value
    return
end

MOI.get(optimizer::Optimizer, ::MOI.Silent) = optimizer.silent

function MOI.set(optimizer::Optimizer, ::MOI.ObjectiveSense, value::Bool)
    optimizer.max_sense = value
    return
end

# MOI.supports

function MOI.supports(
    ::Optimizer,
    ::Union{MOI.ObjectiveSense,MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}},
) where {T}
    return true
end

const SUPPORTED_CONES = Union{NNG,PSD}

function MOI.supports_constraint(::Optimizer{T}, ::Type{VAF{T}}, ::Type{<:SUPPORTED_CONES}) where {T}
    return true
end

function MOI.optimize!(optimizer::Optimizer)
    optimizer.solver.to = TimerOutput()
    solve(optimizer.solver, optimizer.halpha)
    return
end

function MOI.copy_to(dest::Optimizer{T}, src::OptimizerCache{T}) where {T}
    MOI.empty!(dest)
    psd_AC = MOI.Utilities.constraints(src.constraints, VAF{T}, PSD)
    Cd_lin = MOI.Utilities.constraints(src.constraints, VAF{T}, NNG)
    SM = SparseMatrixCSC{T,Int64}
    psd_A = convert(SM, psd_AC.coefficients)
    C_lin = convert(SM, Cd_lin.coefficients)
    C_lin = -convert(SM, C_lin')
    n = MOI.get(src, MOI.NumberOfVariables())
    nlmi = MOI.get(src, MOI.NumberOfConstraints{VAF{T},PSD}())
    A = Matrix{Tuple{Vector{Int64},Vector{Int64},Vector{T},Int64,Int64}}(undef, nlmi, n + 1)
    back = Vector{Tuple{Int64,Int64,Int64}}(undef, size(psd_A, 1))
    empty!(dest.lmi_id)
    row = 0
    msizes = Int64[]
    for (lmi_id, ci) in enumerate(MOI.get(src, MOI.ListOfConstraintIndices{VAF{T},PSD}()))
        dest.lmi_id[ci] = lmi_id
        set = MOI.get(src, MOI.ConstraintSet(), ci)
        d = set.side_dimension
        push!(msizes, d)
        for k = 1:(n+1)
            A[lmi_id, k] = (Int64[], Int64[], T[], d, d)
        end
        for j = 1:d
            for i = 1:j
                row += 1
                back[row] = (lmi_id, i, j)
            end
        end
    end
    function __add(lmi_id, k, i, j, v)
        I, J, V, _, _ = A[lmi_id, k]
        push!(I, i)
        push!(J, j)
        push!(V, v)
        return
    end
    function _add(lmi_id, k, i, j, coef)
        __add(lmi_id, k, i, j, coef)
        if i != j
            __add(lmi_id, k, j, i, coef)
        end
        return
    end
    for row in eachindex(back)
        lmi_id, i, j = back[row]
        _add(lmi_id, 1, i, j, -psd_AC.constants[row])
    end
    for var = 1:n
        for k in nzrange(psd_A, var)
            lmi_id, i, j = back[rowvals(psd_A)[k]]
            col = 1 + var
            _add(lmi_id, col, i, j, nonzeros(psd_A)[k])
        end
    end
    dest.max_sense = MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
    obj = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}())
    # objective_constant = MOI.constant(obj) # TODO # MK: done(?)
    b_const = obj.constant
    b_const = dest.max_sense ? -b_const : b_const
    b0 = zeros(T, n)
    for term in obj.terms
        b0[term.variable.value] += term.coefficient
    end
    b = dest.max_sense ? b0 : -b0
    # b = max_sense ? -b0 : b0

    AA = SparseMatrixCSC{T,Int}[sparse(IJV...) for IJV in A]
    model = MyModel(
        -AA[:,1],
        AA[:,2:end],
        b,
        b_const,
        convert(SparseVector{T,Int64}, sparsevec(Cd_lin.constants)),
        C_lin,
        msizes,
    )
    # FIXME this does not work if an option is changed between `MOI.copy_to` and `MOI.optimize!`
    options = copy(dest.options)
    if dest.silent
        options["verb"] = 0
    end
    dest.lin_cones = Cd_lin.sets
    dest.solver, dest.halpha = load(model, options; T)
    return MOI.Utilities.identity_index_map(src)
end

function MOI.copy_to(dest::Optimizer{T}, src::MOI.Utilities.UniversalFallback{OptimizerCache{T}}) where {T}
    return MOI.copy_to(dest, src.model)
end

function MOI.copy_to(dest::Optimizer{T}, src::MOI.ModelLike) where {T}
    cache = OptimizerCache{T}()
    index_map = MOI.copy_to(cache, src)
    MOI.copy_to(dest, cache)
    return index_map
end

function MOI.get(optimizer::Optimizer, ::MOI.SolveTimeSec)
    return optimizer.solver.tottime
end

function MOI.get(optimizer::Optimizer, ::MOI.RawStatusString)
    # TODO I'd probably do an `if`-`else` here like for `MOI.TerminationStatus`
    #      except that here you are free to communicate any message to the user,
    #      you are not constrained to an any like `MOI.TerminationStatus`
    return "Terminated with status $(optimizer.solver.status)"
end

function MOI.get(optimizer::Optimizer, ::MOI.TerminationStatus)
    if isnothing(optimizer.solver) || optimizer.solver.status == 0
        return MOI.OPTIMIZE_NOT_CALLED
    elseif optimizer.solver.status == 1
        return MOI.OPTIMAL
    elseif optimizer.solver.status == 2
        return MOI.INFEASIBLE
    elseif optimizer.solver.status == 3
        return MOI.INFEASIBLE_OR_UNBOUNDED
    else
        @assert optimizer.solver.status == 4
        return MOI.ITERATION_LIMIT
    end
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index != MOI.get(model, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    term = MOI.get(model, MOI.TerminationStatus())
    if term == MOI.OPTIMIZE_NOT_CALLED
        return MOI.NO_SOLUTION
    elseif term == MOI.OPTIMAL
        return MOI.FEASIBLE_POINT
    elseif term == MOI.INFEASIBLE
        return MOI.INFEASIBLE_POINT
    elseif term == MOI.INFEASIBLE_OR_UNBOUNDED
        return MOI.UNKNOWN_RESULT_STATUS
    else
        @assert term == MOI.ITERATION_LIMIT
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if attr.result_index != MOI.get(model, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    term = MOI.get(model, MOI.TerminationStatus())
    if term == MOI.OPTIMIZE_NOT_CALLED
        return MOI.NO_SOLUTION
    elseif term == MOI.OPTIMAL
        return MOI.FEASIBLE_POINT
    elseif term == MOI.INFEASIBLE
        # The solution doesn't seem to be a valid `MOI.INFEASIBILITY_CERTIFICATE`,
        # e.g., it's failing `test_infeasible_affine_MIN_SENSE`
        return MOI.UNKNOWN_RESULT_STATUS
    elseif term == MOI.INFEASIBLE_OR_UNBOUNDED
        return MOI.UNKNOWN_RESULT_STATUS
    else
        @assert term == MOI.ITERATION_LIMIT
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    if MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        return 0
    else
        return 1
    end
end

function MOI.get(optimizer::Optimizer{T}, attr::MOI.ObjectiveValue) where {T}
    MOI.check_result_index_bounds(optimizer, attr)
    val = Solvers.dual_obj(optimizer.solver.model, optimizer.solver.y)::T
    return optimizer.max_sense ? -val : val
end

function MOI.get(optimizer::Optimizer{T}, attr::MOI.DualObjectiveValue) where {T}
    MOI.check_result_index_bounds(optimizer, attr)
    val = Solvers.obj(optimizer.solver.model, optimizer.solver.X_lin, optimizer.solver.X)::T
    return optimizer.max_sense ? -val : val
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    return model.solver.y[vi.value]
end

function MOI.get(
    optimizer::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{VAF{T},PSD},
) where {T}
    MOI.check_result_index_bounds(optimizer, attr)
    lmi_id = optimizer.lmi_id[ci]
    X = optimizer.solver.X[lmi_id]
    n = optimizer.solver.model.msizes[lmi_id]
    return [X[i, j] for j = 1:n for i = 1:j]::Vector{T}
end

function MOI.get(
    optimizer::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{VAF{T},NNG},
) where {T}
    MOI.check_result_index_bounds(optimizer, attr)
    rows = MOI.Utilities.rows(optimizer.lin_cones, ci)
    return optimizer.solver.X_lin[rows]::Vector{T}
end
