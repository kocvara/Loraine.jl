using MathOptInterface
const MOI = MathOptInterface

MOI.Utilities.@product_of_sets(Nonnegatives, MOI.Nonnegatives)

MOI.Utilities.@product_of_sets(PSD, MOI.PositiveSemidefiniteConeTriangle)

MOI.Utilities.@struct_of_constraints_by_set_types(
    PSDOrNot,
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.Nonnegatives,
)

const OptimizerCache = MOI.Utilities.GenericModel{
    Float64,
    MOI.Utilities.ObjectiveContainer{Float64},
    MOI.Utilities.VariablesContainer{Float64},
    PSDOrNot{Float64}{
        MOI.Utilities.MatrixOfConstraints{
            Float64,
            MOI.Utilities.MutableSparseMatrixCSC{
                Float64,
                Int,
                MOI.Utilities.OneBasedIndexing,
            },
            Vector{Float64},
            PSD{Float64},
        },
        MOI.Utilities.MatrixOfConstraints{
            Float64,
            MOI.Utilities.MutableSparseMatrixCSC{
                Float64,
                Int,
                MOI.Utilities.OneBasedIndexing,
            },
            Vector{Float64},
            Nonnegatives{Float64},
        },
    },
}

mutable struct Optimizer <: MOI.AbstractOptimizer
    model::Union{Nothing,MySolver}
    halpha::Union{Nothing,Halpha}

    function Optimizer()
        return new(nothing, nothing)
    end
end

function MOI.default_cache(::Optimizer, ::Type{Float64})
    return MOI.Utilities.UniversalFallback(OptimizerCache())
end

function MOI.is_empty(optimizer::Optimizer)
    return isnothing(optimizer.model)
end

function MOI.empty!(optimizer::Optimizer)
    optimizer.model = nothing
    return
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Loraine"

# MOI.supports

function MOI.supports(
    ::Optimizer,
    ::Union{
        MOI.ObjectiveSense,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
    },
)
    return true
end

const SUPPORTED_CONES = Union{MOI.Nonnegatives, MOI.PositiveSemidefiniteConeTriangle}

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.VectorAffineFunction{Float64}},
    ::Type{<:SUPPORTED_CONES},
)
    return true
end

function MOI.optimize!(optimizer::Optimizer)
    optimizer.model.to = TimerOutput()
    solve(optimizer.model, optimizer.halpha)
    return
end

function MOI.copy_to(dest::Optimizer, src::OptimizerCache)
    MOI.empty!(dest)
    F = MOI.VectorAffineFunction{Float64}
    PSD = MOI.PositiveSemidefiniteConeTriangle
    psd_AC = MOI.Utilities.constraints(src.constraints, F, PSD)
    Dc_lin = MOI.Utilities.constraints(
        src.constraints,
        F,
        MOI.Nonnegatives,
    )
    psd_A = convert(SparseMatrixCSC{Float64,Int}, psd_AC.coefficients)
    D_lin = convert(SparseMatrixCSC{Float64,Int}, Dc_lin.coefficients)
    n = MOI.get(src, MOI.NumberOfVariables())
    nlmi = MOI.get(src, MOI.NumberOfConstraints{F,PSD}())
    A = Tuple{Vector{Int},Vector{Int},Vector{Float64}}[(Int[], Int[], Float64[]) for _ in 1:nlmi, _ in 1:(1 + n)]
    back = Vector{Tuple{Int,Int,Int}}(undef, size(psd_A, 1))
    row = 0
    msizes = Int[]
    for (lmi_id, ci) in enumerate(MOI.get(src, MOI.ListOfConstraintIndices{F,PSD}()))
        set = MOI.get(src, MOI.ConstraintSet(), ci)
        push!(msizes, set.side_dimension)
        for j in 1:set.side_dimension
            for i in 1:j
                row += 1
                back[row] = (lmi_id, i, j)
            end
        end
    end
    function __add(lmi_id, k, i, j, v)
        I, J, V = A[lmi_id, k]
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
    for var in 1:n
        for k in nzrange(psd_A, var)
            lmi_id, i, j = back[rowvals(psd_A)[k]]
            col = 1 + var
            _add(lmi_id, col, i, j, nonzeros(psd_A)[k])
        end
    end
    max_sense = MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
    obj = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    # objective_constant = MOI.constant(obj) # TODO
    b0 = zeros(n)
    for term in obj.terms
        b0[term.variable.value] += term.coefficient
    end
    b = max_sense ? b0 : -b0
    AA = Any[sparse(IJV...) for IJV in A]
    for i in 1:nlmi
        @show i
        for j in 1:(n+1)
            @show j
            display(AA[i, j])
        end
    end
    @show b
    @show Dc_lin.constants
    display(D_lin)
    @show n
    @show msizes
    @show nlmi
    model = MyModel(
        AA,
        Solvers._prepare_A(AA)...,
        _sparse(b),
        sparsevec(Dc_lin.constants),
        D_lin,
        n,
        msizes,
        length(Dc_lin.constants),
        nlmi,
    )
    options = Dict()
    dest.model, dest.halpha = load(model, options)
    return MOI.Utilities.identity_index_map(src)
end

_sparse(x::Vector) = sparse(reshape(x, length(x), 1))

function MOI.copy_to(dest::Optimizer, src::MOI.ModelLike)
    cache = OptimizerCache()
    index_map = MOI.copy_to(cache, src)
    MOI.copy_to(dest, cache)
    return index_map
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if isnothing(model.model) || model.model.status == 0
        return MOI.OPTIMIZE_NOT_CALLED
    elseif model.model.status == 1
        return MOI.OPTIMAL
    else
        @assert model.model.status == 2
        return MOI.ITERATION_LIMIT
    end
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    if MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        return 0
    else
        return 1
    end
end

function MOI.get(model::Optimizer, attr::MOI.VariablePrimal, vi::MOI.VariableIndex)
    MOI.check_result_index_bounds(model, attr)
    return model.model.y[vi.value]
end
