# min (1-x^2) * (1-y^2) s.t. 1-x^2 >= 0 , 1 - y^2 >= 0

using LinearAlgebra, DynamicPolynomials
using JuMP
using CSDP
using Loraine
using SumOfSquares
using MultiFloats
using Brisk
# using Mosek,MosekTools

# loraine = Loraine.Optimizer{Float64x4}
# import Dualization
# dual_loraine = Dualization.dual_optimizer(loraine)
# solver = optimizer_with_attributes(dual_loraine, MOI.Silent() => false)

solver = optimizer_with_attributes(Brisk.Optimizer, MOI.Silent() => false)

model = SOSModel(solver)
# set_attribute(model, "initpoint", 1)
# set_attribute(model, "eDIMACS", 1e-6)
# # set_attribute(model, "eDIMACS", 1e-20)
# set_attribute(model, "maxit", 300)
# set_attribute(model, "MSK_IPAR_INTPNT_MAX_NUM_COR", 0)   # kill multiple correctors
set_attribute(model, "prec", 500)
# set_attribute(model, "prec", "qd")
set_attribute(model, MOI.NumberOfThreads(), 12)

@polyvar x y

p = (1-x^2) * (1-y^2)
S = @set 1-x^2 >= 0 && 1 - y^2 >= 0
d = 18

@variable(model, α)
@objective(model, Max, α)
@constraint(model, c1, p >= α, domain = S, maxdegree = 2*d)
optimize!(model)

solution_summary(model)

# --- effective condition number ---
# Filters out numerically-zero eigenvalues (rank-deficiency at the optimum
# is expected/generic for SOS relaxations) before taking the ratio, so the
# result reflects conditioning on the matrix's effective range space
# rather than being dominated by ~0 (or slightly negative) noise eigenvalues.
function effective_cond(M::AbstractMatrix; reltol = 1e-12)
    ev = eigvals(Symmetric(Matrix(M)))
    λmax = maximum(abs, ev)
    sig = filter(λ -> λ > reltol * λmax, ev)
    println("Number of effective eigenvalues: ", length(sig))
    isempty(sig) && return Inf
    # @show sig
    return maximum(sig) / minimum(sig)
end

# --- main (index 0) Gram / moment matrices ---
Q = gram_matrix(c1).Q
evQ = eigvals(Symmetric(Matrix(Q)))
println("Gram matrix eigenvalues:")
for λ in evQ
    println(@sprintf("%.2e", λ))
end
println("Gram matrix Q0 effective condition number: ", @sprintf("%.2e", effective_cond(Q)))

ν = moment_matrix(c1)
evν = eigvals(Symmetric(Matrix(ν.Q)))
println("Moment matrix eigenvalues:")
for λ in evν
    println(@sprintf("%.2e", λ))
end
println("Moment matrix ν0 effective condition number: ", @sprintf("%.2e", effective_cond(ν.Q)))
# println("Effective condition number: ", @sprintf("%.2e", effective_cond(ν.Q) * effective_cond(Q)))


# # --- localizing moment matrices (dual side), reconstructed by hand ---
# ν0mono = SumOfSquares.MB.keys_as_monomials(ν.basis)
# momdict = Dict{eltype(ν0mono),Float64}()
# for (i, mi) in enumerate(ν0mono), (j, mj) in enumerate(ν0mono)
#     momdict[mi * mj] = ν.Q[i, j]
# end

# function localizing_matrix(gi, basis_i, momdict)
#     n = length(basis_i)
#     L = zeros(Float64, n, n)
#     for a in 1:n, b in 1:n
#         s = 0.0
#         for t in SumOfSquares.MP.terms(gi)
#             mono = basis_i[a] * basis_i[b] * SumOfSquares.MP.monomial(t)
#             s += SumOfSquares.MP.coefficient(t) * get(momdict, mono, 0.0)
#         end
#         L[a, b] = s
#     end
#     return L
# end
# Ls = Matrix{Float64}[]
# for (i, gm) in enumerate(mults)
#     basis_i = SumOfSquares.MB.keys_as_monomials(gm.basis)
#     push!(Ls, localizing_matrix(g[i], basis_i, momdict))
# end

# for (i, Li) in enumerate(Ls)
#     Li = (Li + Li') / 2
#     println("Localizing moment matrix L$i effective condition number: ", @sprintf("%.2e", effective_cond(Li)))
# end

# # --- combined effective condition number per localizing constraint ---
# for i in 1:length(mults)
#     Li = (Ls[i] + Ls[i]') / 2
#     kappa_eff_i = effective_cond(Qs[i]) * effective_cond(Li)
#     println("kappa_eff (constraint $i, loc. matrix): ", @sprintf("%.2e", kappa_eff_i))
# end

# # --- combined effective condition number (main Gram x main moment) ---
# println("kappa_eff: ", @sprintf("%.2e", effective_cond(ν.Q) * effective_cond(Q)))

# println("Effective ratio:            ", @sprintf("%.2e", effective_cond(ν.Q) / effective_cond(Q)))