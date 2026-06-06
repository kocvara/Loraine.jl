# =============================================================================
# delzell_momsos_loraine.jl
#
# Moment-SOS hierarchy for the Delzell PSD-not-SOS quaternary octic on the
# unit sphere S^3, solved with Loraine.jl in multiprecision arithmetic.
#
#   rho_r = min  L(f)   s.t.  L : R[x]/(1-|x|^2) -> R,  L(1) = 1,
#                             L(p^2) >= 0  for deg p <= r,
#
# for relaxation orders r = RMIN..RMAX, together with the dual SOS certificate
#   f - rho_r = sigma + q * (1 - |x|^2),   sigma SOS, deg sigma <= 2r.
#
# Design (validated against a brute-force monomial Lasserre relaxation in an
# independent Python/Clarabel prototype, values agree to solver tolerance):
#
#   * Quotient basis. All polynomials are reduced modulo the sphere ideal
#     (g) = (1 - sum x_i^2) using the exact normal form that eliminates the
#     coordinate x_E (E = 4): the Gram basis is {T_alpha : |alpha| <= r,
#     alpha_E <= 1} (506 elements at r = 10 instead of 1001), and the moment
#     matrix is indexed by it. This is *exactly* equivalent to the standard
#     order-r relaxation (NF preserves total degree; multipliers stay of
#     degree <= 2r-2), it is not a weakening.
#
#   * Chebyshev basis. Everything is expressed in tensorized Chebyshev
#     polynomials T_gamma(x) = prod_i T_{gamma_i}(x_i) for conditioning.
#     All structural data (products, normal forms, cofactors) is computed in
#     exact rational arithmetic (Rational{BigInt}) and rounded to Float64
#     only once, when handed to the solver.
#
#   * Symmetry. The group of f here has order 32: the swap x1 <-> x2 and all
#     sign flips (the script *detects* the group of whatever f you define, so
#     you may change f below). Sign symmetry => pseudo-moments supported on
#     componentwise-even multi-indices and the moment matrix block-
#     diagonalizes into <= 16 parity cells. Permutation symmetry => moments
#     constant on orbits (variable count drops), only one cell per cell-orbit
#     needs a PSD constraint, and stabilized cells split further through
#     +/-1 character combinations for commuting involutions. At r = 10 the
#     largest block is 34 (vs 1001 unreduced) with ~170 free moments.
#
#   * Solver. Loraine.jl (interior point, direct linear solver kit = 0) via
#     JuMP, with iterates in T = BigFloat at PRECISION_BITS = 512 bits by
#     default. IMPORTANT CAVEATS:
#       - Loraine's MOI wrapper stores the problem *data* in Float64
#         regardless of T (verified in src/MOI_wrapper.jl of v0.2.5: the
#         OptimizerCache and copy_to path are Float64-typed). Multiprecision
#         buys you a numerically robust interior point method that can push
#         the DIMACS errors of the *Float64-rounded problem* down to ~1e-30,
#         curing the Schur-complement ill-conditioning that kills Float64
#         IPMs near the optimum of these hierarchies. The distance between
#         the Float64-rounded problem and the exact one is O(1e-16) in the
#         data; plan accordingly when interpreting digits of rho_r.
#       - The officially supported multiprecision types are Float64xN
#         (MultiFloats.jl), N = 2..8; Float64x8 has ~424 bits. BigFloat goes
#         through the same generic code paths (the kit = 0 direct solver uses
#         generic Cholesky only) and worked in our source review, but it is
#         not advertised by the README. If BigFloat errors on your Loraine
#         version, set ARITH = :float64x8 below (one line) -- this is the
#         documented configuration, at 424 bits instead of 512.
#       - High-precision quantities are extracted by reading the inner
#         optimizer directly (solver.y, solver.X); the standard MOI getters
#         truncate constraint duals to Float64.
#
#   * Reference values (independent Python/Clarabel validation, accuracy
#     ~1e-7 .. 1e-8; your Loraine run should reproduce them within the
#     Float64-data limitation):
#         rho_4 ~ -4.719e-4,  rho_5 ~ -5.594e-5,  rho_6 ~ -1.118e-5.
#     rho_r < 0 strictly for ALL r: f has "bad points" (the x4-axis), so
#     |x|^{2N} f is never SOS and finite convergence is impossible -- this is
#     what makes the example a good stress test for convergence rates.
#
#   * Output. For each r a file OUTDIR/order_r.jls (Julia Serialization)
#     containing the pseudo-moment vector (on the quotient basis and as a
#     full dictionary of Chebyshev moments), the Gram/SOS certificate blocks
#     in precision T, the multiplier q, exact bases and group data, residuals
#     and solver diagnostics; plus a human-readable OUTDIR/summary.txt.
#
# Usage:
#     import Pkg; Pkg.add(["JuMP", "Loraine"])                 # once
#     julia delzell_momsos_loraine.jl                # or include(...) in a REPL
#
# Reload results later with Loraine in scope (it provides the MultiFloat
# types if ARITH = :float64x8 was used):
#     using Loraine, Serialization
#     data = deserialize("delzell_results/order_6.jls")
#
# Expected runtime (single thread): seconds for r = 4..6, minutes for
# r = 7..8, and possibly hours for r = 10 with 512-bit BigFloat (Float64x8 is
# substantially faster). Memory is modest (< 2 GB).
#
# Validation companion: validate_pipeline.py (same architecture in Python,
# cross-checked against brute force at r = 4, 5).
# =============================================================================

# ----------------------------------------------------------------- 0. config
const RMIN = 8
const RMAX = 8
const PRECISION_BITS = 512          # BigFloat precision for iterates/post-processing
const ARITH = :bigfloat            # :bigfloat (512 bits) or :float64x8 (~424 bits, officially supported)
const USE_PERM_SYMMETRY = true      # false => parity (sign) reduction only
const OUTDIR = "delzell_results"
const EDIMACS = 1.0e-30             # Loraine stopping tolerance (DIMACS errors)
const MAXIT = 300
const VERB = 1                      # Loraine verbosity 0/1/2

using LinearAlgebra, SparseArrays, Serialization, Printf, Dates, Random
using JuMP, Loraine

using MultiFloats
using Base.Threads
# Plain threaded mul! — no LoopVectorization, just @threads over columns
function LinearAlgebra.mul!(
        C::Matrix{T},
        A::Matrix{T},
        B::Matrix{T}) where {T <: MultiFloats.MultiFloat}
    m, k = size(A)
    _, n = size(B)
    fill!(C, zero(T))
    @threads for j in 1:n
        for l in 1:k
            @inbounds blj = B[l, j]
            for i in 1:m
                @inbounds C[i, j] += A[i, l] * blj
            end
        end
    end
    return C
end

setprecision(BigFloat, PRECISION_BITS)
# Float64x8 is reached through Loraine's own dependency on MultiFloats, so no
# extra package is needed; the branch is only evaluated when selected.
const T = ARITH === :bigfloat ? BigFloat : Loraine.Solvers.MultiFloats.Float64x8

const N = 4                          # number of variables x1..x4
const E = 4                          # eliminated coordinate (must be fixed by the
                                     # permutation symmetries of f; asserted below)

const Expo = NTuple{N,Int}
const Q    = Rational{BigInt}
const Poly = Dict{Expo,Q}            # Chebyshev coefficient dictionaries

# ------------------------------------------------------ 1. === DEFINE f HERE ===
# Delzell's quaternary octic with bad points (PhD thesis, Stanford 1980,
# pp. 59-61; see O. Benoist, arXiv:2103.16134, S2, for the historical record):
# we use the representative
#     f(x1,x2,x3,x4) = x4^2 * M(x1,x2,x3),
#     M(x,y,z) = x^4 y^2 + x^2 y^4 + z^6 - 3 x^2 y^2 z^2   (Motzkin),
# a PSD form whose bad locus is the whole x4-axis: no identity
# |x|^{2N} f = SOS exists, hence rho_r < 0 for every r.
# If your intended "Delzell octic" is a different representative, replace the
# dictionary below (exponent tuple => exact rational coefficient); group
# detection, symmetry reduction and all bases adapt automatically, provided
# the permutation symmetries of your f fix coordinate E (else change E).
f_mono = Dict{Expo,Q}(
    (4, 2, 0, 2) => Q(1),
    (0, 4, 2, 2) => Q(1),
    (2, 0, 4, 2) => Q(1),
    (2, 2, 2, 2) => Q(-3),
    (0, 0, 8, 0) => Q(1),
)
# Alternative (Choi-Lam based, bad points on the x4-axis as well):
# f_mono = Dict{Expo,Q}((4,2,0,2)=>Q(1), (0,4,2,2)=>Q(1), (2,0,4,2)=>Q(1),
#                       (2,2,2,2)=>Q(-3))

# --------------------------------------------- 2. exact Chebyshev/NF layer
"x^k in the Chebyshev basis: x^k = 2^{1-k} sum_{j == k (2)} binom(k,(k-j)/2) T_j, j=0 halved."
function mono1d_to_cheb(k::Int)::Dict{Int,Q}
    k == 0 && return Dict(0 => Q(1))
    out = Dict{Int,Q}()
    for j in (k % 2):2:k
        c = Q(binomial(BigInt(k), BigInt((k - j) ÷ 2)), BigInt(2)^(k - 1))
        j == 0 && (c //= 2)
        out[j] = get(out, j, Q(0)) + c
    end
    return out
end

"Monomial dictionary -> Chebyshev dictionary (exact)."
function mono_to_cheb(d::Dict{Expo,Q})::Poly
    out = Poly()
    for (gamma, c) in d
        tabs = [mono1d_to_cheb(g) for g in gamma]
        stack = [(Int[], c)]
        for t in tabs
            stack = [(vcat(idx, [j]), coef * cc) for (idx, coef) in stack for (j, cc) in t]
        end
        for (idx, coef) in stack
            key = Expo(idx)
            out[key] = get(out, key, Q(0)) + coef
        end
    end
    return Poly(k => v for (k, v) in out if v != 0)
end

"""
T_alpha * T_beta as an exact Chebyshev dictionary. Per coordinate
T_a T_b = (T_{a+b} + T_{|a-b|})/2, ACCUMULATED so that the degenerate cases
a = 0 or b = 0 (where a+b = |a-b|) correctly give T_a T_0 = T_a.
(A non-accumulating implementation silently halves these products; this was
caught by an exact evaluation test at rational points on the sphere.)
"""
function cheb_mul(alpha::Expo, beta::Expo)::Poly
    parts = Vector{Dict{Int,Q}}(undef, N)
    for i in 1:N
        a, b = alpha[i], beta[i]
        p = Dict{Int,Q}()
        for (k, c) in ((a + b, Q(1, 2)), (abs(a - b), Q(1, 2)))
            p[k] = get(p, k, Q(0)) + c
        end
        parts[i] = p
    end
    res = Poly()
    stack = [(Int[], Q(1))]
    for p in parts
        stack = [(vcat(idx, [j]), coef * cc) for (idx, coef) in stack for (j, cc) in p]
    end
    for (idx, coef) in stack
        key = Expo(idx)
        res[key] = get(res, key, Q(0)) + coef
    end
    return res
end

addterm!(d::Poly, idx::Expo, c::Q) = (d[idx] = get(d, idx, Q(0)) + c; nothing)

const _nf_cache = Dict{Expo,Tuple{Poly,Poly}}()

"""
Normal form of T_gamma modulo g = 1 - sum_i x_i^2, eliminating coordinate E:
returns (nf, h) with T_gamma = nf + g*h exactly, nf supported on indices with
gamma_E <= 1, total degree and componentwise parity preserved.
Uses T_2(x_E) = -2 - sum_{i != E} T_2(x_i) - 2g and, for k = gamma_E >= 3,
T_k = 2 T_2 T_{k-2} - T_{|k-4|}; the recursion strictly decreases gamma_E.
"""
function nfT(gamma::Expo)::Tuple{Poly,Poly}
    haskey(_nf_cache, gamma) && return _nf_cache[gamma]
    k = gamma[E]
    if k <= 1
        r = (Poly(gamma => Q(1)), Poly())
        _nf_cache[gamma] = r
        return r
    end
    nf, h = Poly(), Poly()
    others = [i for i in 1:N if i != E]
    terms = Poly()
    if k == 2
        gp = Expo(ntuple(i -> i == E ? 0 : gamma[i], N))
        addterm!(terms, gp, Q(-2))
        for i in others
            a = gp[i]
            for (j, cc) in ((a + 2, Q(1, 2)), (abs(a - 2), Q(1, 2)))
                idx = Expo(ntuple(t -> t == i ? j : gp[t], N))
                addterm!(terms, idx, -cc)
            end
        end
        addterm!(h, gp, Q(-2))
    else
        g2 = Expo(ntuple(i -> i == E ? k - 2 : gamma[i], N))
        g4 = Expo(ntuple(i -> i == E ? abs(k - 4) : gamma[i], N))
        addterm!(terms, g2, Q(-4))
        for i in others
            a = g2[i]
            for (j, cc) in ((a + 2, Q(1, 2)), (abs(a - 2), Q(1, 2)))
                idx = Expo(ntuple(t -> t == i ? j : g2[t], N))
                addterm!(terms, idx, -2 * cc)
            end
        end
        addterm!(terms, g4, Q(-1))
        addterm!(h, g2, Q(-4))
    end
    for (idx, c) in terms
        c == 0 && continue
        n2, h2 = nfT(idx)
        for (jj, cc) in n2; addterm!(nf, jj, c * cc); end
        for (jj, cc) in h2; addterm!(h, jj, c * cc); end
    end
    filter!(p -> p.second != 0, nf)
    filter!(p -> p.second != 0, h)
    _nf_cache[gamma] = (nf, h)
    return (nf, h)
end

"Normal form of a Chebyshev dictionary: p = nf + g*h."
function nf_poly(p::Poly)::Tuple{Poly,Poly}
    nf, h = Poly(), Poly()
    for (gamma, c) in p
        n2, h2 = nfT(gamma)
        for (j, cc) in n2; addterm!(nf, j, c * cc); end
        for (j, cc) in h2; addterm!(h, j, c * cc); end
    end
    filter!(p_ -> p_.second != 0, nf)
    filter!(p_ -> p_.second != 0, h)
    return (nf, h)
end

"Exact T_j(x) at a rational (or any numeric) point, by the recurrence."
function chebT1(j::Int, x)
    t0, t1 = one(x), x
    j == 0 && return t0
    for _ in 1:(j - 1)
        t0, t1 = t1, 2x * t1 - t0
    end
    return t1
end
function evalcheb(d::Poly, pt)
    s = zero(typeof(pt[1] * pt[1]))
    for (gamma, c) in d
        v = oftype(s, c)
        for i in 1:N
            v *= chebT1(gamma[i], pt[i])
        end
        s += v
    end
    return s
end

# Exact self-test of the NF identity at random rational points.
function selftest_nf()
    rng = Random.MersenneTwister(0)
    for _ in 1:20
        gamma = Expo(ntuple(_ -> rand(rng, 0:6), N))
        pt = ntuple(_ -> Q(rand(rng, -9:9), rand(rng, 1:9)), N)
        gval = 1 - sum(x * x for x in pt)
        nf, h = nfT(gamma)
        lhs = evalcheb(Poly(gamma => Q(1)), pt)
        rhs = evalcheb(nf, pt) + gval * evalcheb(h, pt)
        @assert lhs == rhs "NF identity failed at gamma=$gamma"
        @assert all(idx[E] <= 1 for idx in keys(nf)) "NF not in quotient basis"
    end
    @assert mono1d_to_cheb(2) == Dict(0 => Q(1, 2), 2 => Q(1, 2))
    @assert mono1d_to_cheb(4) == Dict(0 => Q(3, 8), 2 => Q(1, 2), 4 => Q(1, 8))
    println("NF identity: exact OK")
end

# ----------------------------------------------------- 3. symmetry group of f
"Action of x_i -> sgn_i * x_{perm(i)} on a monomial dictionary (exact)."
function act_poly_mono(d::Dict{Expo,Q}, perm::Expo, sgn::NTuple{N,Int})::Dict{Expo,Q}
    out = Dict{Expo,Q}()
    for (gamma, c) in d
        ng = zeros(Int, N)
        s = 1
        for i in 1:N
            ng[perm[i]] = gamma[i]
            isodd(gamma[i]) && sgn[i] < 0 && (s = -s)
        end
        key = Expo(ng)
        out[key] = get(out, key, Q(0)) + c * s
    end
    return Dict{Expo,Q}(k => v for (k, v) in out if v != 0)
end

function detect_group(fm::Dict{Expo,Q})
    allperms = [Expo(p) for p in permutations_(collect(1:N))]
    allsigns = [NTuple{N,Int}(s) for s in Iterators.product(ntuple(_ -> (1, -1), N)...)]
    G = Tuple{Expo,NTuple{N,Int}}[]
    for p in allperms, s in allsigns
        act_poly_mono(fm, p, s) == fm && push!(G, (p, s))
    end
    permpart = sort(unique(first.(G)))
    return G, permpart
end

# minimal permutations generator (avoid Combinatorics dependency)
function permutations_(v::Vector{Int})
    length(v) <= 1 && return [copy(v)]
    out = Vector{Vector{Int}}()
    for (i, x) in enumerate(v)
        rest = vcat(v[1:i-1], v[i+1:end])
        for p in permutations_(rest)
            push!(out, vcat([x], p))
        end
    end
    return out
end

# ----------------------------------------------------------- 4. index sets
"Moment indices Delta(r) = {delta componentwise even, delta_E = 0, |delta| <= 2r}."
function moment_basis(r::Int)::Vector{Expo}
    out = Expo[]
    for g in Iterators.product(ntuple(_ -> 0:2:2r, N - 1)...)
        sum(g) <= 2r || continue
        idx = collect(Int, g)
        insert!(idx, E, 0)
        push!(out, Expo(idx))
    end
    return sort(out)
end

"Quotient Gram basis B_r^q = {alpha : |alpha| <= r, alpha_E <= 1}."
function gram_basis(r::Int)::Vector{Expo}
    out = Expo[]
    for g in Iterators.product(ntuple(_ -> 0:r, N)...)
        if sum(g) <= r && g[E] <= 1
            push!(out, g)
        end
    end
    return sort(out)
end

"Permutation action on multi-indices: (p.a)[p[i]] = a[i]."
function permute_idx(a::Expo, p::Expo)::Expo
    v = zeros(Int, N)
    for i in 1:N
        v[p[i]] = a[i]
    end
    return Expo(v)
end

# =============================================================================
# 5. build and solve one relaxation order
# =============================================================================
"""
Build the symmetry-reduced order-r moment relaxation, solve it with Loraine,
reconstruct the SOS certificate, verify it in BigFloat, and return everything.
"""
function solve_order(r::Int, f_nf::Poly, f_h::Poly, permpart::Vector{Expo})
    t_total = time()
    Delta = moment_basis(r)
    m = length(Delta)
    dindex = Dict(d => i for (i, d) in enumerate(Delta))
    B = gram_basis(r)

    # ----- parity cells of the Gram basis (sign symmetry)
    cells = Dict{Expo,Vector{Expo}}()
    for a in B
        c = Expo(ntuple(i -> a[i] % 2, N))
        push!(get!(cells, c, Expo[]), a)
    end
    for c in keys(cells); sort!(cells[c]); end

    # ----- permutation invariance of moments: y constant on permpart-orbits.
    # (Each p in permpart fixes E, so T_{p.delta} is again a basis element:
    # the invariance constraints y_{p.delta} = y_delta are a pure permutation
    # system; the exact RREF of the validated prototype reduces to orbit
    # identification, which we use directly.)
    id_perm = Expo(collect(1:N))
    use_perm = USE_PERM_SYMMETRY && length(permpart) > 1
    orbit_rep = Dict{Expo,Expo}()
    for d in Delta
        orb = use_perm ? sort(unique(permute_idx(d, p) for p in permpart)) : [d]
        for x in orb
            orbit_rep[x] = orb[1]
        end
    end
    freeidx = sort(unique(values(orbit_rep)))            # orbit representatives
    fpos = Dict(d => k for (k, d) in enumerate(freeidx))
    zerokey = Expo(zeros(Int, N))
    f0 = fpos[zerokey]                                   # column of y_0 == 1
    nfree = length(freeidx)
    nu = nfree - 1                                       # JuMP variables (y_0 fixed)
    ucol(d::Expo) = (k = fpos[orbit_rep[d]]; k == f0 ? 0 : (k < f0 ? k : k - 1))

    # affine data of sum_delta vec_delta * y_delta in (const, sparse lin over u)
    function ycomb_exact(vec::Dict{Expo,Q})
        c0 = Q(0)
        lin = Dict{Int,Q}()
        for (d, c) in vec
            c == 0 && continue
            k = ucol(d)
            if k == 0
                c0 += c
            else
                lin[k] = get(lin, k, Q(0)) + c
            end
        end
        return c0, lin
    end

    # ----- cell orbits under the permutation part (PSD needed on reps only)
    actcell(c::Expo, p::Expo) = permute_idx(c, p)
    reps = Expo[]
    if use_perm
        seen = Set{Expo}()
        for c in sort(collect(keys(cells)))
            c in seen && continue
            orb = sort(unique(actcell(c, p) for p in permpart))
            push!(reps, orb[1])
            union!(seen, orb)
        end
    else
        reps = sort(collect(keys(cells)))
    end

    # ----- maximal set of commuting involutions of the permutation part
    compose(p::Expo, q::Expo) = Expo(ntuple(i -> p[q[i]], N))
    invols = [p for p in permpart if p != id_perm && compose(p, p) == id_perm]
    Emax = Expo[]
    for p in invols
        if all(compose(p, q) == compose(q, p) for q in Emax)
            push!(Emax, p)
        end
    end

    # ----- congruence blocks: per representative cell, either the full cell or
    # +/-1 character-sum splits under the stabilizing involutions
    # block = (cell, Ublocks) ; each U a Vector of sparse vectors [(pos, coeff)]
    BlockU = Vector{Vector{Tuple{Int,Q}}}
    blocks = Tuple{Expo,Vector{BlockU},String}[]
    for c in reps
        idxs = cells[c]
        pos = Dict(a => i for (i, a) in enumerate(idxs))
        Ec = use_perm ? [p for p in Emax if actcell(c, p) == c] : Expo[]
        if isempty(Ec)
            U = BlockU([[(i, Q(1))] for i in 1:length(idxs)])
            push!(blocks, (c, [U], "full"))
            continue
        end
        # elementary abelian group generated by Ec
        elems = Set{Expo}([id_perm])
        for p in Ec
            elems = union(elems, Set(compose(q, p) for q in elems))
        end
        # orbits of the cell under it
        orbs = Vector{Vector{Expo}}()
        seen2 = Set{Expo}()
        for a in idxs
            a in seen2 && continue
            o = sort(unique(permute_idx(a, g) for g in elems))
            push!(orbs, o)
            union!(seen2, o)
        end
        kgen = length(Ec)
        Ublocks = BlockU[]
        for chi in Iterators.product(ntuple(_ -> (1, -1), kgen)...)
            vecs = Vector{Vector{Tuple{Int,Q}}}()
            for o in orbs
                a0 = o[1]
                # signed word map over the group: g -> chi-sign; nothing if chi
                # is nontrivial on the stabilizer of a0 (vector vanishes)
                seenw = Dict{Expo,Union{Int,Nothing}}(id_perm => 1)
                frontier = [(id_perm, 1)]
                while !isempty(frontier)
                    (g0, s0) = pop!(frontier)
                    for (gi, p) in enumerate(Ec)
                        g1 = compose(g0, p)
                        s1 = s0 * chi[gi]
                        if !haskey(seenw, g1)
                            seenw[g1] = s1
                            push!(frontier, (g1, s1))
                        elseif seenw[g1] !== nothing && seenw[g1] != s1
                            seenw[g1] = nothing
                        end
                    end
                end
                v = Dict{Expo,Int}()
                ok = true
                for (g0, s0) in seenw
                    if s0 === nothing
                        ok = false; break
                    end
                    ai = permute_idx(a0, g0)
                    if haskey(v, ai) && v[ai] != s0
                        ok = false; break
                    end
                    v[ai] = s0
                end
                ok || continue
                push!(vecs, [(pos[a], Q(s)) for (a, s) in sort(collect(v); by = first)])
            end
            isempty(vecs) || push!(Ublocks, vecs)
        end
        push!(blocks, (c, Ublocks, "split$(length(Ublocks))"))
    end

    # ----- moment-matrix structural data R^{ab}: NF(T_a T_b) and cofactor h_ab
    Rcache = Dict{Tuple{Expo,Expo},Tuple{Poly,Poly}}()
    function getR(a::Expo, b::Expo)
        key = a <= b ? (a, b) : (b, a)
        get!(Rcache, key) do
            nf, h = nf_poly(cheb_mul(key[1], key[2]))
            for (idx, _) in nf
                @assert all(iseven, idx) && idx[E] == 0 "R support outside Delta"
                @assert haskey(dindex, idx) "R support beyond degree 2r"
            end
            (nf, h)
        end
    end

    # ----- JuMP model (moment side):  min  L(f)  s.t. per-block M(y) >= 0
    model = Model(() -> Loraine.Optimizer{T}(); add_bridges = false)
    set_attribute(model, "kit", 0)            # direct solver (required for multiprecision)
    set_attribute(model, "eDIMACS", EDIMACS)
    set_attribute(model, "maxit", MAXIT)
    set_attribute(model, "verb", VERB)
    @variable(model, u[1:nu])

    fconst, flin = ycomb_exact(f_nf)
    obj = AffExpr(Float64(fconst))
    for (k, c) in flin
        add_to_expression!(obj, Float64(c), u[k])
    end
    @objective(model, Min, obj)

    conrefs = Tuple{Expo,BlockU,ConstraintRef}[]
    blocksizes = Int[]
    for (c, Ublocks, _) in blocks
        idxs = cells[c]
        for U in Ublocks
            nb = length(U)
            Mex = Matrix{AffExpr}(undef, nb, nb)
            for i in 1:nb, j in i:nb
                vec = Dict{Expo,Q}()
                for (pi, ci) in U[i], (pj, cj) in U[j]
                    nf, _ = getR(idxs[pi], idxs[pj])
                    w = ci * cj
                    for (idx, cc) in nf
                        vec[idx] = get(vec, idx, Q(0)) + w * cc
                    end
                end
                c0, lin = ycomb_exact(vec)
                ex = AffExpr(Float64(c0))
                for (k, cc) in lin
                    add_to_expression!(ex, Float64(cc), u[k])
                end
                Mex[i, j] = ex
                i != j && (Mex[j, i] = ex)
            end
            cr = @constraint(model, LinearAlgebra.Symmetric(Mex) in PSDCone())
            push!(conrefs, (c, U, cr))
            push!(blocksizes, nb)
        end
    end
    @printf("r = %d : |Delta| = %d, free moments = %d, blocks = %s\n",
            r, m, nfree, string(sort(blocksizes, rev = true)))

    optimize!(model)
    status = termination_status(model)
    stime = solve_time(model)

    # ----- high-precision extraction from the inner optimizer
    inner = unsafe_backend(model)              # Loraine.Optimizer{T}
    rawstatus = inner.solver.status
    uT = Vector{T}(undef, nu)
    for k in 1:nu
        uT[k] = inner.solver.y[optimizer_index(u[k]).value]
    end
    uB = BigFloat.(uT)
    # objective in BigFloat from the exact rational data
    rho = BigFloat(fconst) + sum(BigFloat(c) * uB[k] for (k, c) in flin; init = BigFloat(0))

    # pseudo-moments on Delta (BigFloat) and as a full even Chebyshev dictionary
    yD = Vector{BigFloat}(undef, m)
    for (i, d) in enumerate(Delta)
        k = ucol(d)
        yD[i] = k == 0 ? BigFloat(1) : uB[k]
    end
    moments = Dict{Expo,BigFloat}()
    for g in Iterators.product(ntuple(_ -> 0:2:2r, N)...)
        sum(g) <= 2r || continue
        gamma = g
        nf, _ = nfT(gamma)
        moments[gamma] = sum(BigFloat(c) * yD[dindex[idx]] for (idx, c) in nf;
                             init = BigFloat(0))
    end

    # ----- Gram blocks (duals) in precision T, assembled per representative cell
    Scells = Dict{Expo,Matrix{BigFloat}}()
    for (c, U, cr) in conrefs
        ci = optimizer_index(cr)
        lmi = inner.lmi_id[ci]
        S = BigFloat.(inner.solver.X[lmi])
        S = (S + S') / 2
        idxs = cells[c]
        nc = length(idxs)
        Uden = zeros(BigFloat, nc, length(U))
        for (jj, vecU) in enumerate(U), (pi, cc) in vecU
            Uden[pi, jj] = BigFloat(cc)
        end
        Qc = get!(Scells, c, zeros(BigFloat, nc, nc))
        Scells[c] = Qc + Uden * S * Uden'
    end
    # Reynolds-average to all cells (exact symmetrization of the certificate)
    perms_used = use_perm ? permpart : [id_perm]
    np = length(perms_used)
    Qfull = Dict{Expo,Matrix{BigFloat}}()
    for (c0, Qc) in Scells
        idxs0 = cells[c0]
        for p in perms_used
            c1 = actcell(c0, p)
            idxs1 = cells[c1]
            pos1 = Dict(a => i for (i, a) in enumerate(idxs1))
            P = zeros(BigFloat, length(idxs1), length(idxs0))
            for (i, a) in enumerate(idxs0)
                P[pos1[permute_idx(a, p)], i] = 1
            end
            Q1 = P * Qc * P' / np
            Qfull[c1] = haskey(Qfull, c1) ? Qfull[c1] + Q1 : Q1
        end
    end

    # ----- certificate residual on Delta, with automatic dual-sign detection,
    # and multiplier q from the exact cofactors:
    #   f - rho = sum_{a,b} Q_ab T_a T_b + q * g,
    #   q = f_h - sum_{a,b} Q_ab h_ab.
    function residual_and_q(sgn::Int)
        res = zeros(BigFloat, m)
        for (idx, c) in f_nf
            res[dindex[idx]] += BigFloat(c)
        end
        res[dindex[zerokey]] -= rho
        hsum = Dict{Expo,BigFloat}()
        for (c0, Qc) in Qfull
            idxs = cells[c0]
            for i in 1:length(idxs), j in 1:length(idxs)
                w = sgn * Qc[i, j]
                abs(w) < big"1e-60" && continue
                a, b = idxs[i], idxs[j]
                nf, h = getR(min(a, b), max(a, b))
                for (idx, cc) in nf
                    res[dindex[idx]] -= w * BigFloat(cc)
                end
                for (gg, cc) in h
                    hsum[gg] = get(hsum, gg, BigFloat(0)) + w * BigFloat(cc)
                end
            end
        end
        qch = Dict{Expo,BigFloat}()
        for (g, c) in f_h
            qch[g] = BigFloat(c) - get(hsum, g, BigFloat(0))
        end
        for (g, c) in hsum
            haskey(f_h, g) || (qch[g] = -c)
        end
        return maximum(abs.(res)), qch
    end
    res_p, q_p = residual_and_q(+1)
    res_m, q_m = residual_and_q(-1)
    dual_sign = res_p <= res_m ? +1 : -1
    residual = min(res_p, res_m)
    q_cheb = dual_sign == 1 ? q_p : q_m
    if dual_sign == -1
        for c in collect(keys(Qfull))
            Qfull[c] = -Qfull[c]
        end
    end
    residual > 1e-6 && @warn "certificate residual is large" r residual

    # pointwise check of f - rho = sigma + q*g at random points (BigFloat)
    rng = Random.MersenneTwister(1)
    pterr = BigFloat(0)
    f_cheb_full = mono_to_cheb(f_mono)
    for _ in 1:5
        x = ntuple(_ -> BigFloat(rand(rng)) * big"1.8" - big"0.9", N)
        gval = 1 - sum(t * t for t in x)
        fx = evalcheb(f_cheb_full, x)
        sig = BigFloat(0)
        for (c0, Qc) in Qfull
            idxs = cells[c0]
            vals = [evalcheb(Poly(a => Q(1)), x) for a in idxs]
            sig += vals' * Qc * vals
        end
        qx = sum(c * prod(chebT1(g[i], x[i]) for i in 1:N) for (g, c) in q_cheb;
                 init = BigFloat(0))
        pterr = max(pterr, abs(fx - rho - sig - qx * gval))
    end

    elapsed = time() - t_total
    @printf("        rho_%d = %.15e   status=%s (raw %d)   residual=%.2e   pointwise=%.2e   solve=%.1fs total=%.1fs\n",
            r, Float64(rho), string(status), rawstatus, Float64(residual),
            Float64(pterr), Float64(stime), elapsed)

    return Dict{String,Any}(
        "r" => r,
        "rho" => rho,
        "rho_string" => string(rho),
        "status" => string(status),
        "raw_status" => rawstatus,
        "solve_time" => stime,
        "total_time" => elapsed,
        "arithmetic" => string(ARITH),
        "precision_bits" => PRECISION_BITS,
        "eDIMACS" => EDIMACS,
        "u" => uT,
        "freeidx" => freeidx,
        "Delta" => Delta,
        "y_Delta" => yD,
        "moments_chebyshev_even" => moments,
        "gram_basis" => B,
        "cells" => cells,
        "gram_blocks" => Qfull,        # Dict cell => Matrix{BigFloat} (certificate)
        "dual_sign" => dual_sign,
        "q_multiplier_chebyshev" => q_cheb,
        "f_cofactor_h" => f_h,
        "certificate_residual" => residual,
        "pointwise_identity_error" => pterr,
        "block_sizes" => blocksizes,
        "f_mono" => f_mono,
        "permpart" => permpart,
        "timestamp" => string(Dates.now()),
    )
end

# =============================================================================
# main
# =============================================================================
function main()
    selftest_nf()
    G, permpart = detect_group(f_mono)
    println("|G| = $(length(G)), permutation part = $(permpart)")
    @assert all(p[E] == E for p in permpart) "permutation symmetries must fix x_E; change E"
    @assert (Expo(collect(1:N)), NTuple{N,Int}(ntuple(_ -> -1, N))) in G "f must be even in each variable"

    f_cheb = mono_to_cheb(f_mono)
    f_nf, f_h = nf_poly(f_cheb)
    @assert all(all(iseven, g) && g[E] == 0 for g in keys(f_nf))

    mkpath(OUTDIR)
    summary = joinpath(OUTDIR, "summary.txt")
    open(summary, "a") do io
        println(io, "# run $(Dates.now())  ARITH=$(ARITH) bits=$(PRECISION_BITS) " *
                    "eDIMACS=$(EDIMACS) perm_symmetry=$(USE_PERM_SYMMETRY)")
    end

    for r in RMIN:RMAX
        data = solve_order(r, f_nf, f_h, permpart)
        serialize(joinpath(OUTDIR, "order_$(r).jls"), data)
        open(summary, "a") do io
            @printf(io, "r=%2d  rho_r = %s\n", r, data["rho_string"])
            @printf(io, "      status=%s residual=%.3e pointwise=%.3e blocks=%s solve=%.1fs\n",
                    data["status"], Float64(data["certificate_residual"]),
                    Float64(data["pointwise_identity_error"]),
                    string(sort(data["block_sizes"], rev = true)), Float64(data["solve_time"]))
        end
    end
    println("done; results in $(OUTDIR)/")
end

main()
