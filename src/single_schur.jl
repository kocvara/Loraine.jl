# The Schur complement factored in single precision, as in brisk. The Float32
# factor preconditions conjugate gradient on the Float64 system. Far from the
# optimum, the Schur complement is well conditioned so a few steps give the
# accuracy needed by the Newton direction in about half the time of a Float64
# factorization. Once conjugate gradient needs more steps, the Float64
# factorization is used for the rest of the solve.
mutable struct SingleSchur
    work::Matrix{Float32}
    factor::Cholesky{Float32,Matrix{Float32}}
    # Whether the current factorization is in single precision
    active::Bool
    # Whether single precision was not accurate enough for a previous solve
    disabled::Bool
    # Conjugate gradient vectors
    r::Vector{Float64}
    z::Vector{Float64}
    p::Vector{Float64}
    q::Vector{Float64}
    z32::Vector{Float32}
end

function SingleSchur(m::Integer)
    work = zeros(Float32, m, m)
    return SingleSchur(
        work,
        Cholesky(work, 'U', 0),
        false,
        false,
        zeros(m),
        zeros(m),
        zeros(m),
        zeros(m),
        zeros(Float32, m),
    )
end

# Relative residual targeted with the single precision factor, as in brisk
const SINGLE_SCHUR_RTOL = 1e-10
# Single precision is disabled once a solve needs more steps than this
const SINGLE_SCHUR_MAXSTEPS = 4
# Number of steps after which the solve falls back to Float64
const SINGLE_SCHUR_MAXITER = 50

# Factor the upper triangle of `B` in single precision. Return `false` if it
# is disabled or if `B` is not numerically positive definite in single
# precision.
function factor_single!(s::SingleSchur, B::Matrix{Float64})
    s.active = false
    if s.disabled
        return false
    end
    for j in axes(B, 2), i in 1:j
        v = B[i, j]
        # Avoid subnormals in the factor
        s.work[i, j] = abs(v) < 1e-30 ? 0.0f0 : Float32(v)
    end
    factor = cholesky!(Hermitian(s.work, :U); check = false)
    if !issuccess(factor)
        return false
    end
    s.factor = factor
    s.active = true
    return true
end

function _precondition!(s::SingleSchur, z, r)
    s.z32 .= r
    ldiv!(s.factor, s.z32)
    z .= s.z32
    return z
end

# Solve `B x = b` by conjugate gradient preconditioned by the single precision
# factor of `B`. Return `false` if it does not reach the tolerance.
function solve_single!(s::SingleSchur, x, B::Matrix{Float64}, b)
    H = Hermitian(B, :U)
    tol = SINGLE_SCHUR_RTOL * norm(b)
    r, z, p, q = s.r, s.z, s.p, s.q
    _precondition!(s, x, b)
    copyto!(r, b)
    mul!(r, H, x, -1, 1)
    if norm(r) <= tol
        return true
    end
    _precondition!(s, z, r)
    copyto!(p, z)
    rz = dot(r, z)
    for k in 1:SINGLE_SCHUR_MAXITER
        mul!(q, H, p)
        α = rz / dot(p, q)
        axpy!(α, p, x)
        axpy!(-α, q, r)
        if norm(r) <= tol
            if k > SINGLE_SCHUR_MAXSTEPS
                s.disabled = true
            end
            return true
        end
        _precondition!(s, z, r)
        rz_prev = rz
        rz = dot(r, z)
        p .= z .+ (rz / rz_prev) .* p
    end
    s.disabled = true
    return false
end
