module LapackWorkspaces

using LinearAlgebra
using LinearAlgebra: BlasInt
using LinearAlgebra.BLAS: @blasfunc
import FastLapackInterface as FLI

export InverseWorkspace, make_workspace, block_inverse!

struct InverseWorkspace{T<:AbstractFloat}
    lu::FLI.LUWs
    inverse_work::Vector{T}
end

# getri! has no workspace overload in FastLapackInterface. This follows Julia's
# LAPACK wrapper, retaining its work vector instead of querying it on every call.
for (T, getri) in ((Float64, :dgetri_), (Float32, :sgetri_))
    @eval function cached_getri!(work, ipiv, A::Matrix{$T}, lwork)
        n = LinearAlgebra.checksquare(A)
        info = Ref{BlasInt}()
        ccall(
            (@blasfunc($getri), LinearAlgebra.libblastrampoline),
            Cvoid,
            (
                Ref{BlasInt},
                Ptr{$T},
                Ref{BlasInt},
                Ptr{BlasInt},
                Ptr{$T},
                Ref{BlasInt},
                Ref{BlasInt},
            ),
            n,
            A,
            max(1, stride(A, 2)),
            ipiv,
            work,
            lwork,
            info,
        )
        LAPACK.chklapackerror(info[])
        return A
    end
end

make_workspace(A::Matrix) = nothing
function make_workspace(A::Matrix{T}) where {T<:Union{Float32,Float64}}
    lu = FLI.LUWs(A)
    work = Vector{T}(undef, 1)
    cached_getri!(work, lu.ipiv, A, BlasInt(-1))
    resize!(work, Int(work[1]))
    return InverseWorkspace(lu, work)
end

function block_inverse!(ws::InverseWorkspace, A)
    _, pivots, info = LAPACK.getrf!(ws.lu, A; resize = false)
    info > 0 && throw(SingularException(info))
    return cached_getri!(ws.inverse_work, pivots, A, BlasInt(length(ws.inverse_work)))
end
block_inverse!(::Nothing, A) = LinearAlgebra.inv!(lu!(A))

end
