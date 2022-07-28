module Precon

# export MyA

# using SparseArrays
# using LinearAlgebra


# struct MyA
#     W
#     AA
#     nlin
#     C_lin
#     X_lin
#     S_lin_inv
# end

# function (t::MyA)(Ax::Vector{Float64}, x::Vector{Float64})

#     nlmi = size(t.AA,2) 
#     m = size(t.AA[1],1);
#     ax = zeros(size(t.AA[1],2))
#     # Ax = zeros(m);
#     # for ilmi = 1:nlmi
#         mul!(ax, transpose(t.AA[1]), x)
#         waxw = vec(t.W[1] * mat(ax) * t.W[1])
#         mul!(Ax,t.AA[1],waxw)
#     # end
#     # if t.nlin>0
#     #     Ax = Ax + t.C_lin * ((t.X_lin .* t.S_lin_inv) .* (t.C_lin' * x))
#     # end

# end

end #module