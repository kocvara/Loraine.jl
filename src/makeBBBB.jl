function makeBBBB_rank1(n,nlmi,B,G,to)
    @timeit to "BBBB_rank1" begin
    tmp = zeros(Float64, n, n)
    BBBB = zeros(Float64, n, n)
    for ilmi = 1:nlmi
        # @timeit to "BBBB_rank1_a" begin
            BB = transpose(B[ilmi] * G[ilmi])
        # end
        # @timeit to "BBBB_rank1_b" begin
            mul!(tmp,BB',BB)
            if ilmi == 1
                BBBB = tmp .^ 2
            else
                BBBB += tmp .^ 2
            end
        # end
    end     
    end
    return BBBB
end

#########################

function makeBBBBs(n,nlmi,A,AA,W,to,qA,sigmaA)
    BBBB = zeros(Float64, n, n)
    @inbounds for ilmi = 1:nlmi
        Wilmi = W[ilmi]
        AAilmi = AA[ilmi]
        Ailmi = A[ilmi,:]
        @timeit to "BBBBs" begin
        BBBB += makeBBBBsi(ilmi,Ailmi,AAilmi,Wilmi,n,to,qA,sigmaA)
        end
    end

    return BBBB
end

# Computes `⟨A * W, W * B⟩` for symmetric sparse matrices `A` and `B`
function _dot(A::SparseMatrixCSC, B::SparseMatrixCSC, W::Matrix)
    @assert LinearAlgebra.checksquare(W) == LinearAlgebra.checksquare(A) == LinearAlgebra.checksquare(B)
    # After these asserts, we know that `A`, `B` and `W` are square and
    # have the same sizes so we can safely use `@inbounds`
    result = zero(eltype(A))
    @inbounds for i in axes(A, 2)
        nzA = nzrange(A, i)
        if !isempty(nzA)
            for j in axes(B, 2)
                nzB = nzrange(B, j)
                if !isempty(nzB)
                    AW = zero(result)
                    for k in nzA
                        AW += nonzeros(A)[k] * W[rowvals(A)[k], j]
                    end
                    WB = zero(result)
                    for k in nzB
                        WB += W[i, rowvals(B)[k]] * nonzeros(B)[k]
                    end
                    result += AW * WB
                end
            end
        end
    end
    return result
end

#####
function makeBBBBsi(ilmi,Ailmi,AAilmi,Wilmi::Matrix{T},n,to,qA,sigmaA) where {T}
    BBBB = zeros(T, n, n)
    tmp1 = Matrix{T}(undef,size(Wilmi, 2), size(Ailmi[1], 1))
    # tmp = Matrix{Float64}(undef,size(Wilmi, 1), size(Wilmi, 1))
    tmp2 = Matrix{T}(undef,size(AAilmi, 1), 1)
    tmp3 = Vector{Float64}(undef,size(Wilmi, 1))
    ilmi1 = (ilmi-1)*n

    # @timeit to "BBBBsi" begin

    @inbounds for ii = 1:n
        # tmp1 = zeros(Float64,size(Wilmi, 2), size(Ailmi[1], 1))
        i = sigmaA[ii,ilmi]
        if nnz(Ailmi[i+1]) > 0
            if ii <= qA[1,ilmi]
                tmp  = zeros(T,size(Wilmi, 2), size(Ailmi[1], 1))
            # if 1==1
                # @show "one"
                # @show ii
                @timeit to "BBBBone" begin
                    @timeit to "BBBBone1" begin
                        mul!(tmp1,Wilmi,Ailmi[i+1])
                    end
                    @timeit to "BBBBone2" begin
                        # mul!(tmp,tmp1,Wilmi)
                        tmp = tmp1 * Wilmi
                    end
                    @timeit to "BBBBone3" begin
                        tmp2 = AAilmi * vec(tmp)
                        # mul!(tmp2,AAilmi,vec(tmp))
                    end
                    @timeit to "BBBBone4" begin
                        indi = sigmaA[ii:end,ilmi]
                        BBBB[indi,i] .= -tmp2[indi]
                        BBBB[i,indi] .= -tmp2[indi]
                        # @show BBBB[1:2,1:2]
                    end
                end
            # elseif ii <= qA[2,ilmi]
            elseif 1==0
            # @show "two"
                @timeit to "BBBBtwo" begin
                mul!(tmp1,Ailmi[i+1],Wilmi)
                @inbounds for jj = ii:n
                    j = sigmaA[jj,ilmi]
                    Ajjj = Ailmi[j+1]
                    if !iszero(nnz(Ajjj))
                        ttt = 0.0
                        # @timeit to "BBBBtwo_i" begin
                        @inbounds for jjjjAj in axes(Ajjj, 2)
                            for k in nzrange(Ajjj, jjjjAj)
                            # @timeit to "BBBBtwo_ii_A" begin
                                iiijAj = rowvals(Ajjj)[k]
                            # end
                            # vvvj = -vvv_j[iAj]    
                            # @timeit to "BBBBtwo_i_B" begin
                                ttt1 = dot(tmp1[:,iiijAj],Wilmi[:,jjjjAj])
                            # end
                            # @timeit to "BBBBtwo_i_C" begin
                                ttt += ttt1 * nonzeros(Ajjj)[k]
                            end
                            # end
                        end
                        # end
                        BBBB[i,j] = ttt
                        if !=(i,j)
                            BBBB[j,i] = ttt
                        end
                    end
                end  
                end       
                # end
            else
                @timeit to "BBBBthree" begin
                # @show "three"
                # @show ilmi
                # @show ii
                if !iszero(nnz(Ailmi[i+1]))
                    if nnz(Ailmi[i+1]) > 1
                        # @timeit to "BBBBthree>1" begin
                        # @show iii_i,myAiii.jind
                        # iii_is = iii_i[1:Int64(sqrt(length(iii_i)))]
                        # jjj_i = myAiii.jind
                        # vvv_i = myAiii.nzval
                        @inbounds for jj = ii:n
                            j = sigmaA[jj,ilmi]
                            if !iszero(nnz(Ailmi[j+1]))
                                # @timeit to "BBBBthree_1>1" begin
                                # iii_js = iii_j[1:Int64(sqrt(length(iii_j)))]
                                # jjj_j = myAjjj.jind
                                # vvv_j = myAjjj.nzval
                                # ttt = 0.0
                                # end
                                # @timeit to "BBBBthree_2>1" begin
                                ttt = _dot(Ailmi[i+1], Ailmi[j+1], Wilmi)
                                # end
                                # @inbounds for iAj in eachindex(iii_j)
                                #     ttt1 = 0.0
                                #     iiijAj = iii_j[iAj]
                                #     jjjjAj = jjj_j[iAj]
                                #     vvvj = vvv_j[iAj]    
                                #     @inbounds for iAi in eachindex(iii_i)
                                #         iiiiAi = iii_i[iAi]
                                #         jjjiAi = jjj_i[iAi]
                                #         vvvi = vvv_i[iAi]    
                                #         ttt1 += vvvi * Wilmi[iiiiAi,iiijAj] * Wilmi[jjjiAi,jjjjAj]
                                #         # ttt1 -= vvv_i[iAi] * Wilmi[iii_i[iAi],iiijAj] * Wilmi[jjj_i[iAi],jjjjAj]
                                #     end
                                #     ttt += ttt1 * vvvj
                                # end
                                # @timeit to "BBBBthree_3>1" begin
                                if i >= j
                                    BBBB[i,j] = ttt
                                else
                                    BBBB[j,i] = ttt
                                end
                            # end
                            end  
                        # end    
                        end   
                    else
                        @timeit to "BBBBthree=1" begin
                        # A is symmetric
                        iiiiAi = jjjiAi = only(rowvals(Ailmi[i+1]))
                        vvvi = only(nonzeros(Ailmi[i+1]))
                        @inbounds for jj = ii:n
                            j = sigmaA[jj,ilmi]
                            Ajjj = Ailmi[j+1]
                            # As we sort the matrices in decreasing `nnz` order,
                            # the rest of matrices is either zero or have only
                            # one entry
                            if !iszero(nnz(Ajjj))
                                iiijAj = jjjjAj = only(rowvals(Ajjj))
                                vvvj = only(nonzeros(Ajjj))
                                ttt = vvvi * Wilmi[iiiiAi,iiijAj] * Wilmi[jjjiAi,jjjjAj] * vvvj
                                if i >= j
                                    BBBB[i,j] = ttt
                                else
                                    BBBB[j,i] = ttt
                                end
                            end
                        end  
                        end 
                    end   
                end
                end
            end
        end
    end
# end
    return BBBB
end


function makeRHS(nlmi,AA,W,S,Rp,Rd)
    h = Rp  # RHS for the Hessian equation
    for i = 1:nlmi
        # h = h + AA[i] * my_kron(G[i], G[i], (G[i]' * Rd[i] * G[i] + diagm(D[i])))
        h = h + AA[i]*vec(W[i]*(Rd[i]+S[i])*W[i]);  #equivalent
    end
return h
end
