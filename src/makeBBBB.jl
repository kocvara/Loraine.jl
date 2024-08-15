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

function makeBBBBs(n,nlmi,A,AA,myA,W,to,qA,sigmaA)
    BBBB = zeros(Float64, n, n)
    @inbounds for ilmi = 1:nlmi
        Wilmi = W[ilmi]
        AAilmi = AA[ilmi]
        Ailmi = A[ilmi,:]
        @timeit to "BBBBs" begin
        BBBB += makeBBBBsi(ilmi,Ailmi,AAilmi,myA,Wilmi,n,to,qA,sigmaA)
        end
    end

    return BBBB
end

#####
function makeBBBBsi(ilmi,Ailmi,AAilmi,myA,Wilmi::Matrix{T},n,to,qA,sigmaA) where {T}
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
                    if ~isempty(myA[ilmi1+j].iind)
                        myAjjj = myA[ilmi1+j]
                        iii_j = myAjjj.iind
                        jjj_j = myAjjj.jind
                        vvv_j = myAjjj.nzval
                        ttt = 0.0
                        # @timeit to "BBBBtwo_i" begin
                        @inbounds for iAj in eachindex(iii_j)
                            # @timeit to "BBBBtwo_ii_A" begin
                            iiijAj = iii_j[iAj]
                            jjjjAj = jjj_j[iAj]
                            # end
                            # vvvj = -vvv_j[iAj]    
                            # @timeit to "BBBBtwo_i_B" begin
                            ttt1 = dot(tmp1[:,iiijAj],Wilmi[:,jjjjAj])
                            # end
                            # @timeit to "BBBBtwo_i_C" begin
                            ttt -= ttt1 * vvv_j[iAj]
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
                if ~isempty(myA[ilmi1+i].iind)
                    myAiii = myA[ilmi1+i]
                    if size(myAiii.iind,1) > 1
                        # @timeit to "BBBBthree>1" begin
                        iii_is = union(myAiii.iind)
                        # @show iii_i,myAiii.jind
                        # iii_is = iii_i[1:Int64(sqrt(length(iii_i)))]
                        # jjj_i = myAiii.jind
                        # vvv_i = myAiii.nzval
                        mya1 = Ailmi[i+1][iii_is,iii_is]
                        @inbounds for jj = ii:n
                            j = sigmaA[jj,ilmi]
                            if ~isempty(myA[ilmi1+j].iind)
                                # @timeit to "BBBBthree_1>1" begin
                                myAjjj = myA[ilmi1+j]
                                iii_js = union(myAjjj.iind)
                                # iii_js = iii_j[1:Int64(sqrt(length(iii_j)))]
                                # jjj_j = myAjjj.jind
                                # vvv_j = myAjjj.nzval
                                # ttt = 0.0
                                myw = Wilmi[iii_is,iii_js]                            
                                mya2 = Ailmi[j+1][iii_js,iii_js]
                                # end
                                # @timeit to "BBBBthree_2>1" begin
                                ttt = transpose(vec(transpose(mya1 * myw))) * vec(mya2 * transpose(myw))
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
                        iiiiAi = myAiii.iind[1]
                        jjjiAi = myAiii.jind[1]
                        vvvi = myAiii.nzval[1]
                        @inbounds for jj = ii:n
                            j = sigmaA[jj,ilmi]
                            if ~isempty(myA[ilmi1+j].iind)
                                myAjjj = myA[ilmi1+j]
                                iiijAj = myAjjj.iind[1]
                                jjjjAj = myAjjj.jind[1]
                                vvvj = myAjjj.nzval[1]
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
