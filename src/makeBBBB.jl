# using SuiteSparseGraphBLAS
function makeBBBB_rank1(n,nlmi,B,G,to)
    @timeit to "BBBB_rank1" begin
    tmp = zeros(Float64, n, n)
    BBBB = zeros(Float64, n, n)
    for ilmi = 1:nlmi
        @timeit to "BBBB_rank1_a" begin
            BB = transpose(B[ilmi] * G[ilmi])
        end
        @timeit to "BBBB_rank1_b" begin
            mul!(tmp,BB',BB)
            if ilmi == 1
                BBBB = tmp .^ 2
            else
                BBBB += tmp .^ 2
            end
        end
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
        BBBB += makeBBBBsi(ilmi,Ailmi,AAilmi,myA,Wilmi,n,to,qA,sigmaA)
    end

    return BBBB
end

#####
function makeBBBBsi(ilmi,Ailmi,AAilmi,myA,Wilmi,n,to,qA,sigmaA)
    BBBB = zeros(Float64x8, n, n)
    tmp1 = Matrix{Float64x8}(undef,size(Wilmi, 2), size(Ailmi[1], 1))
    # tmp = Matrix{Float64}(undef,size(Wilmi, 1), size(Wilmi, 1))
    tmp2 = Matrix{Float64x8}(undef,size(AAilmi, 1), 1)
    tmp3 = Vector{Float64x8}(undef,size(Wilmi, 1))
    ilmi1 = (ilmi-1)*n

    # @show qA
    # @show sigmaA[1:9,1]

    @inbounds for ii = 1:n
        # tmp1 = zeros(Float64,size(Wilmi, 2), size(Ailmi[1], 1))
        tmp  = zeros(Float64x8,size(Wilmi, 2), size(Ailmi[1], 1))
        i = sigmaA[ii,ilmi]
        # if ii <= qA[1]
        if 1==1
             # @show "one"
            # @show ii
            @timeit to "BBBBone" begin
                @timeit to "BBBBone1" begin
                    mul!(tmp1,Wilmi,Ailmi[i+1])
                end
                @timeit to "BBBBone2" begin
                    mul!(tmp,tmp1,Wilmi)
                end
                @timeit to "BBBBone3" begin
                    # tmp2 = AAilmi * vec(tmp)
                    mul!(tmp2,AAilmi,vec(tmp))
                end
                @timeit to "BBBBone4" begin
                    BBBB[i:end,i] .= -tmp2[i:end]
                end
            end
        elseif 1==0
            if ~isempty(myA[ilmi1+i].iind)
                @timeit to "BBBBone1a" begin
                    myAiii = myA[ilmi1+i]
                    iii = myAiii.iind
                    jjj = myAiii.jind
                    vvv = myAiii.nzval
                    for j in eachindex(iii)
                        @timeit to "BBBBone1aaa" begin
                            tmptmp = Wilmi[:,iii[j]] * vvv[j]
                        end
                        @timeit to "BBBBone1abb" begin
                            mul!(tmp1,tmptmp,(Wilmi[:,jjj[j]])')
                        end
                        @timeit to "BBBBone1acc" begin
                            tmp += tmp1
                        end
                    end
                end
                # @show norm(tmp - tmp1*Wilmi)
                @timeit to "BBBBone3a" begin
                    mul!(tmp2,AAilmi,vec(tmp))
                end
                @timeit to "BBBBone4a" begin
                    BBBB[i:end,i] .= tmp2[i:end]
                end
            end
        # elseif ii <= qA[2]
         elseif 1==0
            @timeit to "BBBBtwo" begin
            mul!(tmp1,Ailmi[i+1],Wilmi)
            @inbounds for jj = ii:n
                j = sigmaA[jj,ilmi]
                
                if ~isempty(myA[ilmi1+j].iind)               
                    row = Ailmi[j+1].rowval
                    # @timeit to "BBBBtwo_i_A" begin
                    myAjjj = myA[ilmi1+j]
                    colval = myAjjj.jind
                    rowval = myAjjj.iind
                    # end
                    # @show row
                    # @show colval
                    ttt = 0.0
                    # @timeit to "BBBBtwo_i_B" begin
                    for iAj = 1:length(row)
                        # @timeit to "BBBBtwo_i_C" begin
                        ttt1 = dot(Wilmi[:,rowval[iAj]],tmp1[colval[iAj],:])
                        # end
                        ttt += ttt1 * Ailmi[j+1].nzval[iAj]
                    end
                    # end

                    BBBB[i,j] = ttt
                    if !=(i,j)
                        BBBB[j,i] = ttt
                    end
                end

                # if ~isempty(myA[ilmi1+j].iind)
                #     myAjjj = myA[ilmi1+j]
                #     iii_j = myAjjj.iind
                #     jjj_j = myAjjj.jind
                #     vvv_j = myAjjj.nzval
                #     ttt = 0.0
                #     # @timeit to "BBBBtwo_i" begin
                #     @inbounds for iAj in eachindex(iii_j)
                #         # @timeit to "BBBBtwo_ii_A" begin
                #         iiijAj = iii_j[iAj]
                #         jjjjAj = jjj_j[iAj]
                #         # end
                #         # vvvj = -vvv_j[iAj]    
                #         # @show size(tmp1[1,:])
                #         # @show size(Wilmi[:,1]')
                #         # @timeit to "BBBBtwo_i_B" begin
                #         ttt1 = dot(tmp1[:,iiijAj],Wilmi[:,jjjjAj])
                #         # end
                #         # @timeit to "BBBBtwo_i_C" begin
                #         ttt -= ttt1 * vvv_j[iAj]
                #         # end
                #     end
                #     # end
                #     BBBB[i,j] = ttt
                #     if !=(i,j)
                #         BBBB[j,i] = ttt
                #     end
                end  
            end       
            # end
        else
            @timeit to "BBBBthree" begin
            # @show "three"
            # @show ii
            if ~isempty(myA[ilmi1+i].iind)
                myAiii = myA[ilmi1+i]
                iii_i = myAiii.iind
                jjj_i = myAiii.jind
                vvv_i = myAiii.nzval
                @inbounds for jj = ii:n
                    j = sigmaA[jj,ilmi]
                    if ~isempty(myA[ilmi1+j].iind)
                        myAjjj = myA[ilmi1+j]
                        iii_j = myAjjj.iind
                        jjj_j = myAjjj.jind
                        vvv_j = myAjjj.nzval
                        ttt = 0.0
                        # @timeit to "inner_a" begin
                        @inbounds for iAj in eachindex(iii_j)
                            ttt1 = 0.0
                            iiijAj = iii_j[iAj]
                            jjjjAj = jjj_j[iAj]
                            vvvj = vvv_j[iAj]    
                            # @timeit to "inner" begin
                            @inbounds for iAi in eachindex(iii_i)
                                # @timeit to "BBBBinner_a" begin
                                iiiiAi = iii_i[iAi]
                                jjjiAi = jjj_i[iAi]
                                vvvi = vvv_i[iAi]    
                                # end
                                # @timeit to "BBBBinner_b" begin
                                ttt1 += vvvi * Wilmi[iiiiAi,iiijAj] * Wilmi[jjjiAi,jjjjAj]
                                # ttt1 -= vvv_i[iAi] * Wilmi[iii_i[iAi],iiijAj] * Wilmi[jjj_i[iAi],jjjjAj]
                                # end
                            end
                            # mul!(tmp3,Ailmi[i+1], Wilmi[:,jjjjAj])
                            # ttt1 = dot(Wilmi[:,iiijAj],tmp3)
                            # end
                            ttt += ttt1 * vvvj
                        end
                        # end
                        # BBBB[i,j] = ttt
                        # if !=(i,j)
                        #     BBBB[j,i] = ttt
                        # end
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
    # BBBB = (BBBB + BBBB') ./ 2
    return BBBB
end

###########################################################################
function makeBBBB(n,nlmi,A,G)
    BBBB = zeros(Float64x8, n, n)
    @inbounds for ilmi = 1:nlmi
        m = size(G[ilmi],1)
        nn = Int(m*m)
        BB = zeros(Float64x8, nn, n)
        Gilmi = G[ilmi]
        for i = 1:n
            BB[:,i] = vec(Gilmi' * A[ilmi,i+1] * Gilmi);
        end
        BBBB += BB' * BB
    end
    # @show norm(BBBB-BBBB')
    return BBBB
end

###########################################################################
function makeBBBBalt(n,nlmi,A,AA,W,to)
    # @timeit to "BBBB" begin
    BBBB = zeros(Float64x8, n, n)
    @inbounds for ilmi = 1:nlmi
        Wilmi = W[ilmi]
        AAilmi = AA[ilmi]
        Ailmi = A[ilmi,:]
        BBBB += makeBBBBalti(Ailmi,AAilmi,Wilmi,n,to)
    end
# end
    return BBBB
end
#####
function makeBBBBalti(Ailmi,AAilmi,Wilmi,n,to)
    Hnn = zeros(Float64x8, n, n)
    tmp1 = Matrix{Float64}(undef,size(Wilmi, 2), size(Ailmi[1], 1))
    tmp = Matrix{Float64}(undef,size(Wilmi, 1), size(Wilmi, 1))
    tmp2 = Matrix{Float64}(undef,size(AAilmi, 1), 1)

    @timeit to "BBBB1a" begin
        @inbounds for i = 1:n
            # @timeit to "BBBB1a1" begin
                mul!(tmp1,Wilmi,transpose(Ailmi[i+1]))
            # end
            @timeit to "BBBB1a2" begin
                mul!(tmp,tmp1,Wilmi)
            end
            @timeit to "BBBB1a3" begin
                tmp2 .= AAilmi * vec(tmp)
            end
            @timeit to "BBBB1a4" begin    
                Hnn[:,i] .= -tmp2
            end
        end
    end
    # Hnn = Hermitian(Hnn)
    Hnn = (Hnn + Hnn') ./2
    return Hnn
end

###########################################################################
function makeBBBBalt1(n,nlmi,A,AA,W)
    # @timeit to "BBBB" begin
    BBBB = zeros(Float64x8, n, n)
    @inbounds for ilmi = 1:nlmi
        # @timeit to "BBBBkron" begin
        Wilmi = kron(W[ilmi],W[ilmi])
        # end
        AAilmi = AA[ilmi]
        Ailmi = A[ilmi,:]
        BBBB += makeBBBBalti1(Ailmi,AAilmi,Wilmi,n)
    end
# end
    return BBBB
end
#####
function makeBBBBalti1(Ailmi,AAilmi,Wilmi,n)
    Hnn = zeros(Float64x8, n, n)
    tmp1 = Matrix{Float64x8}(undef,size(Wilmi, 2), size(Ailmi[1], 1))
    # tmp1 = spzeros(size(Wilmi, 2), size(Ailmi[1], 1))
    tmp = Matrix{Float64x8}(undef,size(Wilmi, 1), size(Wilmi, 1))
    tmp2 = Matrix{Float64x8}(undef,size(AAilmi, 1), 1)

    # @timeit to "BBBB1a" begin
        @inbounds for i = 1:n
                Fkok = Wilmi*Ailmi[i+1][:]
                mul!(tmp2,AAilmi,Fkok)
                Hnn[:,i] .= -tmp2
        end
    # end
    # Hnn = Hermitian(Hnn)
    Hnn = (Hnn + Hnn') ./ 2
    return Hnn
end


###########################################################################
function makeBBBBsp(n,nlmi,A,myA,W)
    # @timeit to "BBBBsp" begin
    BBBB = zeros(Float64x8, n, n)
    for ilmi = 1:nlmi
        Wilmi = W[ilmi]
        Ailmi = A[ilmi,:]
        BBBB += makeBBBBspi(ilmi,Ailmi,myA,Wilmi,n)
    end   
    # end  
    return BBBB
end
#####
function makeBBBBspi(ilmi,Ailmi,myA,Wilmi,n)
    BBBB = zeros(Float64x8, n, n)
    ilmi1 = (ilmi-1)*n
    @inbounds for i = 1:n
        if ~isempty(myA[ilmi1+i].iind)
        # println(i)
        # Aiii = Ailmi[i+1]
        myAiii = myA[ilmi1+i]
        iii_i = myAiii.iind
        jjj_i = myAiii.jind
        vvv_i = myAiii.nzval
        @inbounds for j = i:n
            if ~isempty(myA[ilmi1+j].iind)
            # Ajjj = Ailmi[j+1]
            myAjjj = myA[ilmi1+j]
            iii_j = myAjjj.iind
            jjj_j = myAjjj.jind
            vvv_j = myAjjj.nzval
            ttt = 0.0
            @inbounds for iAj in eachindex(iii_j)
                ttt1 = 0.0
                iiijAj = iii_j[iAj]
                jjjjAj = jjj_j[iAj]
                vvvj = -vvv_j[iAj]    
                @inbounds for iAi in eachindex(iii_i)
                    # @timeit to "inner" begin
                        iiiiAi = iii_i[iAi]
                        jjjiAi = jjj_i[iAi]
                        vvvi = -vvv_i[iAi]    
                    ttt1 += vvvi * Wilmi[iiiiAi,iiijAj] * Wilmi[jjjiAi,jjjjAj]
                    # end
                end
                # @timeit to "outer" begin
                ttt += ttt1 * vvvj
                # end
            end
            BBBB[i,j] = ttt
            if !=(i,j)
                BBBB[j,i] = ttt
            end
        end
        end
    end
    end
    return BBBB
end

###########################################################################
function makeBBBBsp2(n,nlmi,A,myA,W)
    # @timeit to "BBBBsp" begin
    BBBB = zeros(Float64x8, n, n)
    for ilmi = 1:nlmi
        Wilmi = W[ilmi]
        Ailmi = A[ilmi,:]
        BBBB += makeBBBBsp2i(ilmi,Ailmi,myA,Wilmi,n)
    end   
    # end  
    return BBBB
end
#####
function makeBBBBsp2i(ilmi,Ailmi,myA,Wilmi,n)
    BBBB = zeros(Float64x8, n, n)
    @inbounds for i = 1:n
        F = Matrix{Float64}(undef,size(Wilmi, 2), size(Ailmi[1], 1))
        # @timeit to "BBBB1a1" begin
            mul!(F,Wilmi,transpose(Ailmi[i+1]))
        # end
        ilmi1 = (ilmi-1)*n
        @inbounds for j = i:n
            if ~isempty(myA[ilmi1+j].iind)
                # Ajjj = Ailmi[j+1]
                myAjjj = myA[ilmi1+j]
                iii_j = myAjjj.iind
                jjj_j = myAjjj.jind
                vvv_j = myAjjj.nzval
                ttt = 0.0
                @inbounds for iAj in eachindex(iii_j)
                    iiii = iii_j[iAj]
                    jjjj = jjj_j[iAj]
                    vvv = -vvv_j[iAj]
                    # @timeit to "outer" begin
                    ttt += vvv * dot(F[iiii,:], Wilmi[:,jjjj])
                    # end
                end
                BBBB[i,j] = ttt
                if !=(i,j)
                    BBBB[j,i] = ttt
                end
            end
        end
    end
    return BBBB
end


# function makeBBBBsp2i(Ailmi,iAilmi,jAilmi,Wilmi,n,nnzs_ilmi)
#     BBBB = zeros(Float64, n, n)
#     @inbounds for i = 1:n
#         F = Matrix{Float64}(undef,size(Wilmi, 2), size(Ailmi[1], 1))
#         @timeit to "BBBB1a1" begin
#             mul!(F,Wilmi,transpose(Ailmi[i+1]))
#         end
#         @inbounds for j = i:n
#             Ajjj = Ailmi[j+1]
#             rowval = Ajjj.rowval
#             nzval = Ajjj.nzval
#             tmp = 0.0
#             @timeit to "BBBB1a2" begin
#             for col = 1:Ajjj.n, k=Ajjj.colptr[col]:(Ajjj.colptr[col+1]-1)
#                 ki=rowval[k]
#                 kv=nzval[k]
#                 @timeit to "BBBB1a3" begin
#                 for multivec_row=1:size(Wilmi, 2)
#                     WF = dot(F[multivec_row,:], Wilmi[:,ki])
#                     tmp += kv * WF
#                 end
#             end
#             end
#         end
#             BBBB[i,j] = tmp
#             if !=(i,j)
#                 BBBB[j,i] = tmp
#             end
#         end
#     end
#     return BBBB
# end

function makeRHS(nlmi,AA,W,S,Rp,Rd)
    h = Rp  # RHS for the Hessian equation
    for i = 1:nlmi
        # h = h + AA[i] * my_kron(G[i], G[i], (G[i]' * Rd[i] * G[i] + diagm(D[i])))
        h = h + AA[i]*vec(W[i]*(Rd[i]+S[i])*W[i]);  #equivalent
    end
return h
end
