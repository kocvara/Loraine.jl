# using SuiteSparseGraphBLAS


function makeBBBB(n,nlmi,A,G)
    # BBBB = Matrix{Float64}(zeros, n, n)
    # @timeit to "BBBB" begin
    BBBB = zeros(Float64, n, n)
    for ilmi = 1:nlmi
        Gilmi = G[ilmi]
        Ailmi = A[ilmi,:]
        BBBB += makeBBBBi(Ailmi,Gilmi,n)
    end     
# end
    return BBBB
end
#####
function makeBBBBi(Ailmi,Gilmi,n)
    # BBBB = Matrix{Float64}(zeros, n, n)
    BBBB = zeros(Float64, n, n)
    BB = Matrix{Float64}(undef,size(Gilmi, 1)^2, n)
    # BB = zeros(Float64, size(Gilmi, 1)^2, n)
    gugu1 = Matrix{Float64}(undef,size(Gilmi, 1), size(Ailmi[1], 2))
    gugu = Matrix{Float64}(undef,size(Gilmi, 1), size(Gilmi, 1))
    @inbounds @fastmath for i = 1:n
        # @timeit to "BBBB1a" begin
            mul!(gugu1,Gilmi',transpose(Ailmi[i+1]))
        # end
        # @timeit to "BBBB1b" begin
            mul!(gugu,gugu1,Gilmi)
        # end
        # @timeit to "BBBB1c" begin
            BB[:, i] = vec(gugu)
        # end
        # BB[:, i] = vec((Gilmi' * A[ilmi, i+1]) * Gilmi)
    end
    # @timeit to "BBBB1d" begin
    mul!(BBBB,BB',BB)
    # end
    return BBBB 
end

###########################################################################
function makeBBBBalt(n,nlmi,A,AA,W,to)
    # @timeit to "BBBB" begin
    BBBB = zeros(Float64, n, n)
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
    Hnn = zeros(Float64, n, n)
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
    Hnn = Hermitian(Hnn)
    return Hnn
end

###########################################################################
function makeBBBBalt1(n,nlmi,A,AA,W)
    # @timeit to "BBBB" begin
    BBBB = zeros(Float64, n, n)
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
    Hnn = zeros(Float64, n, n)
    tmp1 = Matrix{Float64}(undef,size(Wilmi, 2), size(Ailmi[1], 1))
    # tmp1 = spzeros(size(Wilmi, 2), size(Ailmi[1], 1))
    tmp = Matrix{Float64}(undef,size(Wilmi, 1), size(Wilmi, 1))
    tmp2 = Matrix{Float64}(undef,size(AAilmi, 1), 1)

    # @timeit to "BBBB1a" begin
        @inbounds for i = 1:n
                Fkok = Wilmi*Ailmi[i+1][:]
                mul!(tmp2,AAilmi,Fkok)
                Hnn[:,i] .= -tmp2
        end
    # end
    Hnn = Hermitian(Hnn)
    return Hnn
end


###########################################################################
function makeBBBBsp(n,nlmi,A,myA,W)
    # @timeit to "BBBBsp" begin
    BBBB = zeros(Float64, n, n)
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
    BBBB = zeros(Float64, n, n)
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
            @inbounds for iAj = 1:length(iii_j)
                ttt1 = 0.0
                iiijAj = iii_j[iAj]
                jjjjAj = jjj_j[iAj]
                vvvj = -vvv_j[iAj]    
                @inbounds for iAi = 1:length(iii_i)
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
    BBBB = zeros(Float64, n, n)
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
    BBBB = zeros(Float64, n, n)
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
                @inbounds for iAj = 1:length(iii_j)
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
