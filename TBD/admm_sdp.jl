
using LinearAlgebra
using SparseArrays
using Printf

function admm_sdp(sdpdata, maxiter=10000)
    # ADMM_SDP - Solves general linear SDP problems by the ADMM method as 
    # described in Z. Wen, D. Goldfarb, and W. Yin. "Alternating direction 
    # augmented Lagrangian methods for semidefinite programming." Mathematical
    # Programming Computation 2.3-4 (2010): 203-230.
    #
    # Input file: Julia dictionary in POEMA sparse format
    #
    # Elements of the dictionary "sdpdata"
    #   name ... filename of the input file
    #   type ... type of the problem (not used)
    #   nvar ... number of primal variables
    #   nlmi ... number of linear matrix inequalities (or diagonal blocks of the
    #            matrix constraint)
    #   c ...... dim (Nx,1), coefficients of the linear objective function
    #   msizes . vector of sizes of matrix constraints (diagonal blocks)
    #   A ...... dictionary of A[k,l] for k=1,...,Na matrix constraint
    #            for l=1 ~ absolute term, l=2..Nx+1 coeficient matrices
    #            (some of them might be empty)
    #
    # Copyright (c) 2019 Michal Kocvara, m.kocvara@bham.ac.uk
    # Last Modified: 9 Dec 2022
    # Translated to Julia: April 18, 2025

    
    # Setting parameters
    eps = 1e-6  # final precision required
    mu = 10.01  # Augmented Lagrangian penalty parameter
    rho = (1 + sqrt(5))/2 - 0.5  # step-length for the multiplier X
    
    # Parameters for update of mu
    gamma = 0.5
    mu_min = 1e-4
    mu_max = 1e4
    eta1 = 10000
    eta2 = 100
    h4 = 100

    # Read SDP input data
    n = Int64(sdpdata["nvar"])
    ncon = Int64(sdpdata["nlmi"])
    b = vec(sdpdata["c"])
    nlin = Int64(sdpdata["nlin"])

    if haskey(sdpdata, "lsi_op")
        key_lsi = 1
    else
        key_lsi = 0
    end

    sumlin = 0
    C_lin = nothing
    d_lin = nothing
    
    if nlin > 0
        if key_lsi == 1
            sumlin = sum(sdpdata["lsi_op"])
            eqc = nlin - sumlin
            if sumlin < nlin
                d_lin = zeros(nlin)
                C_lin = zeros(size(sdpdata["C"], 1), nlin)
                ieq = 1
                indieq = Int[]
                
                for i in 1:nlin
                    if sdpdata["lsi_op"][i] == 0
                        C_lin[:, i] = -sdpdata["C"][:, i]
                        d_lin[i] = sdpdata["d"][i]
                        push!(indieq, i)
                        ieq += 1
                    else
                        C_lin[:, i] = -sdpdata["C"][:, i]
                        d_lin[i] = sdpdata["d"][i]
                    end
                end
                
                # Extend C_lin and d_lin
                C_lin = hcat(C_lin, sdpdata["C"][:, indieq])
                d_lin = vcat(d_lin, -sdpdata["d"][indieq])
            else
                d_lin = -sdpdata["d"]
                C_lin = -sdpdata["C"]
            end
        else
            d_lin = -sdpdata["d"]
            C_lin = -sdpdata["C"]
        end
    end

    if key_lsi == 1 && nlin > 0
        nlin = nlin + eqc
    end

    C = Dict()
    m = zeros(Int, ncon)
    for icon in 1:ncon
        C[icon] = -sdpdata["A"][icon, 1]
        m[icon] = size(C[icon], 1)
    end

    AAT = spzeros(n, n)
    aaa = Dict()
    A = Dict()
    
    for icon in 1:ncon
        aaa[icon] = []
        for i in 1:n
            A[(icon, i)] = sdpdata["A"][icon, i+1]
            aaa[icon] = [aaa[icon]; vec(A[(icon, i)])]
        end
        AAT += aaa[icon]' * aaa[icon]
    end
    
    if nlin > 0
        C_lin = sparse(C_lin)
        AAT += sparse(C_lin * C_lin')
    end
    
    # In Julia, we can use cholesky directly, which returns both the factorization and flag
    chol_result = cholesky(AAT, check=false)
    flag = issuccess(chol_result)
    if flag
        Rchol = chol_result.U
        Pchol = chol_result.P
    else
        # If Cholesky fails, use a more robust alternative
        # For example, using LU factorization or adding a small diagonal perturbation
        AAT += 1e-10 * I  # Add small regularization
        chol_result = cholesky(AAT)
        Rchol = chol_result.U
        Pchol = chol_result.P
    end

    println("================================= A D M M  for  S D P ===============================")
    println("Number of LMI Constraints: $(ncon)")
    println("Number of Variables: $(n)")
    println("Maximal Constraint Size: $(maximum(m))")
    println("Problem Name: \"$(sdpdata["name"])\"")

    # Initializing variables
    y = ones(n)
    S = Dict()
    X = Dict()
    
    for icon in 1:ncon
        S[icon] = sparse(I, m[icon], m[icon])
        X[icon] = sparse(I, m[icon], m[icon])
    end
    
    S_lin = nothing
    X_lin = nothing
    if nlin > 0
        S_lin = ones(nlin)
        X_lin = ones(nlin)
    end

    # Setting parameters
    println("-------------------------------------------------------------------------------------")
    println(" iter    p-infeas     d-infeas       d-gap           mu         error      objective")
    println("-------------------------------------------------------------------------------------")

    count = 1
    err = 1.0
    it_pinf = 0
    it_dinf = 0

    start_time = time()
    
    # ADMM main loop
    while err > eps
        # Update y
        Axb = spzeros(n)
        ASC = spzeros(n)
        
        for icon in 1:ncon
            Axb += aaa[icon]' * vec(X[icon])
            ASC += aaa[icon]' * (vec(S[icon]) - vec(C[icon]))
        end
        
        if nlin > 0
            Axb += C_lin * X_lin
            ASC += C_lin * (S_lin - d_lin)
        end
        
        # Using the Cholesky factorization for solving the system
        rhs = mu .* (Axb - b) + ASC
        y = -permute(Rchol \ (Rchol' \ (Pchol' * rhs)), Pchol)
        
        # Update S
        Vp = Dict()
        V = Dict()
        
        for icon in 1:ncon
            Vp[icon] = C[icon] - reshape(aaa[icon] * y, m[icon], m[icon])
            V[icon] = Vp[icon] - mu .* X[icon]
            
            eigen_decomp = eigen(Matrix(V[icon]))
            peval = max.(0, eigen_decomp.values)
            S[icon] = eigen_decomp.vectors * Diagonal(peval) * eigen_decomp.vectors'
            S[icon] = sparse(0.5 .* (S[icon] + S[icon]'))
        end
        
        Vp_lin = nothing
        V_lin = nothing
        if nlin > 0
            Vp_lin = d_lin - C_lin' * y
            V_lin = Vp_lin - mu .* X_lin
            S_lin = max.(V_lin, 0)
        end
        
        # Update X
        for icon in 1:ncon
            Xp = (1/mu) .* (S[icon] - V[icon])
            X[icon] = (1-rho) .* X[icon] + rho .* Xp
        end
        
        if nlin > 0
            Xp_lin = (1/mu) .* (S_lin - V_lin)
            X_lin = (1-rho) .* X_lin + rho .* Xp_lin
        end
        
        # Calculate current error
        dinf = 0.0
        dinfs = 0.0
        dgap = 0.0
        dgaps = 0.0
        pinf = norm(Axb - b)
        pinfs = pinf / (1 + norm(b))
        
        for icon in 1:ncon
            VS = vec(Vp[icon]) - vec(S[icon])
            dinfi = sqrt(dot(VS, VS))
            dinfsi = dinfi / (1 + norm(C[icon], 1))
            dinf += dinfi
            dinfs += dinfsi
            
            dgapi = dot(vec(C[icon]), vec(X[icon]))
            dgapsi = abs(dgapi)
            dgap += dgapi
            dgaps += dgapsi
        end
        
        if nlin > 0
            dinfi = sqrt(dot(Vp_lin - S_lin, Vp_lin - S_lin))
            dinf += dinfi
            dinfs += dinfi / (1 + norm(d_lin, 1))
            
            dgapi = dot(d_lin, X_lin)
            dgapsi = abs(dgapi)
            dgap += dgapi
            dgaps += dgapsi
        end

        dgap = abs(dot(b, y) - dgap)
        dgaps = dgap / (1 + abs(dot(b, y)) + dgaps)
        
        err = max(pinfs, max(dinfs, dgaps))
        count += 1
        
        # Update penalty parameter mu
        if pinf + dinf > 2
            if pinf / dinf < eta1
                it_pinf += 1
                it_dinf = 0
                if it_pinf > h4
                    mu = max(gamma * mu, mu_min)
                    it_pinf = 0
                end
            elseif pinf / dinf > eta2
                it_dinf += 1
                it_pinf = 0  # Fixed typo from the original code: it_finf -> it_pinf
                if it_dinf > h4
                    mu = min(mu / gamma, mu_max)
                    it_dinf = 0
                end
            end
        end
        
        if count % 100 == 0
            @printf("%5d   %.8f   %.8f   %.8f   %.8f   %.6f   %.8f\n", 
                    count, pinfs, dinfs, dgaps, mu, err, dot(b, y))
        end
        
        if count > maxiter
            break
        end
    end
    
    elapsed_time = time() - start_time

    y = -y
    
    # Save results to JLD2 file (Julia's equivalent to MATLAB's save)
    # using JLD2
    # @save "admm_sol.jld2" y X S
    
    # Alternative: just return the results
    @printf("%5d   %.8f   %.8f   %.8f   %.8f   %.6f   %.8f\n", 
            count, pinfs, dinfs, dgaps, mu, err, dot(b, y))
    println("-------------------------------------------------------------------------------------")
    
    lambda_min = minimum(eigen(Matrix(X[1])).values)
    min_el = minimum(X[1])
    @printf("Minimal eigenvalue of X: %4.2e; Minimal element of X:  %4.2e\n", lambda_min, min_el)
    println("-------------------------------------------------------------------------------------")
    @printf("Total ADMM iterations: %3d; Final precision: %4.2e; CPU Time %2.2fsec\n", 
            count, err, elapsed_time)
    println("=====================================================================================")
    
    return y, X, S
end