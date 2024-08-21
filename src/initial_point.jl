function initial_point(solver)

   find_initial!(solver)

    solver.sigma = 3

    # These two parameters may influence convergence:
    solver.tau = 0.95  #lower value; such as 0.9 leads to more iterations but more robust algo
    solver.expon = 3.0

    solver.DIMACS_error = 1.0
    solver.iter = 0
    solver.status = 0 

end

function  find_initial!(solver)

    C_lin = solver.model.C_lin'
    
    n = length(solver.model.b)
    solver.y = zeros(n,1)
    
    b2 = 1 .+ abs.(solver.model.b')
    f = zeros(1,n)
    for i=1:solver.model.nlmi
        if solver.initpoint == 0
            Eps = 1.0
        else
            f = norm(b2)/(1+norm(solver.model.AA[i]))
            Eps = sqrt.(solver.model.msizes[i]).* max(1,sqrt.(solver.model.msizes[i]).* f)
        end
        solver.X[i] = Eps * Matrix(1.0I, Int64(solver.model.msizes[i]), Int64(solver.model.msizes[i]))
        
        if solver.initpoint == 0
            Eta = solver.model.n
        else
            mf = max(f,norm(solver.model.C[i],2))
            mf = (1 + mf)./ sqrt(solver.model.msizes[i])
            Eta = sqrt(solver.model.msizes[i]).* max(1,mf)
        end
        solver.S[i] = Eta * Matrix(1.0I, Int64(solver.model.msizes[i]), Int64(solver.model.msizes[i]))
    end
    
    p = zeros(1,n)
    pp = zeros(1,n)
    dd = size(solver.model.d_lin,1)
    if solver.model.nlin>0
        if solver.initpoint == 0
            Epss = 1.0
        else
            for j=1:n
                normClin = 1+norm(solver.model.C_lin[j,:])
                p[j] = b2[j] ./ normClin;
            end
            Epss = max(1, maximum(p))
        end
        solver.X_lin = 1 .* Epss * ones(dd,1)
        
        if solver.initpoint == 0
            Etaa = 1.0
        else
            for j=1:n
                pp[j]=norm(solver.model.C_lin[j,:])
            end
            mf = max(maximum(pp),norm(solver.model.d_lin))
            mf = (0 + mf) ./ sqrt(dd)
            Etaa =  max(1,mf)
        end
        solver.S_lin = 1 .* Etaa * ones(dd,1)
        solver.S_lin_inv = 1 ./ solver.S_lin
    else
        solver.X_lin = Float64[]; solver.S_lin = Float64[]
    end
    if solver.model.nlin==0
        solver.X_lin=Float64[]
        solver.S_lin=Float64[]
        solver.S_lin_inv=Float64[]
    end
    
end
