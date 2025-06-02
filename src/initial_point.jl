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

    b2 = 1 .+ abs.(cons_constant(solver.model)')
    n = length(b2)
    solver.y = zeros(n)
    
    f = zeros(n)
    for mat_idx in LRO.matrix_indices(solver.model)
        i = mat_idx.value
        dim = LRO.side_dimension(solver.model, mat_idx)
        if solver.initpoint == 0
            Eps = 1.0
        else
            f = norm(b2)/(1+norm_jac(solver.model, mat_idx))
            Eps = sqrt(dim) * max(1, sqrt.(dim) * f)
        end
        solver.X[i] = Eps * Matrix(1.0I, dim, dim)
        
        if solver.initpoint == 0
            Eta = n
        else
            mf = max(f, norm(objgrad(solver.model, mat_idx), 2))
            mf = (1 + mf) / dim
            Eta = sqrt(dim).* max(1, mf)
        end
        solver.S[i] = Eta * Matrix(1.0I, dim, dim)
    end
    
    p = zeros(n)
    pp = zeros(n)
    if solver.initpoint == 0
        Epss = 1.0
    else
        for con_idx in constraint_indices(solver.model)
            j = con_idx.value
            pp[j] = norm(jac(solver.model, con_idx, ScalarIndex))
            p[j] = b2[j] / (1 + pp[j])
        end
        Epss = max(1.0, maximum(p, init = 0.0))
    end
    solver.X_lin = 1 .* Epss * ones(LRO.num_scalars(solver.model))
    
    if solver.initpoint == 0
        Etaa = 1.0
    else
        mf = max(maximum(pp, init = 0.0), norm(objgrad(solver.model, ScalarIndex)))
        mf = (0 + mf) ./ sqrt(LRO.num_scalars(solver.model))
        Etaa =  max(1, mf)
    end
    solver.S_lin = 1 .* Etaa * ones(LRO.num_scalars(solver.model))
    solver.S_lin_inv = 1 ./ solver.S_lin
    
end
