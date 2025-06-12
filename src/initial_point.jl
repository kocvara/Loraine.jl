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

    b2 = 1 .+ abs.(LRO.cons_constant(solver.model)')
    n = length(b2)
    solver.y .= 0
    
    f = zeros(n)
    for mat_idx in LRO.matrix_indices(solver.model)
        i = mat_idx.value
        dim = LRO.side_dimension(solver.model, mat_idx)
        if solver.initpoint == 0
            Eps = 1.0
        else
            f = norm(b2)/(1+LRO.norm_jac(solver.model, mat_idx))
            Eps = sqrt(dim) * max(1, sqrt.(dim) * f)
        end
        solver.X[mat_idx] .= Eps * Matrix(1.0I, dim, dim)
        
        if solver.initpoint == 0
            Eta = n
        else
            mf = max(f, norm(NLPModels.grad(solver.model, mat_idx), 2))
            mf = (1 + mf) / dim
            Eta = sqrt(dim).* max(1, mf)
        end
        solver.S[mat_idx] .= Eta * Matrix(1.0I, dim, dim)
    end
    
    if solver.initpoint == 0
        pp = zeros(n)
        p = zeros(n)
    else
        pp = [norm(NLPModels.jac(solver.model, j, LRO.ScalarIndex)) for j in 1:n]
        p = b2 ./ (1 .+ pp)
    end
    Epss = max(1.0, maximum(p, init = 0.0))
    solver.X[LRO.ScalarIndex] .= Epss
    
    if solver.initpoint == 0
        Etaa = 1.0
    else
        mf = max(maximum(pp, init = 0.0), norm(NLPModels.grad(solver.model, LRO.ScalarIndex)))
        mf = (0 + mf) ./ sqrt(LRO.num_scalars(solver.model))
        Etaa =  max(1, mf)
    end
    solver.S[LRO.ScalarIndex] .= Etaa
    solver.S_lin_inv = inv.(solver.S[LRO.ScalarIndex])
    
end
