# Options

The list of Loraine options (default values are in the [bracket]):
```julia
kit             # kit = 0 for direct solver; kit = 1 for CG [0]
tol_cg          # initial tolerance for CG solver [1.0e-2]
tol_cg_up       # tolerance update [0.5]
tol_cg_min      # minimal tolerance for CG solver [1.0e-6]
eDIMACS         # epsilon for DIMACS error stopping criterion [1.0e-5]
preconditioner  # 0...no; 1...H_alpha; 2...H_beta; 4...hybrid [1]
erank           # estimated rank [1]
aamat           # 0..A^TA; 1..diag(A^TA); 2..identity [2]
verb            # 2..full output; 1..short output; 0..no output [1]
datarank        # 0..full rank matrices expected [0]
                # -1..rank-1 matrices expected, converted to vectors, if possible
                # (TBD) 1..vectors expected for low-rank data matrices
initpoint       # 0..Loraine heuristics, 1..SDPT3-like heuristics [0]
timing          # 1..yes, 0..no
maxit           # maximal number of global iterations [200]
```

## Some more details

- `eDIMACS` is checked against the maximum of DIMACS errors, measuring (weighted) primal and dual infeasibility, complementary slackness, duality gap.  
    - for the direct solver (`kit = 0`), value about `1e-7` should give a similar precision as default MOSEK
    - for the iterative solver (`kit = 1`), `eDIMACS` may need to be increased to `1e-6` or even `1e-5` to guarantee convergence of Loraine.
    - for the iterative solver (`kit = 1`), `tol_cg_min` should always be smaller than or equal to `eDIMACS`

- `preconditioner`
    - **per CG iteration**, 0 is faster (lower complexity) than 2 which is faster than 1
    - **as a preconditioner**, 1 is better than 2 is better than 0, in the sence of CG iterations needed to solve the linear system
    - some  SDP problems are "easy", meaning that CG always converges without preconditioner (i.e., `preconditioner = 0'), so it's always worth trying this option
    - hybrid (`preconditioner = 4`) starts with (cheaper) `H_beta` and once it gets into difficulties, switches to `H_alpha`

- `erank` (only used when `kit = 1` and `preconditioner > 0`)
    - if you are not sure what the actual rank of the solution is, **always choose** `erank = 1`; with inreasing value of `erank`, the complexity of the preconditioner grows and the whole code could be slower, despite neding fewer CG iterations
    - only if you are sure about the rank of the solution, set `erank` to this value (but you should always compare it to `erank = 1`)

- `datarank` (only used with the direct solver `kit = 0`)
    - choose `datarank = -1` if you know (or suspect) that all the data matrices ``A_i`` have rank one; in this case, the matrices will be factorized as ``A_i = b_i b_i^T`` and vectors ``b_i`` will be used when constructing the Schur complement matrix
    - if you are not sure about the rank of the data matrices, you can always try to set `datarank = -1`; if the factorization of any matrix fails, Lorain will switch to the default option `datarank = 0`
    - for rank-one data matrices, option `datarank = -1` will result in a much faster code than the default `datarank = 0`

- `timing` is not used when Loraine is called from JuMP

- `tol_cg, tol_cg_up, aamat`: it is not recommended to change values of these options, unless you really want to
