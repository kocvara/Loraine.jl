# Loraine.jl

Loraine is a Julia implementation of an interior point method algorithm for linear semidefinite optimization problems. 
The special feature of Loraine is the iterative solver for linear systems. This is to be used for problems with (very) low rank solution matrix.
Standard (non-low-rank) problems can be solved using the direct solver; then the user gets a standard IP method akin SDPT3.

Loraine is to be used with JuMP. Folder `examples` includes a few examples of how to use Loraine via JuMP; in particular, `solve_sdpa.jl` reads an SDP in the SDPA input format and solves it by Loraine. A few sample problems can be found in folder `examples/data`.

To install Loraine, run 

] add https://github.com/kocvara/Loraine/

## Options

The list of options:
```
kit             # kit = 0 for direct solver; kit = 1 for CG [0]
tol_cg          # tolerance for CG solver [1.0e-2]
tol_cg_up       # tolerance update [0.5]
tol_cg_min      # minimal tolerance for CG solver [1.0e-6]
eDIMACS         # epsilon for DIMACS error stopping criterion [1.0e-5]
preconditioner  # 0...no; 1...H_alpha; 2...H_beta; 4...hybrid [1]
erank           # estimated rank [1]
aamat           # 0..A^TA; 1..diag(A^TA); 2..identity [2]
verb            # 2..full output; 1..short output; 0..no output [1]
initpoint       # 0..Loraine heuristics, 1..SDPT3-like heuristics [0]
timing          # 1..yes, 0..no
maxit           # maximal number of global iterations [200]
```