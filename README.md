# Loraine.jl

Loraine is an implementation of interior point method algorithm for linear semidefinite optimization problems. 
The special feature Loraine is the iterative solver for linear system. This is to be used for problems with (very) low rank solution matrix.
Standard (non-low-rank) problems can be solved using the direct solver; then he user gets a standard IP method akin SDPT3.

At the moment, Loraine is a stand-alone code, reading data from a Matlab file - sample input files can be foound in directory "Examples".
To run the code, use the tvp.jl script in the "scripts" directory; in this script you can select one of the provided input files.

## Options

The list of options to be set/changed in tvp.jl

kit          # kit = 0 for direct solver; kit = 1 for CG
tol_cg       # tolerance for CG solver [1e-2]
tol_cg_up    # tolerance update [0.5]
tol_cg_min   # minimal tolerance for CG solver [1e-6]
eDIMACS      # epsilon for DIMACS error stopping criterion [1e-5]
erank        # estimated rank [1]
verb         # 2..full output; 1..short output; 0..no output
maxit        # maximal number of global iterations [200]
