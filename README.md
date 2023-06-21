# Loraine.jl

Loraine.jl is a Julia implementation of an interior point method algorithm for linear semidefinite optimization problems. 
The special feature of Loraine is the iterative solver for linear systems. This is to be used for problems with (very) low rank solution matrix.
Standard (non-low-rank) problems and linear programs can be solved using the direct solver; then the user gets a standard IP method akin SDPT3.

There is also a Matlab version of the code here
```https://github.com/kocvara/Loraine.m```

## License and Original Contributors

Loraine was developed by Soodeh Habibi and Michal Kočvara, University of Birmingham, and Michael Stingl, University of Erlangen, for H2020 ITN POEMA and is distributed under the GNU General Public License 3.0. For commercial applications that may be incompatible with this license, please contact the authors to discuss alternatives. 

The JuMP interface was provided by Benoît Legat. His help is greatly ackowledged.

## Intallation 

] add https://github.com/kocvara/Loraine.jl

## Using with JuMP
```
using JuMP, Loraine
model = Model(Loraine.Optimizer)
set_attribute(model, "maxit", 100)

@variable(model,...)
@constraint(model, ...)
@variable(model, ...)
@objective(model, Max, ...)
optimize!(model)
```
To solve an SDP problem stored in SDPA format, do
```
using JuMP, Loraine
model = read_from_file("examples/data/theta1.dat-s")
set_optimizer(model, Loraine.Optimizer)
optimize!(model)
```
For more examples, see folder `examples` includes a few examples of how to use Loraine via JuMP; in particular, `solve_sdpa.jl` reads an SDP in the SDPA input format and solves it by Loraine. A few sample problems can be found in folder `examples/data`.

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

## Loraine paper
```
@article{loraine2023,
  title={Loraine-An interior-point solver for low-rank semidefinite programming},
  author={Habibi, Soodeh and Ko{\v{c}}vara, Michal and Stingl, Michael},
  www={https://hal.science/hal-04076509/}
  note={Preprint hal-04076509}
  year={2023}
}
```
