# Loraine.jl

*Sweet Lor(r)aine, let the party carry on[^1]...*

[^1]: https://www.youtube.com/watch?v=0D2wNf1lVrI

Loraine.jl is a Julia implementation of an interior point method algorithm for
linear semidefinite optimization problems. 

The special feature of Loraine is the iterative solver for linear systems. This
is to be used for problems with (very) low rank solution matrix.

Standard (non-low-rank) problems and linear programs can be solved using the
direct solver; then the user gets a standard IP method akin SDPT3.

There is also a MATLAB version of the code at [kocvara/Loraine.m](https://github.com/kocvara/Loraine.m).

## License and Original Contributors

Loraine is licensed under the [MIT License](https://github.com/kocvara/Loraine.jl/blob/main/LICENSE.md).

Loraine was developed by Soodeh Habibi and Michal Kočvara, University of
Birmingham, and Michael Stingl, University of Erlangen, for H2020 ITN POEMA. 

The JuMP interface was provided by Benoît Legat. His help is greatly
acknowledged.

## Installation 

Install `Loraine` using `Pkg.add`:
```julia
import Pkg
Pkg.add("Loraine")
```

## Use with JuMP

To use Loraine with JuMP, use `Loraine.Optimizer`:
```julia
using JuMP, Loraine
model = Model(Loraine.Optimizer)
set_attribute(model, "maxit", 100)
```

To solve an SDP problem stored in SDPA format, do
```julia
using JuMP, Loraine
model = read_from_file("examples/data/theta1.dat-s")
set_optimizer(model, Loraine.Optimizer)
optimize!(model)
```

For more examples, the folder [`examples`](https://github.com/kocvara/Loraine.jl/tree/main/examples)
includes a few examples of how to use Loraine via JuMP; in particular,
`solve_sdpa.jl` reads an SDP in the SDPA input format and solves it by Loraine.
A few sample problems can be found in folder `examples/data`.

## Rank-one data
If the solution does not have low rank, it is recommended to use a direct 
solver `kit = 0`. However, if you know that your data matrices are all rank-one, 
use the option `datarank = -1` to get a significant reduction in the complexity 
(and CPU time). Examples of such problems are `maxG11` and `thetaG11` from the 
`SDPLIB` collection.

## Documentation
[Loraine documentation](https://kocvara.github.io/Loraine.jl/)

## Options

The list of options:
```julia
kit             # kit = 0 for direct solver; kit = 1 for CG [0]
tol_cg          # tolerance for CG solver [1.0e-2]
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

## Citing

If you find Loraine useful, please cite the following paper:
```bibtex
@article{loraine2023,
  title={Loraine-An interior-point solver for low-rank semidefinite programming},
  author={Habibi, Soodeh and Ko{\v{c}}vara, Michal and Stingl, Michael},
  www={https://hal.science/hal-04076509/}
  note={Preprint hal-04076509}
  year={2023}
}
```
