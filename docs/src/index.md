# Loraine.jl

Loraine.jl is a Julia implementation of an interior point method algorithm for linear semidefinite optimization problems. 

Special features of Loraine:

- Use of an iterative solver for linear systems. This is to be used for problems with (very) low rank solution matrix. Standard (non-low-rank) problems and linear programs can be solved using the direct solver; then the user gets a standard IP method akin SDPT3.
- Use of high-precision arithmetic (by means of MultiFloats.jl). Only to be used with a direct solver (and relatively small problems). *(New in version 0.2.0.)*




## Installation 

Install `Loraine` using `Pkg.add`:
```julia
import Pkg
Pkg.add("Loraine")
```

## Use with JuMP

To use Loraine with JuMP, use `Loraine.Optimizer` (for a standard double-precision solver) or `Loraine.Optimizer{Float64xN}`, with N = 2,...,8 ; for instance:
```julia
using JuMP, Loraine
model = Model(Loraine.Optimizer)
set_attribute(model, "maxit", 100)
```
or, for high-precision arithmetics,
```julia
using JuMP, Loraine
using MultiFloats
model = Model(Loraine.Optimizer{Float64x2})
```
or, for high-precision arithmetics with high-precision input,
```julia
using JuMP, Loraine
using MultiFloats
model = JuMP.GenericModel{Float64x2}(Loraine.Optimizer{Float64x2})
```
To solve an SDP problem stored in SDPA format, do
```julia
using JuMP, Loraine
# using MultiFloats
model = read_from_file("examples/data/theta1.dat-s")
set_optimizer(model, Loraine.Optimizer)
# set_optimizer(model, Loraine.Optimizer{Float64x8})
optimize!(model)
```

## License and Original Contributors

Loraine is licensed under the [MIT License](https://github.com/kocvara/Loraine.jl/blob/main/LICENSE.md).

Loraine was developed by Soodeh Habibi and Michal Kočvara, University of
Birmingham, and Michael Stingl, University of Erlangen, for H2020 ITN POEMA. 

The JuMP interface was provided by Benoît Legat. His help is greatly
acknowledged.

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