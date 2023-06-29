# Loraine.jl

Loraine.jl is a Julia implementation of an interior point method algorithm for linear semidefinite optimization problems. 

The special feature of Loraine is the iterative solver for linear systems. This is to be used for problems with (very) low rank solution matrix.

Standard (non-low-rank) problems and linear programs can be solved using the direct solver; then the user gets a standard IP method akin SDPT3.

## Installation 

Install `Loraine` using `Pkg.add`:
```julia
import Pkg
Pkg.add("Loraine")
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