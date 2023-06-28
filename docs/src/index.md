# Loraine.jl

Loraine.jl is a Julia implementation of an interior point method algorithm for linear semidefinite optimization problems. 
The special feature of Loraine is the iterative solver for linear systems. This is to be used for problems with (very) low rank solution matrix.
Standard (non-low-rank) problems and linear programs can be solved using the direct solver; then the user gets a standard IP method akin SDPT3.

