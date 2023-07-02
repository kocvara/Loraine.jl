# Loraine - LOw-RAnk INtErior point method
	
Primal-dual predictor-corrector interior-point method together with (optional) iterative solution of the resulting linear systems.

The iterative solver is a preconditioned Krylov-type method with a preconditioner utilizing low rank of the solution.

Loraine is a general purpose SDP solver, particularly efficient for SDP problems with low-rank data and/or very-low-rank solutions.


## Linear SDP problem
Let's first fix the notation and the problem Lorain atempts to solve.
Consider the primal and the dual linear SDP problem with explicit linear constraints in variables 

```math
\text{primal:}\ X \in \mathbb{S}^m,\ x_{\text{lin}}\in \mathbb{R}^p,\qquad 
\text{dual:}\ y \in \mathbb{R}^n,\ S \in \mathbb{S}^m,\ s_{\text{lin}} \in \mathbb{R}^p
```

and with data

```math
A_i \in \mathbb{S}^m\ (i=1,\ldots,n),\ C \in \mathbb{S}^m,\ c\in{\mathbb R}^n,\ D\in{\mathbb R}^{n\times p},\ d\in{\mathbb R}^p.
```

Primal problem
```math
\begin{equation*}\tag{P}
\begin{aligned}
&\max_{X\in\mathbb{S}^m,\,x_{\text{lin}}\in\mathbb{R}^m~}C \bullet X + d^\top x_{\text{lin}}&\\
&\text{{subject to}}\\ 
&\qquad A_i \bullet X+ (D^\top x_{\text{lin}})_i=b_i\,, \quad i=1,\dots,{n}&\\
&\qquad X\succeq 0,\  x_{\text{lin}}\geq 0& 
\end{aligned}
\end{equation*}
```
Dual problem
```math
		\begin{equation*}\tag{D}
	\begin{aligned}
	& \min_{y\in\mathbb{R}^n,\,S\in\mathbb{S}^m,\,s_{\text{lin}}\in\mathbb{R}^m~} c^\top y \\
	& \text{subject to}\\
	&\qquad \sum_{i=1}^{{n}} y_i A_i + S = C,\ \ S\succeq 0\\
	&\qquad Dy+s_{\text{lin}}=d,\ \ s_{\text{lin}}\geq 0
	\end{aligned}
    \end{equation*}
```


## General assumptions (for the IP algorithm to converge)
Slater constraint qualification and strict complementarity

## Low-rank solution
The following assumptions are only needed when the user want to usilize the iterative solver with the low-rank preconditioner (`kit = 1, preconditioner > 0`). *The assumptions are not needed when using Loraine with the direct solver* (`kit = 0`).

### Assumptions:
Main assumption
- We assume that ``X^*``, the solution of (P)(!!), has very low rank.
	
**Be sure about your problem formulation:** If ``X^*`` has low rank then ``S^*``, the solution of the dual problem, has almost full rank and vice versa. Hence if, in your problem, you assume that ``S^*`` has low rank, you cannot use Loraine with iterative solver directly; rather you may need to dualize your formulation using JuMP's `dualize`.

Further assumptions

- **Sparsity of ``A_i``:** Define the matrix ``{\cal A}=[\text{svec}\,A_1,\dots, \text{svec}\,A_n].`` We assume that matrix-vector products with ``{\cal A}`` and ``{\cal A}^\top`` may each be applied in ``O(n)`` flops and memory.
- **Sparsity of ``D``:** The inverse ``(D^\top D)^{-1}`` and matrix-vector product with ``(D^\top D)^{-1}`` may each be computed in ``\mathcal{O}(n)`` flops and memory.

## Low-rank data
- *At the moment, Loraine can only handle rank-one data.*
- *This feature is only relevant for Loraine used with the **direct solver**.*

If you know (or strongly suspect) that *all* data matrices ``A_i`` have rank one, select the option `datarank = -1`. Loraine will factorize the matrices as ``A_i = b_i b_i^\top`` and use only the vectors ``b_i`` in the interior point algorithm. This will gravely reduce the complexity (and the elapsed time) of Loraine.