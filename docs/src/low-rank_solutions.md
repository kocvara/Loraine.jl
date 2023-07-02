# Low-rank solution, more details
In each iteration of the (primal-dual, predictor-corrector) interior-point method we have to solve two systems of linear equations in variable ``y`` with a scaling matrix ``W``:

```math
\begin{equation}
H={\cal A}^T(W\otimes W){\cal A}+ D^T X_{\rm lin}S^{-1}_{\rm lin} D\,.
\end{equation}
```

The complexity of interior point method can be significantly reduced by solving the Schur complement equation ``H y = r`` by an iterative method, rather than Cholesky solver, a standard choice of most IP-based software. 

## Iterative solver

Loraine uses preconditioned conjugate gradient (CG) method.
	
- What can be gained:
    - ``H`` assembly: lower complexity, ``H`` does not have to be stored in memory
	- ``Hy=r`` can only be solved approximately, one CG iteration has very low complexity (matrix-vector multiplication)
	
- Drawback:	
    - ``H`` getting (very) ill-conditioned, CG may need very many iterations and may not work at all

*We need a good preconditioner!*

## Low-rank preconditioners for Interior-Point method

### Preconditioner ``H_\alpha`` (`preconditioner = 1`)

Critical observation (due to Richard Y. Zhang and Javad Lavaei, IEEE, 2017) reveals that
*if the solution ``X^*`` is low-rank then ``W`` will be low-rank.*

Hence 
``
W=W_0+UU^T
``
and
```math
H={\cal A}^T(W_0\otimes W_0){\cal A}+{\cal A}^T(U\otimes Z)(U\otimes Z)^T{\cal A}+ D^T X_{\rm lin}S^{-1}_{\rm lin} D\,.
```

This leads to the following preconditioner called ``H_{\alpha}``:
```math
H_{\alpha}={\tau^2 I} +{V}{V}^T+ D^T X_{\rm lin}S^{-1}_{\rm lin} D\,.
```

Here ``V = {\cal A}^T(U\otimes Z)`` has low rank, so we can use Sherman-Morrison-Woodbury formula to compute ``H_{\alpha}^{-1}``.

### Preconditioner ``H_\beta`` (`preconditioner = 2`)
In many problems, the last term in ... is dominating in the first iterations of the IP algorithm, before the low-rank structure of ``W`` is clearly recognized.

This observation lead to the idea of a simplified preconditioner called ``H_\beta`` and defined as follows

```math
\begin{align*}
H_\beta=\tau^2 I+D^\top X_{\text{lin}}S_{\text{lin}}^{-1}D,
\end{align*}
```
in which $\tau$ is defined as in the previous section.
This matrix is easy to invert; in fact, the matrix is diagonal in many problems. It is therefore an extremely cheap preconditioner that is efficient in the first iterations of the IP algorithm.

### Hybrid preconditioner (`preconditioner = 4`)
For relevant problems, we therefore recommend to use a *hybrid preconditioner*: we start the IP iterations with ``H_\beta`` and, once it becomes inefficient, switch to the more expensive but more efficient ``H_\alpha``. 