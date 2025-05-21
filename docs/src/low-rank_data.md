# Low-rank data, more details

Assume that matrices ``A_i`` in ``\sum_{i=1}^{{n}} y_i A_i + S = C,\ \ S\succeq 0,`` are obtained  by
```math
  A_i = B_i B_i^\top
```
with data matrices ``B_i\in{\mathbb R}^{m\times k}`` and with ``k \ll n``.

The complexity of ``H`` assembly is then reduced from ``nm^3`` to ``knm^2``.

*In particular:*
If we know that ``A_i`` have rank one, the decomposition ``A_i = b_i b_i^\top`` is performed by Loraine automatically (`datarank = -1`).

The model is represented internally via the following `struct`:
```@docs
Loraine.MyModel
```
