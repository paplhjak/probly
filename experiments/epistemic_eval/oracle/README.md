# Oracle layer

Oracles only make sense for first-order datasets where `p*(y|x)` is
observed, so this code is paper-specific and stays here (not in the
probly library).

The oracle takes a dataset config and a loss string and computes
`(A*, E*, H*, p*)` per test point, caching them in `oracle.npz`. It
does **not** depend on any learned method.

Loss dispatch (per `decisions.md` → "Loss handling in the codebase"):

```
config.loss
  ├─ cross_entropy → probly.quantification.decomposition.SecondOrderEntropyDecomposition
  ├─ zero_one      → probly.quantification.decomposition.SecondOrderZeroOneDecomposition
  ├─ squared       → probly.quantification.decomposition.SquaredErrorDecomposition (Task 6.5)
  └─ absolute      → probly.quantification.decomposition.AbsoluteErrorDecomposition (Task 6.5)
```

Files (Task 6):

- `from_p_star.py` — `compute_oracle(p_star, loss)` dispatcher.
