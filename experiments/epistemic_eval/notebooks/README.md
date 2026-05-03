# Notebooks

Exploration only, **not part of the pipeline**. Anything load-bearing
(metric computation, oracle, extraction, paper figure generation)
must live in `scripts/` as a runnable Python entry point per AGENTS.md
→ "Architecture: strict three-layer separation".

Notebooks are useful for:

- Sanity-checking dataset loaders (visualising `p*` histograms per
  Task 4).
- Inspecting cached `predictions.npz` and `oracle.npz` arrays
  manually.
- Exploring qualitative behaviour of selectors at different `(λ, τ)`.

Notebooks are gitignored at the metadata level (`.ipynb_checkpoints`)
but the `.ipynb` files themselves are committed if they document
something useful.
