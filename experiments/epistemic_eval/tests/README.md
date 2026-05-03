# Paper-specific tests

These tests cover the experiment layer: oracle on real datasets,
end-to-end smoke runs, dataset-specific sanity checks. Library tests
(AuReC, selectors, Pareto-gap, regression decompositions) live under
`tests/probly/...` per probly's pytest convention — see AGENTS.md
"Verification".

Planned files:

- `test_oracle_classification.py` — hand-computed `(H*, A*, E*)` for
  small classification examples; verify the oracle output matches.
- `test_oracle_regression.py` — hand-computed for squared and
  absolute losses on small integer-age examples.
- `test_pipeline_smoke.py` — end-to-end on a tiny synthetic dataset
  (10 points, 3 classes), all methods, single seed, < 30s (Task 7).

These run via `pytest experiments/epistemic_eval/tests/` (or are
collected by the project-wide `pytest tests/` if the testpath is
extended in `pyproject.toml`; current config has `testpaths =
["tests"]` so we run this directory explicitly until proven needed
elsewhere).
