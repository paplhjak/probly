# Results

This directory holds aggregated metrics output from
`aggregate_results.py`. Files:

- `results.csv` -- flat, one row per (method, dataset, loss, seed).
  Gitignored.
- `results_table.md` -- pivoted table grouped by dataset and loss,
  with mean +/- std across seeds. Gitignored.

The directory is committed via this README placeholder; the
generated CSV/Markdown outputs are gitignored to avoid version-
controlling stochastic intermediate artifacts. Task 7's keystone
e2e test exercises both outputs on synthetic data.
