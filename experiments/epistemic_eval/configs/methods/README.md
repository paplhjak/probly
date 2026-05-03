# Method configs

One YAML per UQ method. Frozen list for the paper's main results
table per `decisions.md` → "Methods to evaluate":

- `mcd.yaml`       — `probly.method.dropout`
- `ensemble.yaml`  — `probly.method.ensemble`
- `llla.yaml`      — external `laplace-torch` (see Task 5; library
                     promotion to `src/probly/method/laplace/` is TBD)
- `evidential.yaml` — `probly.method.evidential_classification` (and
                      `evidential_regression` for APPA-REAL)

Adding a new method = adding a new YAML and a thin adapter under
`scripts/method_<name>.py`. The pipeline accepts any probly method via
config; the four above are the frozen set for the paper, not a code
restriction.
