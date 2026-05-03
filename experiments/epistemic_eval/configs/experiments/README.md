# Experiment configs

Full grids over (method × dataset × seed) for the paper. Each YAML is
fanned out by `scripts/launch_grid.py` (Task 8).

The main paper grid is `main.yaml`. Smaller debug grids may live
alongside it (e.g. `smoke.yaml` for a single (method, dataset, seed)
end-to-end run on a tiny subset).

See `decisions.md` → "Seeds and replication" for the seed convention
(5 per cell) and `decisions.md` → "Methods to evaluate" for the
frozen method list.
