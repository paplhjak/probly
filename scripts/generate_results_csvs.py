#!/usr/bin/env python3
"""Emit per-(dataset, loss) CSVs of all per-seed metrics + correlations.

One CSV per (dataset, loss) under
``experiments/epistemic_eval/results/csv/``. Each CSV has one row per
(method, seed) the suite is supposed to contain. When the run dir is
missing or the pipeline did not finish, the row is still emitted with
NaN metrics and a ``status`` describing what happened, so a downstream
LaTeX-emitter can format gaps explicitly.

Pulls from two roots:

* APPA-REAL: local ``experiments/epistemic_eval/runs/`` (currently
  the only place APPA-REAL has been run).
* Everything else: cluster-mounted ``cluster_probly/probly/...``.

Rank correlations included (Spearman, per-row):

* ``rho_Ahat_Ehat``     -- method's own decomposition coupling.
* ``rho_Ahat_Astar``    -- method's aleatoric vs oracle aleatoric.
* ``rho_Ahat_regret``   -- method's aleatoric vs realized regret.
* ``rho_Ehat_Astar``    -- method's epistemic vs oracle aleatoric.
* ``rho_Ehat_regret``   -- method's epistemic vs realized regret.
* ``rho_Astar_regret``  -- oracle aleatoric vs realized regret
                            (method-dependent because regret depends
                            on the method's BMA).

Pairs against ``E*`` are omitted because the codebase's oracle stores
``E_star`` identically zero (the Bayes-optimal predictor under p* has
zero epistemic uncertainty about itself), so any Spearman against
``E*`` is undefined.
"""

from __future__ import annotations

import csv
import json
import math
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
sys.path.insert(0, str(_REPO_ROOT))

from experiments.epistemic_eval.scripts import compute_metrics  # noqa: E402
from probly.evaluation.pareto_gap import pareto_gap as _pareto_gap_fn  # noqa: E402
from probly.quantification.realized_regret import compute_realized_regret  # noqa: E402

LOCAL_RUNS = _REPO_ROOT / "experiments" / "epistemic_eval" / "runs"
# When running on the laptop the cluster lives at the read-only sshfs
# mount; when running on the cluster the same script reads its own
# ``runs/`` directory directly. Detect by checking whether the mount
# path exists; fall back to ``LOCAL_RUNS`` when it doesn't.
_CLUSTER_MOUNT = _REPO_ROOT / "cluster_probly" / "probly" / "experiments" / "epistemic_eval" / "runs"
CLUSTER_RUNS = _CLUSTER_MOUNT if _CLUSTER_MOUNT.exists() else LOCAL_RUNS
OUT_DIR = _REPO_ROOT / "experiments" / "epistemic_eval" / "results" / "csv"

DATASET_LOSSES: dict[str, list[str]] = {
    "cifar10h": ["cross_entropy", "zero_one"],
    "benthic": ["cross_entropy", "zero_one"],
    "mice_bone": ["cross_entropy", "zero_one"],
    "pig": ["cross_entropy", "zero_one"],
    "plankton": ["cross_entropy", "zero_one"],
    "quality_mri": ["cross_entropy", "zero_one"],
    "dcic_synthetic": ["cross_entropy", "zero_one"],
    "treeversity_1": ["cross_entropy", "zero_one"],
    "treeversity_6": ["cross_entropy", "zero_one"],
    "turkey": ["cross_entropy", "zero_one"],
    "appa_real": ["squared", "absolute"],
}

# Which runs root to read for each dataset.
DATASET_RUNS_ROOT: dict[str, Path] = {
    "appa_real": LOCAL_RUNS,
}
for ds in DATASET_LOSSES:
    DATASET_RUNS_ROOT.setdefault(ds, CLUSTER_RUNS)

# Which methods are expected per dataset, and how many seeds.
DCIC_METHODS = ("mc_dropout", "evidential", "ensemble", "ddu")
EXPECTED: dict[str, tuple[tuple[str, ...], range]] = {
    "cifar10h": (DCIC_METHODS, range(3)),
    "appa_real": (("mc_dropout", "evidential", "ensemble", "ddu", "laplace"), range(5)),
}
for ds in ("benthic", "mice_bone", "pig", "plankton", "quality_mri",
           "dcic_synthetic", "treeversity_1", "treeversity_6", "turkey"):
    EXPECTED[ds] = (DCIC_METHODS, range(5))

CSV_COLUMNS = [
    "dataset", "loss", "method", "seed", "run_dir",
    "n_test_points",
    "aurec", "excess_aurec", "n_aurec", "aurc", "pareto_gap",
    "rho_Ahat_Ehat", "rho_Ahat_Astar", "rho_Ahat_regret",
    "rho_Ehat_Astar", "rho_Ehat_regret", "rho_Astar_regret",
    "status", "notes",
]

# Status taxonomy.
S_OK = "OK"
S_MISSING_RUN = "MISSING_RUN"
S_NO_PREDICTIONS = "NO_PREDICTIONS"
S_NO_DECOMPOSITION = "NO_DECOMPOSITION"
S_NO_METRICS = "NO_METRICS"
S_NO_ORACLE = "NO_ORACLE"
S_ERROR = "ERROR"


def _spear(a: np.ndarray, b: np.ndarray) -> float:
    """Safe Spearman; NaN if either input is constant."""
    if a.std() <= 0 or b.std() <= 0:
        return float("nan")
    rho = spearmanr(a, b).statistic
    return float(rho) if rho is not None else float("nan")


def _find_run_dirs(runs_root: Path, method: str, dataset: str, seed: int) -> list[Path]:
    """Return ALL run dirs for ``(method, dataset, seed)`` across all dates.

    The cluster aggregator counts a (method, dataset, seed) cell once per
    matching run dir, regardless of date. We mirror that: pre-fix
    May-4 runs and post-fix May-6 runs both contribute. Returns dirs
    in lexicographic (~ chronological) order so the CSV row ordering
    is stable across re-runs.
    """
    return sorted(runs_root.glob(f"*_main_{method}_{dataset}_seed{seed}"))


def _find_oracle_dir(runs_root: Path, dataset: str, seed: int) -> Path | None:
    """Resolve oracle: same-seed if exists, else seed-0 fallback."""
    same = sorted(runs_root.glob(f"*_main_oracle_{dataset}_seed{seed}"))
    if same:
        return same[-1]
    fallback = sorted(runs_root.glob(f"*_main_oracle_{dataset}_seed0"))
    return fallback[-1] if fallback else None


def _row_for(
    dataset: str, loss: str, method: str, seed: int, runs_root: Path,
    rd: Path | None = None,
) -> dict[str, Any]:
    """Build one CSV row; populates NaN + status when artefacts are missing.

    If ``rd`` is provided, treat it as the specific run dir to read.
    If ``rd`` is None and no run dir matches ``(method, dataset, seed)``,
    emit a placeholder row with ``status=MISSING_RUN``.
    """
    out: dict[str, Any] = {c: "" for c in CSV_COLUMNS}
    out.update({"dataset": dataset, "loss": loss, "method": method, "seed": seed})
    nan_keys = [
        "n_test_points", "aurec", "excess_aurec", "n_aurec", "aurc", "pareto_gap",
        "rho_Ahat_Ehat", "rho_Ahat_Astar", "rho_Ahat_regret",
        "rho_Ehat_Astar", "rho_Ehat_regret", "rho_Astar_regret",
    ]
    for k in nan_keys:
        out[k] = float("nan")

    if rd is None:
        out["status"] = S_MISSING_RUN
        out["notes"] = "no run dir matching glob"
        return out
    out["run_dir"] = str(rd.relative_to(_REPO_ROOT))

    oracle_dir = _find_oracle_dir(runs_root, dataset, seed)
    if oracle_dir is None:
        out["status"] = S_NO_ORACLE
        out["notes"] = "no oracle dir for dataset/seed"
        return out

    metrics_path = rd / f"metrics_{loss}.json"
    if not metrics_path.exists():
        # Distinguish stuck-at-fit (no predictions) vs stuck-at-extract
        # (predictions but no decomp).
        if not (rd / "predictions.npz").exists():
            out["status"] = S_NO_PREDICTIONS
            out["notes"] = "fit done but extract did not run"
            return out
        if not (rd / f"decomposition_{loss}.npz").exists():
            out["status"] = S_NO_DECOMPOSITION
            out["notes"] = f"predictions but no decomposition_{loss}.npz"
            return out
        out["status"] = S_NO_METRICS
        out["notes"] = f"decomposition present but no metrics_{loss}.json"
        return out

    try:
        m = json.load(open(metrics_path))
        out["aurec"] = float(m["aurec"])
        out["excess_aurec"] = float(m["excess_aurec"])
        out["n_aurec"] = float(m["n_aurec"])
        out["aurc"] = float(m["aurc"])
        out["pareto_gap"] = float(m["pareto_gap"])
        out["n_test_points"] = int(m["n_test_points"])

        cfg = yaml.safe_load((rd / "config.yaml").read_text())
        schema = (cfg.get("method") or {}).get("output_schema", "logits_nks")
        pred = np.load(rd / "predictions.npz")
        decomp = np.load(rd / f"decomposition_{loss}.npz")
        oracle = np.load(oracle_dir / f"oracle_{loss}.npz")
        p_star_blob = np.load(oracle_dir / "p_star.npz")
        p_star = np.asarray(p_star_blob["p_star"])
        support = np.asarray(p_star_blob["support"])

        bma = compute_metrics._bma_from_predictions(
            {k: np.asarray(pred[k]) for k in pred.files if k != "indices"},
            schema,
        )
        a_hat = np.asarray(decomp["A_hat"])
        e_hat = np.asarray(decomp["E_hat"])
        a_star = np.asarray(oracle["A_star"])
        regret = np.asarray(compute_realized_regret(p_star, bma, loss, support))

        out["rho_Ahat_Ehat"] = _spear(a_hat, e_hat)
        out["rho_Ahat_Astar"] = _spear(a_hat, a_star)
        out["rho_Ahat_regret"] = _spear(a_hat, regret)
        out["rho_Ehat_Astar"] = _spear(e_hat, a_star)
        out["rho_Ehat_regret"] = _spear(e_hat, regret)
        out["rho_Astar_regret"] = _spear(a_star, regret)

        # Pareto-gap fallback: if the cluster's metrics_<loss>.json was
        # written under the pre-fix Pareto-gap version (<3, where
        # ``e_star = 0`` made the oracle surface degenerate), recompute
        # locally with the corrected frequentist convention. With v3+
        # the JSON value is already correct and we just keep what we
        # loaded above. The Pareto-gap call is the expensive part of
        # this script (51 lambdas x 91 directions x N points), so this
        # short-circuit makes the regenerator fast when the cluster is
        # up-to-date.
        pg_version = int(m.get("_pareto_gap_version", 0))
        if pg_version < 3 and regret.std() > 0:
            try:
                out["pareto_gap"] = float(
                    _pareto_gap_fn(a_hat, e_hat, a_star, regret)
                )
            except Exception as exc:  # noqa: BLE001
                out["pareto_gap"] = float("nan")
                notes_bits = list(
                    out.get("notes", "").split(",") if out.get("notes") else []
                )
                notes_bits.append(f"pareto_gap_failed:{type(exc).__name__}")
                out["notes"] = ",".join(b for b in notes_bits if b)

        notes_bits: list[str] = []
        if a_hat.std() <= 0:
            notes_bits.append("constant_A_hat")
        if e_hat.std() <= 0:
            notes_bits.append("constant_E_hat")
        if regret.std() <= 0:
            notes_bits.append("constant_regret")
        out["status"] = S_OK
        out["notes"] = ",".join(notes_bits)
    except Exception as exc:  # noqa: BLE001 - emit an error row
        out["status"] = S_ERROR
        out["notes"] = f"{type(exc).__name__}: {exc}"
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for dataset, losses in DATASET_LOSSES.items():
        runs_root = DATASET_RUNS_ROOT[dataset]
        if not runs_root.exists():
            print(f"  SKIP {dataset}: runs root missing at {runs_root}", flush=True)
            continue
        methods, seeds = EXPECTED[dataset]
        for loss in losses:
            csv_path = OUT_DIR / f"{dataset}__{loss}.csv"
            rows: list[dict[str, Any]] = []
            for method in methods:
                for seed in seeds:
                    dirs = _find_run_dirs(runs_root, method, dataset, seed)
                    if not dirs:
                        rows.append(_row_for(dataset, loss, method, seed, runs_root))
                        continue
                    # One CSV row per matching run dir (May-4 + May-6 are
                    # both kept; the LaTeX aggregator collapses them).
                    for rd in dirs:
                        rows.append(_row_for(dataset, loss, method, seed, runs_root, rd=rd))
            with csv_path.open("w", newline="") as fp:
                w = csv.DictWriter(fp, fieldnames=CSV_COLUMNS)
                w.writeheader()
                for r in rows:
                    # csv.DictWriter writes "" for NaN by default; use
                    # explicit "nan" string so downstream pandas parses
                    # them as NaN.
                    out = {
                        k: ("nan" if isinstance(v, float) and math.isnan(v) else v)
                        for k, v in r.items()
                    }
                    w.writerow(out)
            n_ok = sum(1 for r in rows if r["status"] == S_OK)
            n_total = len(rows)
            print(f"  wrote {csv_path.relative_to(_REPO_ROOT)}  "
                  f"({n_ok}/{n_total} OK)", flush=True)
            summary_rows.append({
                "dataset": dataset, "loss": loss, "n_ok": n_ok, "n_total": n_total,
                "n_missing_run": sum(1 for r in rows if r["status"] == S_MISSING_RUN),
                "n_no_extract": sum(1 for r in rows if r["status"] == S_NO_PREDICTIONS),
                "n_no_decomp": sum(1 for r in rows if r["status"] == S_NO_DECOMPOSITION),
                "n_no_metrics": sum(1 for r in rows if r["status"] == S_NO_METRICS),
                "n_error": sum(1 for r in rows if r["status"] == S_ERROR),
            })

    sum_path = OUT_DIR / "_summary.csv"
    with sum_path.open("w", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"\n  wrote {sum_path.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
