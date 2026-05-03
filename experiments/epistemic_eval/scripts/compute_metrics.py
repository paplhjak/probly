#!/usr/bin/env python3
"""Stage 6: compute the per-run AuReC, AuRC, and Pareto-gap metrics.

Reads the cached predictions, the per-loss decomposition outputs, the
loss-specific oracle, and the loss-independent ``p_star`` sidecar
(emitted by :mod:`compute_oracle`); computes the per-point realized
regret via :func:`probly.quantification.realized_regret.compute_realized_regret`;
and produces ``metrics_<loss>.json`` with the schema documented in
``tasks.md`` Task 7.

Outputs (one per ``--loss`` invocation)::

    runs/<run_id>/metrics_<loss>.json   -- numeric metrics per the schema
    runs/<run_id>/metrics_<loss>.config_hash  -- 16-hex sidecar for cache invalidation

Schema (locked, see ``tasks.md`` Task 7)::

    {
        "run_id":         "<run dir basename>",
        "method":         <method name>,
        "dataset":        <dataset name>,
        "loss":           <one of cross_entropy, zero_one, squared, absolute>,
        "seed":           <int>,
        "aurec":          <float>,
        "aurc":           <float>,
        "pareto_gap":     <float>,
        "n_test_points":  <int>,
        "config_hash":    "<16-char hex>",
        "_metrics_version":      <int>,
        "_aurec_version":        <int>,
        "_pareto_gap_version":   <int>,
        "git_commit":     "<hex>",
        "timestamp_utc":  "<ISO 8601 UTC>"
    }

Idempotence: if ``metrics_<loss>.config_hash`` matches the current
hash and ``--force-recompute`` is not passed, the JSON is left
untouched. The hash covers ``method``, ``dataset``, ``loss``,
``seed``, ``_metrics_version``, ``_aurec_version``,
``_pareto_gap_version``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import typing
from typing import Any, cast

import numpy as np
import yaml

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.epistemic_eval.methods._base import hash_config  # noqa: E402
from probly.evaluation.pareto_gap import _PARETO_GAP_VERSION, pareto_gap  # noqa: E402
from probly.evaluation.regret_coverage import (  # noqa: E402
    _AUREC_VERSION,
    aurec,
    aurc,
)
from probly.quantification.realized_regret import (  # noqa: E402
    LossName,
    compute_realized_regret,
)

#: Bumped on math changes to invalidate cached metrics JSON. Independent
#: of ``_AUREC_VERSION`` and ``_PARETO_GAP_VERSION``; this captures
#: changes to the orchestration in this script (e.g. how AuRC's risk
#: is computed, how the BMA is derived from logits, etc.).
_METRICS_VERSION: int = 1


def _git_commit() -> str:
    """Return the current git HEAD short SHA, or ``"unknown"`` on failure."""
    result = subprocess.run(  # noqa: S603
        ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return "unknown"
    return result.stdout.strip() or "unknown"


def _config_hash(
    method: str,
    dataset: str,
    loss: str,
    seed: int,
) -> str:
    """Stable hash covering the fields that decide cache validity."""
    payload = {
        "method": method,
        "dataset": dataset,
        "loss": loss,
        "seed": int(seed),
        "_metrics_version": int(_METRICS_VERSION),
        "_aurec_version": int(_AUREC_VERSION),
        "_pareto_gap_version": int(_PARETO_GAP_VERSION),
    }
    return hash_config(payload)


def _load_run_config(run_dir: Path) -> dict[str, Any]:
    """Read ``config.yaml`` written by stage 2b (``fit_uncertainty.py``)."""
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        msg = f"config.yaml not found at {config_path}; was fit_uncertainty.py run?"
        raise FileNotFoundError(msg)
    return yaml.safe_load(config_path.read_text())


def _resolve_method_dataset_seed(run_dir: Path) -> tuple[str, str, int]:
    """Pull (method_name, dataset_name, seed) from the run's resolved config."""
    cfg = _load_run_config(run_dir)
    method_block = cfg.get("method") or {}
    dataset_block = cfg.get("dataset") or {}
    method = str(method_block.get("name") or "")
    dataset = str(dataset_block.get("name") or "")
    seed_val = cfg.get("seed")
    if not method or not dataset or seed_val is None:
        msg = (
            f"resolved config at {run_dir / 'config.yaml'} is missing method.name, "
            f"dataset.name or seed; got method={method!r}, dataset={dataset!r}, "
            f"seed={seed_val!r}."
        )
        raise KeyError(msg)
    return method, dataset, int(seed_val)


def _bma_from_logits(logits: np.ndarray) -> np.ndarray:
    """Return ``softmax(logits, axis=1).mean(axis=2)``.

    Uses the standard log-sum-exp trick for numerical stability.
    """
    if logits.ndim != 3:
        msg = f"logits must be 3-D (N, K, S); got shape={logits.shape}."
        raise ValueError(msg)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    softmax = exp / exp.sum(axis=1, keepdims=True)
    return np.asarray(softmax.mean(axis=2), dtype=np.float64)


def _load_decomposition(run_dir: Path, loss: str) -> tuple[np.ndarray, np.ndarray]:
    """Load only ``A_hat`` and ``E_hat`` from the per-loss decomposition cache."""
    path = run_dir / f"decomposition_{loss}.npz"
    if not path.exists():
        msg = (
            f"decomposition cache not found at {path}; run "
            f"compute_decomposition.py first."
        )
        raise FileNotFoundError(msg)
    blob = np.load(path, allow_pickle=False)
    return np.asarray(blob["A_hat"]), np.asarray(blob["E_hat"])


def _load_oracle(run_dir: Path, loss: str) -> tuple[np.ndarray, np.ndarray]:
    """Load only ``A_star`` and ``E_star`` from the per-loss oracle cache."""
    path = run_dir / f"oracle_{loss}.npz"
    if not path.exists():
        msg = f"oracle cache not found at {path}; run compute_oracle.py first."
        raise FileNotFoundError(msg)
    blob = np.load(path, allow_pickle=False)
    return np.asarray(blob["A_star"]), np.asarray(blob["E_star"])


def _load_p_star(p_star_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``p_star`` and ``support`` from a sidecar emitted by ``compute_oracle.py``."""
    if not p_star_path.exists():
        msg = f"p_star sidecar not found at {p_star_path}."
        raise FileNotFoundError(msg)
    blob = np.load(p_star_path, allow_pickle=False)
    if "p_star" not in blob.files or "support" not in blob.files:
        msg = (
            f"p_star sidecar at {p_star_path} is missing required keys; "
            f"got {blob.files}."
        )
        raise KeyError(msg)
    return np.asarray(blob["p_star"]), np.asarray(blob["support"])


def main(argv: list[str] | None = None) -> int:
    """Entry point. See module docstring."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Run directory.")
    parser.add_argument(
        "--loss",
        type=str,
        required=True,
        # Derived from the LossName Literal so adding a fifth loss
        # (e.g. CRPS) requires no edits to this orchestration script.
        choices=tuple(typing.get_args(LossName)),
    )
    parser.add_argument(
        "--oracle-run",
        type=Path,
        required=True,
        help="Directory holding oracle_<loss>.npz and (optionally) p_star.npz.",
    )
    parser.add_argument(
        "--p-star-path",
        type=Path,
        default=None,
        help=(
            "Override; defaults to <oracle-run>/p_star.npz. Useful for tests "
            "that build the sidecar at a non-standard location."
        ),
    )
    parser.add_argument("--force-recompute", action="store_true")
    args = parser.parse_args(argv)

    run_dir: Path = args.run
    if not run_dir.exists():
        msg = f"run directory not found at {run_dir}."
        raise FileNotFoundError(msg)
    oracle_run: Path = args.oracle_run
    if not oracle_run.exists():
        msg = f"oracle run directory not found at {oracle_run}."
        raise FileNotFoundError(msg)
    p_star_path = args.p_star_path or (oracle_run / "p_star.npz")

    method, dataset, seed = _resolve_method_dataset_seed(run_dir)
    loss_str = str(args.loss)
    cfg_hash = _config_hash(method, dataset, loss_str, seed)
    metrics_path = run_dir / f"metrics_{loss_str}.json"
    hash_path = run_dir / f"metrics_{loss_str}.config_hash"

    if (
        metrics_path.exists()
        and hash_path.exists()
        and hash_path.read_text().strip() == cfg_hash
        and not args.force_recompute
    ):
        print(
            f"cache hit for run={run_dir.name} loss={loss_str}; "
            f"skipping (sidecar matches)."
        )
        return 0

    predictions_path = run_dir / "predictions.npz"
    if not predictions_path.exists():
        msg = (
            f"predictions.npz not found at {predictions_path}; run "
            f"extract_uncertainties.py first."
        )
        raise FileNotFoundError(msg)
    pred_blob = np.load(predictions_path, allow_pickle=False)
    logits = np.asarray(pred_blob["logits"])
    indices = np.asarray(pred_blob["indices"])

    a_hat, e_hat = _load_decomposition(run_dir, loss_str)
    a_star, e_star = _load_oracle(oracle_run, loss_str)
    p_star, support = _load_p_star(p_star_path)

    # Sanity: the oracle, decomposition and p_star arrays must agree on N.
    n = int(indices.shape[0])
    for arr_name, arr in [
        ("logits", logits),
        ("a_hat", a_hat),
        ("e_hat", e_hat),
        ("a_star", a_star),
        ("e_star", e_star),
        ("p_star", p_star),
    ]:
        if int(arr.shape[0]) != n:
            msg = (
                f"row-count mismatch on {arr_name}: got {arr.shape[0]}, "
                f"expected {n}."
            )
            raise ValueError(msg)

    bma = _bma_from_logits(logits)
    regret = compute_realized_regret(
        p_star,
        bma,
        cast("LossName", loss_str),
        support,
    )

    # AuReC's per-point selector score is the empirical epistemic
    # uncertainty E_hat. AuRC's per-point risk is the absolute realized
    # loss E_y[ell(H_hat, y)], equal to regret + A_star since by
    # definition
    #   regret = E_y[ell(H_hat, y) - ell(H_star, y)]
    #   A_star = E_y[ell(H_star, y)]
    # We compute the cheaper sum.
    risk = regret.astype(np.float64) + a_star.astype(np.float64)
    aurec_value = aurec(e_hat, regret)
    aurc_value = aurc(e_hat, risk)
    pareto_gap_value = pareto_gap(a_hat, e_hat, a_star, e_star)

    payload: dict[str, Any] = {
        "run_id": run_dir.name,
        "method": method,
        "dataset": dataset,
        "loss": loss_str,
        "seed": int(seed),
        "aurec": float(aurec_value),
        "aurc": float(aurc_value),
        "pareto_gap": float(pareto_gap_value),
        "n_test_points": int(n),
        "config_hash": cfg_hash,
        "_metrics_version": int(_METRICS_VERSION),
        "_aurec_version": int(_AUREC_VERSION),
        "_pareto_gap_version": int(_PARETO_GAP_VERSION),
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    metrics_path.write_text(json.dumps(payload, indent=2))
    hash_path.write_text(cfg_hash)
    print(f"wrote {metrics_path}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
