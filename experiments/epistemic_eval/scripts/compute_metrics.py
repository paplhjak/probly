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
from probly.quantification.decomposition import OutputSchema  # noqa: E402
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
    output_schema: str,
) -> str:
    """Stable hash covering the fields that decide cache validity.

    Includes ``output_schema`` so a method that switches schemas
    (e.g. evidential moving from ``evidential_alpha`` to a future
    ``evidential_logits_alpha`` variant) invalidates the cached
    metrics JSON instead of silently reusing stale values that were
    computed against a different cache layout.
    """
    payload = {
        "method": method,
        "dataset": dataset,
        "loss": loss,
        "seed": int(seed),
        "output_schema": output_schema,
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


def _resolve_output_schema(run_dir: Path) -> str:
    """Read ``output_schema`` from the run's method config.

    Defaults to ``"logits_nks"`` when the field is absent so caches
    written before Task 5b's schema-aware refactor (which had only
    sampling-based methods) keep working.

    Raises ``ValueError`` on any unknown schema string so a typo in
    the config blows up at metric-computation time rather than after
    a 12-hour SLURM job.
    """
    cfg = _load_run_config(run_dir)
    method_block = cfg.get("method") or {}
    raw = method_block.get("output_schema")
    if raw is None:
        return "logits_nks"
    if not isinstance(raw, str):
        msg = (
            f"method config 'output_schema' must be a string, got "
            f"{type(raw).__name__}: {raw!r}."
        )
        raise TypeError(msg)
    valid = OutputSchema.__args__  # type: ignore[attr-defined]
    if raw not in valid:
        msg = (
            f"unknown output_schema {raw!r} in method config at "
            f"{run_dir / 'config.yaml'}; expected one of {valid!r}."
        )
        raise ValueError(msg)
    return raw


def _bma_from_predictions(
    predictions: dict[str, np.ndarray],
    schema: str,
) -> np.ndarray:
    """Return the empirical Bayes (categorical) predictor from cached arrays.

    Each schema implies its own way of collapsing the cache to an
    ``(N, K)`` row-stochastic predictor, used downstream by
    :func:`compute_realized_regret`:

    * ``"logits_nks"``       -- ``softmax(logits, axis=1).mean(axis=2)``;
                                 the per-sample softmax averaged over
                                 the ``S``-axis (Bayesian model
                                 average over posterior samples).
                                 Numerically stable via log-sum-exp.
    * ``"evidential_alpha"`` -- ``alpha / alpha.sum(axis=1, keepdims=True)``;
                                 the Dirichlet posterior mean
                                 (equivalent to the BMA over the
                                 implied categorical posterior).
                                 No S-axis: evidential is a single
                                 forward pass that emits Dirichlet
                                 concentration parameters directly.
    * ``"ddu_probs_density"`` -- ``probs`` (already row-stochastic);
                                 DDU caches its softmax classifier
                                 head's output directly. The
                                 GMM-density is an OOD signal, NOT
                                 part of the categorical predictor.

    Args:
        predictions: Dict of cached arrays from ``predictions.npz``.
            The required keys depend on ``schema``.
        schema: One of the entries in
            :data:`probly.quantification.decomposition.OutputSchema`.

    Returns:
        ``(N, K) float64`` row-stochastic categorical predictor.

    Raises:
        ValueError: If ``schema`` is unknown, the schema-required
            keys are missing, or the cached arrays have the wrong
            shape.
    """
    if schema == "logits_nks":
        if "logits" not in predictions:
            msg = (
                f"schema={schema!r} requires `predictions['logits']`; "
                f"got keys {sorted(predictions)}."
            )
            raise ValueError(msg)
        logits = predictions["logits"]
        if logits.ndim != 3:
            msg = f"logits must be 3-D (N, K, S); got shape={logits.shape}."
            raise ValueError(msg)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        softmax = exp / exp.sum(axis=1, keepdims=True)
        return np.asarray(softmax.mean(axis=2), dtype=np.float64)
    if schema == "evidential_alpha":
        if "alpha" not in predictions:
            msg = (
                f"schema={schema!r} requires `predictions['alpha']`; "
                f"got keys {sorted(predictions)}."
            )
            raise ValueError(msg)
        alpha = np.asarray(predictions["alpha"], dtype=np.float64)
        if alpha.ndim != 2:
            msg = f"alpha must be 2-D (N, K); got shape={alpha.shape}."
            raise ValueError(msg)
        alpha0 = alpha.sum(axis=1, keepdims=True)
        return alpha / alpha0
    if schema == "ddu_probs_density":
        if "probs" not in predictions:
            msg = (
                f"schema={schema!r} requires `predictions['probs']`; "
                f"got keys {sorted(predictions)}."
            )
            raise ValueError(msg)
        probs = np.asarray(predictions["probs"], dtype=np.float64)
        if probs.ndim != 2:
            msg = f"probs must be 2-D (N, K); got shape={probs.shape}."
            raise ValueError(msg)
        return probs
    msg = (
        f"unknown output_schema: {schema!r}; expected one of "
        f"{OutputSchema.__args__!r}."  # type: ignore[attr-defined]
    )
    raise ValueError(msg)


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
    output_schema = _resolve_output_schema(run_dir)
    loss_str = str(args.loss)
    cfg_hash = _config_hash(method, dataset, loss_str, seed, output_schema)
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
    # Dict-style load: each schema's per-method extract() writes its
    # own keys (logits / alpha+evidence / probs+density). The "indices"
    # key is universal and lives outside the schema dispatch.
    if "indices" not in pred_blob.files:
        msg = (
            f"predictions.npz at {predictions_path} is missing the "
            f"required 'indices' key; got {list(pred_blob.files)}."
        )
        raise KeyError(msg)
    predictions: dict[str, np.ndarray] = {
        key: np.asarray(pred_blob[key])
        for key in pred_blob.files
        if key != "indices"
    }
    indices = np.asarray(pred_blob["indices"])

    a_hat, e_hat = _load_decomposition(run_dir, loss_str)
    a_star, e_star = _load_oracle(oracle_run, loss_str)
    p_star, support = _load_p_star(p_star_path)

    bma = _bma_from_predictions(predictions, output_schema)

    # Sanity: the BMA, oracle, decomposition and p_star arrays must
    # all agree on N. We use BMA (rather than any specific cached
    # array) as the schema-agnostic source for the row count, since
    # `_bma_from_predictions` already validates the schema-required
    # array's shape contract.
    n = int(indices.shape[0])
    for arr_name, arr in [
        ("bma", bma),
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
