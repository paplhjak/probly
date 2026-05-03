"""Schema-aware ``compute_metrics.py`` regression tests.

Pre-fix, the script unconditionally indexed ``predictions.npz`` by
the ``"logits"`` key (line 258 in the pre-fix version), which broke
evidential and DDU runs whose ``extract()`` writes ``alpha`` +
``evidence`` and ``probs`` + ``density`` respectively per their
declared schemas.

These tests pin the post-fix behaviour: the script must read
``output_schema`` from the run's ``config.yaml`` and compute the
empirical Bayes (categorical) predictor from the schema-required
keys via :func:`compute_metrics._bma_from_predictions`. Strategy is
the same as ``test_extract_uncertainties_schema_aware``: build a
synthetic run dir + oracle dir, write a per-schema
``predictions.npz``, drive ``compute_metrics.main()`` directly, and
assert the JSON output is finite and well-formed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import yaml  # noqa: E402

from experiments.epistemic_eval.scripts import compute_metrics  # noqa: E402


_N = 12
_K = 4
_S = 3


def _write_oracle_dir(tmp_path: Path, loss: str, seed: int = 0) -> Path:
    """Write an oracle directory with ``oracle_<loss>.npz`` + ``p_star.npz``.

    Values are ``rng``-driven; magnitudes are kept small so AuReC /
    AuRC / Pareto-gap come out finite and in their natural ranges
    regardless of which schema the run uses.
    """
    rng = np.random.default_rng(seed + 17)
    oracle_dir = tmp_path / f"oracle_{loss}"
    oracle_dir.mkdir(parents=True, exist_ok=True)
    indices = np.arange(_N, dtype=np.int64)

    a_star = rng.uniform(0.0, 1.0, size=_N).astype(np.float32)
    e_star = np.zeros(_N, dtype=np.float32)
    h_star = np.zeros(_N, dtype=np.float32)
    np.savez_compressed(
        oracle_dir / f"oracle_{loss}.npz",
        H_star=h_star,
        A_star=a_star,
        E_star=e_star,
        indices=indices,
    )

    p_star = rng.dirichlet(np.ones(_K), size=_N).astype(np.float32)
    support = np.arange(_K, dtype=np.int64)
    np.savez_compressed(
        oracle_dir / "p_star.npz",
        p_star=p_star,
        support=support,
        indices=indices,
    )
    return oracle_dir


def _write_decomposition(run_dir: Path, loss: str, seed: int = 0) -> None:
    """Write a synthetic ``decomposition_<loss>.npz`` with finite (A_hat, E_hat)."""
    rng = np.random.default_rng(seed + 29)
    a_hat = rng.uniform(0.0, 1.0, size=_N).astype(np.float32)
    e_hat = rng.uniform(0.0, 1.0, size=_N).astype(np.float32)
    h_hat = np.zeros(_N, dtype=np.float32)
    np.savez_compressed(
        run_dir / f"decomposition_{loss}.npz",
        A_hat=a_hat,
        E_hat=e_hat,
        H_hat=h_hat,
        indices=np.arange(_N, dtype=np.int64),
    )


def _write_run_config(
    run_dir: Path,
    *,
    method_name: str = "synthetic_method",
    dataset_name: str = "synthetic_dataset",
    seed: int = 0,
    output_schema: str | None = "logits_nks",
) -> None:
    """Write the ``config.yaml`` produced by stage 2b.

    ``output_schema=None`` omits the field entirely so the
    backwards-compat default code path is exercised.
    """
    method_block: dict[str, Any] = {"name": method_name}
    if output_schema is not None:
        method_block["output_schema"] = output_schema
    config = {
        "method": method_block,
        "dataset": {"name": dataset_name},
        "seed": int(seed),
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))


def _run_compute_metrics_inprocess(
    run_dir: Path,
    oracle_dir: Path,
    loss: str,
    *extra_args: str,
) -> int:
    """Drive ``compute_metrics.main()`` in-process (no subprocess)."""
    return compute_metrics.main(
        [
            "--run",
            str(run_dir),
            "--loss",
            loss,
            "--oracle-run",
            str(oracle_dir),
            "--p-star-path",
            str(oracle_dir / "p_star.npz"),
            *extra_args,
        ]
    )


def _assert_finite_locked_schema(payload: dict[str, Any], loss: str) -> None:
    """Assert the JSON payload has the locked keys and finite numerics."""
    expected_keys = {
        "run_id",
        "method",
        "dataset",
        "loss",
        "seed",
        "aurec",
        "aurc",
        "pareto_gap",
        "n_test_points",
        "config_hash",
        "_metrics_version",
        "_aurec_version",
        "_pareto_gap_version",
        "git_commit",
        "timestamp_utc",
    }
    assert expected_keys <= set(payload), payload
    for key in ("aurec", "aurc", "pareto_gap"):
        assert isinstance(payload[key], float), key
        assert np.isfinite(payload[key]), key
    assert payload["loss"] == loss
    assert payload["n_test_points"] == _N


# ---------------------------------------------------------------------------
# Per-schema tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss", ["cross_entropy", "zero_one"])
def test_logits_nks_schema_produces_finite_metrics(
    tmp_path: Path, loss: str
) -> None:
    """``output_schema='logits_nks'`` (the legacy default) keeps working."""
    rng = np.random.default_rng(0)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    logits = rng.standard_normal(size=(_N, _K, _S)).astype(np.float32)
    np.savez_compressed(
        run_dir / "predictions.npz",
        logits=logits,
        indices=np.arange(_N, dtype=np.int64),
    )
    _write_decomposition(run_dir, loss)
    _write_run_config(run_dir, output_schema="logits_nks")
    oracle_dir = _write_oracle_dir(tmp_path, loss)

    rc = _run_compute_metrics_inprocess(run_dir, oracle_dir, loss)
    assert rc == 0
    metrics = json.loads((run_dir / f"metrics_{loss}.json").read_text())
    _assert_finite_locked_schema(metrics, loss)


@pytest.mark.parametrize("loss", ["cross_entropy", "zero_one"])
def test_evidential_alpha_schema_produces_finite_metrics(
    tmp_path: Path, loss: str
) -> None:
    """``output_schema='evidential_alpha'`` post-fix smoke check.

    Pre-fix, this test would have raised ``KeyError: 'logits'`` at
    line 258 of compute_metrics.py. Post-fix, it must drive through
    ``_bma_from_predictions`` -> ``alpha / alpha_0`` and produce
    finite metrics.
    """
    rng = np.random.default_rng(1)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Strictly-positive alphas via softplus(logit) + 1, mimicking
    # what evidential_classification.extract() emits in production.
    raw = rng.standard_normal(size=(_N, _K)).astype(np.float32)
    alpha = np.log1p(np.exp(raw)) + 1.0
    evidence = (alpha.sum(axis=1) - float(_K)).astype(np.float32)
    np.savez_compressed(
        run_dir / "predictions.npz",
        alpha=alpha.astype(np.float32),
        evidence=evidence,
        indices=np.arange(_N, dtype=np.int64),
    )
    _write_decomposition(run_dir, loss)
    _write_run_config(run_dir, output_schema="evidential_alpha")
    oracle_dir = _write_oracle_dir(tmp_path, loss)

    rc = _run_compute_metrics_inprocess(run_dir, oracle_dir, loss)
    assert rc == 0
    metrics = json.loads((run_dir / f"metrics_{loss}.json").read_text())
    _assert_finite_locked_schema(metrics, loss)


@pytest.mark.parametrize("loss", ["cross_entropy", "zero_one"])
def test_ddu_probs_density_schema_produces_finite_metrics(
    tmp_path: Path, loss: str
) -> None:
    """``output_schema='ddu_probs_density'`` post-fix smoke check.

    Pre-fix, this test would have raised ``KeyError: 'logits'``.
    Post-fix, ``_bma_from_predictions`` returns ``probs`` directly
    (no S-axis reduction; DDU is single-pass).
    """
    rng = np.random.default_rng(2)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    probs_raw = rng.uniform(0.0, 1.0, size=(_N, _K)).astype(np.float32)
    probs = probs_raw / probs_raw.sum(axis=1, keepdims=True)
    density = rng.uniform(-10.0, 0.0, size=_N).astype(np.float32)
    np.savez_compressed(
        run_dir / "predictions.npz",
        probs=probs.astype(np.float32),
        density=density,
        indices=np.arange(_N, dtype=np.int64),
    )
    _write_decomposition(run_dir, loss)
    _write_run_config(run_dir, output_schema="ddu_probs_density")
    oracle_dir = _write_oracle_dir(tmp_path, loss)

    rc = _run_compute_metrics_inprocess(run_dir, oracle_dir, loss)
    assert rc == 0
    metrics = json.loads((run_dir / f"metrics_{loss}.json").read_text())
    _assert_finite_locked_schema(metrics, loss)


def test_default_schema_is_logits_nks(tmp_path: Path) -> None:
    """Configs without ``output_schema`` fall back to ``logits_nks``.

    The pre-Task-5b configs didn't declare an ``output_schema``
    field; this default keeps them readable.
    """
    loss = "cross_entropy"
    rng = np.random.default_rng(3)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    logits = rng.standard_normal(size=(_N, _K, _S)).astype(np.float32)
    np.savez_compressed(
        run_dir / "predictions.npz",
        logits=logits,
        indices=np.arange(_N, dtype=np.int64),
    )
    _write_decomposition(run_dir, loss)
    _write_run_config(run_dir, output_schema=None)  # no field -> default
    oracle_dir = _write_oracle_dir(tmp_path, loss)

    rc = _run_compute_metrics_inprocess(run_dir, oracle_dir, loss)
    assert rc == 0
    metrics = json.loads((run_dir / f"metrics_{loss}.json").read_text())
    _assert_finite_locked_schema(metrics, loss)


def test_unknown_schema_raises(tmp_path: Path) -> None:
    """Unknown ``output_schema`` strings must fail loudly, not silently."""
    loss = "cross_entropy"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    np.savez_compressed(
        run_dir / "predictions.npz",
        logits=np.zeros((_N, _K, _S), dtype=np.float32),
        indices=np.arange(_N, dtype=np.int64),
    )
    _write_decomposition(run_dir, loss)
    _write_run_config(run_dir, output_schema="not_a_real_schema")
    oracle_dir = _write_oracle_dir(tmp_path, loss)

    with pytest.raises(ValueError, match=r"unknown output_schema"):
        _run_compute_metrics_inprocess(run_dir, oracle_dir, loss)


def test_missing_schema_required_key_raises(tmp_path: Path) -> None:
    """If predictions.npz omits a schema-required key, fail loudly."""
    loss = "cross_entropy"
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    # Schema declares evidential_alpha (needs 'alpha') but cache
    # only has 'logits'. The error should come from
    # _bma_from_predictions, not a downstream NaN.
    np.savez_compressed(
        run_dir / "predictions.npz",
        logits=np.zeros((_N, _K, _S), dtype=np.float32),
        indices=np.arange(_N, dtype=np.int64),
    )
    _write_decomposition(run_dir, loss)
    _write_run_config(run_dir, output_schema="evidential_alpha")
    oracle_dir = _write_oracle_dir(tmp_path, loss)

    with pytest.raises(ValueError, match=r"requires `predictions\['alpha'\]`"):
        _run_compute_metrics_inprocess(run_dir, oracle_dir, loss)


# ---------------------------------------------------------------------------
# Direct-call sanity checks for _bma_from_predictions
# ---------------------------------------------------------------------------


def test_bma_evidential_matches_dirichlet_mean() -> None:
    """``_bma_from_predictions`` on evidential schema returns alpha / alpha_0."""
    alpha = np.array(
        [
            [10.0, 1.0, 1.0],   # alpha_0 = 12, p_hat = [10/12, 1/12, 1/12]
            [1.0, 1.0, 1.0],    # uniform
        ],
        dtype=np.float64,
    )
    evidence = alpha.sum(axis=1) - alpha.shape[1]
    bma = compute_metrics._bma_from_predictions(
        {"alpha": alpha, "evidence": evidence},
        "evidential_alpha",
    )
    expected = alpha / alpha.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(bma, expected, atol=1e-12)
    np.testing.assert_allclose(bma.sum(axis=1), 1.0, atol=1e-12)


def test_bma_ddu_returns_probs_unchanged() -> None:
    """``_bma_from_predictions`` on DDU schema returns probs (with float64 cast)."""
    probs = np.array(
        [[0.7, 0.2, 0.1], [1 / 3, 1 / 3, 1 / 3]],
        dtype=np.float32,
    )
    density = np.array([-2.0, -10.0], dtype=np.float32)
    bma = compute_metrics._bma_from_predictions(
        {"probs": probs, "density": density},
        "ddu_probs_density",
    )
    assert bma.dtype == np.float64
    np.testing.assert_allclose(bma, probs.astype(np.float64), atol=1e-7)


def test_bma_logits_nks_lse_stable() -> None:
    """``_bma_from_predictions`` on logits schema is numerically stable on extreme logits."""
    # Logits with one very large value per row to exercise the LSE shift.
    logits = np.zeros((2, 3, 4), dtype=np.float32)
    logits[0, 0, :] = 1000.0  # extreme but tractable via shift trick
    logits[1, 2, :] = 500.0
    bma = compute_metrics._bma_from_predictions(
        {"logits": logits}, "logits_nks"
    )
    assert np.all(np.isfinite(bma))
    np.testing.assert_allclose(bma.sum(axis=1), 1.0, atol=1e-7)
    # Row 0: class 0 wins; row 1: class 2 wins.
    assert bma[0, 0] > 0.99
    assert bma[1, 2] > 0.99


def test_resolve_output_schema_default(tmp_path: Path) -> None:
    """Pre-Task-5b configs (no ``output_schema``) resolve to ``logits_nks``."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run_config(run_dir, output_schema=None)
    schema = compute_metrics._resolve_output_schema(run_dir)
    assert schema == "logits_nks"


def test_resolve_output_schema_unknown_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_run_config(run_dir, output_schema="bogus")
    with pytest.raises(ValueError, match=r"unknown output_schema"):
        compute_metrics._resolve_output_schema(run_dir)
