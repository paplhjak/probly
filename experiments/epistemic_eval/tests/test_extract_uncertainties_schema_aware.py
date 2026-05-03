"""Schema-aware ``extract_uncertainties.py`` regression tests.

Pre-Task-5b, the script hard-coded ``predictions.npz``'s ``logits``
key, which broke evidential and ddu (whose ``extract()`` returns
``alpha`` + ``evidence`` and ``probs`` + ``density`` respectively
per their declared schemas). These tests pin the post-fix behaviour:
the script must read ``output_schema`` from the run's config and
write the schema-required keys.

Strategy: install a tiny "fake method" Python module in
``sys.modules`` that returns a controlled dict from ``extract()``;
write the corresponding ``config.yaml`` + a placeholder ``method.pth``
into the run dir; invoke ``extract_uncertainties.main()`` directly;
assert the resulting ``predictions.npz`` has the right keys for the
declared schema.

This approach exercises the script's dispatch logic without paying
the cost of a real wrapper / model.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import yaml  # noqa: E402

from experiments.epistemic_eval.methods._base import FeatureProvider  # noqa: E402
from experiments.epistemic_eval.scripts import extract_uncertainties  # noqa: E402


#: Valid args for ``probly.method.head.MlpHead`` so the linear-probe
#: branch's ``_head_factory()`` call inside ``extract_uncertainties.main()``
#: can build a model. The actual model is never called -- the fake
#: method's ``extract()`` ignores it -- but the script's preamble
#: still constructs one before invoking ``extract()``.
_HEAD_FACTORY_ARGS = {
    "in_features": 4,
    "num_classes": 3,
    "hidden": 4,
    "dropout_p": 0.0,
}


# ---------------------------------------------------------------------------
# Fake method module factory
# ---------------------------------------------------------------------------


def _install_fake_method(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    extract_return: dict[str, Any],
    *,
    head_factory_args: dict[str, Any] | None = None,
) -> None:
    """Install a fake method module under ``module_name`` in sys.modules.

    The fake module satisfies the wrapper protocol: ``load(path)``,
    ``save(handle, path)``, ``extract(handle, provider, n_samples,
    *, model_factory) -> dict``. ``extract`` is hard-wired to return
    ``extract_return`` regardless of inputs, so the test can fully
    control the dict that lands in ``predictions.npz``.
    """

    @dataclass
    class _FakeHandle:
        method_config: dict[str, Any] = field(default_factory=dict)
        dataset_config: dict[str, Any] = field(default_factory=dict)
        seed: int = 0
        head_factory_args: dict[str, Any] | None = None

    def _save(handle: _FakeHandle, path: Path) -> None:  # noqa: ARG001
        # The real wrappers torch.save the handle. For the test we
        # only need *something* on disk so the script can load it.
        path.write_bytes(b"fake")

    def _load(path: Path) -> _FakeHandle:  # noqa: ARG001
        return _FakeHandle(head_factory_args=head_factory_args)

    def _extract(
        handle: _FakeHandle,  # noqa: ARG001
        data_provider: FeatureProvider,  # noqa: ARG001
        n_samples: int = 1,  # noqa: ARG001
        *,
        model_factory: Any = None,  # noqa: ANN401, ARG001
    ) -> dict[str, np.ndarray]:
        return extract_return

    module = types.ModuleType(module_name)
    module.load = _load  # ty: ignore[unresolved-attribute]
    module.save = _save  # ty: ignore[unresolved-attribute]
    module.extract = _extract  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, module_name, module)


def _write_run_dir(
    run_dir: Path,
    method_name: str,
    method_module_name: str,
    output_schema: str,
    n_classes: int,
) -> None:
    """Write a minimal run_dir: config.yaml + method.pth placeholder."""
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "method": {
            "name": method_name,
            "method_module": method_module_name,
            "output_schema": output_schema,
            "n_samples": 1,
        },
        "dataset": {
            "name": "synthetic_schema_test",
            "extraction_mode": "linear_probe",  # picks the cheap branch
            "metadata": {"num_classes": n_classes},
        },
        "seed": 0,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run_dir / "method.pth").write_bytes(b"fake")


def _make_feature_cache(cache_dir: Path, n: int, dim: int) -> None:
    """Write a synthetic features .npz for the linear_probe test branch."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    features = rng.standard_normal((n, dim)).astype(np.float32)
    filenames = np.asarray([f"sample_{i}" for i in range(n)], dtype=object)
    np.savez_compressed(cache_dir / "test.npz", features=features, filenames=filenames)


# ---------------------------------------------------------------------------
# Per-schema tests
# ---------------------------------------------------------------------------


def test_logits_nks_schema_writes_logits_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Backward-compat: ``output_schema='logits_nks'`` writes a 'logits' key."""
    n, k, s = 8, 3, 2
    extract_return = {
        "logits": np.zeros((n, k, s), dtype=np.float32),
        "indices": np.arange(n, dtype=np.int64),
    }
    _install_fake_method(
        monkeypatch,
        "_test_fake_logits_method",
        extract_return,
        head_factory_args=_HEAD_FACTORY_ARGS,
    )

    run_dir = tmp_path / "run"
    _write_run_dir(
        run_dir,
        method_name="logits_method",
        method_module_name="_test_fake_logits_method",
        output_schema="logits_nks",
        n_classes=k,
    )
    feature_cache = tmp_path / "features"
    _make_feature_cache(feature_cache, n=n, dim=4)

    rc = extract_uncertainties.main(
        ["--run", str(run_dir), "--feature-cache-dir", str(feature_cache)]
    )
    assert rc == 0

    blob = np.load(run_dir / "predictions.npz")
    assert set(blob.files) == {"logits", "indices"}
    assert blob["logits"].shape == (n, k, s)
    assert blob["logits"].dtype == np.float32
    assert blob["indices"].shape == (n,)


def test_evidential_alpha_schema_writes_alpha_and_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``output_schema='evidential_alpha'`` writes 'alpha' + 'evidence', NOT 'logits'."""
    n, k = 8, 3
    extract_return = {
        "alpha": np.full((n, k), 2.0, dtype=np.float32),
        "evidence": np.full(n, k * 1.0, dtype=np.float32),
        "indices": np.arange(n, dtype=np.int64),
    }
    _install_fake_method(
        monkeypatch,
        "_test_fake_evidential_method",
        extract_return,
        head_factory_args=_HEAD_FACTORY_ARGS,
    )

    run_dir = tmp_path / "run"
    _write_run_dir(
        run_dir,
        method_name="evidential_method",
        method_module_name="_test_fake_evidential_method",
        output_schema="evidential_alpha",
        n_classes=k,
    )
    feature_cache = tmp_path / "features"
    _make_feature_cache(feature_cache, n=n, dim=4)

    rc = extract_uncertainties.main(
        ["--run", str(run_dir), "--feature-cache-dir", str(feature_cache)]
    )
    assert rc == 0

    blob = np.load(run_dir / "predictions.npz")
    assert set(blob.files) == {"alpha", "evidence", "indices"}
    assert "logits" not in blob.files
    assert blob["alpha"].shape == (n, k)
    assert blob["evidence"].shape == (n,)
    assert blob["indices"].shape == (n,)


def test_ddu_probs_density_schema_writes_probs_and_density(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``output_schema='ddu_probs_density'`` writes 'probs' + 'density', NOT 'logits'."""
    n, k = 8, 3
    extract_return = {
        "probs": np.full((n, k), 1.0 / k, dtype=np.float32),
        "density": np.full(n, -2.0, dtype=np.float32),
        "indices": np.arange(n, dtype=np.int64),
    }
    _install_fake_method(
        monkeypatch,
        "_test_fake_ddu_method",
        extract_return,
        head_factory_args=_HEAD_FACTORY_ARGS,
    )

    run_dir = tmp_path / "run"
    _write_run_dir(
        run_dir,
        method_name="ddu_method",
        method_module_name="_test_fake_ddu_method",
        output_schema="ddu_probs_density",
        n_classes=k,
    )
    feature_cache = tmp_path / "features"
    _make_feature_cache(feature_cache, n=n, dim=4)

    rc = extract_uncertainties.main(
        ["--run", str(run_dir), "--feature-cache-dir", str(feature_cache)]
    )
    assert rc == 0

    blob = np.load(run_dir / "predictions.npz")
    assert set(blob.files) == {"probs", "density", "indices"}
    assert "logits" not in blob.files
    assert blob["probs"].shape == (n, k)
    assert blob["density"].shape == (n,)
    assert blob["indices"].shape == (n,)


def test_default_schema_is_logits_nks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-schema runs (no ``output_schema`` field) default to 'logits_nks'."""
    n, k, s = 4, 3, 2
    extract_return = {
        "logits": np.zeros((n, k, s), dtype=np.float32),
        "indices": np.arange(n, dtype=np.int64),
    }
    _install_fake_method(
        monkeypatch,
        "_test_fake_default_method",
        extract_return,
        head_factory_args=_HEAD_FACTORY_ARGS,
    )

    # Write config.yaml WITHOUT output_schema.
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = {
        "method": {
            "name": "legacy_method",
            "method_module": "_test_fake_default_method",
            "n_samples": 1,
        },
        "dataset": {
            "name": "synthetic_schema_test",
            "extraction_mode": "linear_probe",
            "metadata": {"num_classes": k},
        },
        "seed": 0,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg))
    (run_dir / "method.pth").write_bytes(b"fake")
    feature_cache = tmp_path / "features"
    _make_feature_cache(feature_cache, n=n, dim=4)

    rc = extract_uncertainties.main(
        ["--run", str(run_dir), "--feature-cache-dir", str(feature_cache)]
    )
    assert rc == 0

    blob = np.load(run_dir / "predictions.npz")
    assert set(blob.files) == {"logits", "indices"}


def test_unknown_schema_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    extract_return = {
        "logits": np.zeros((4, 3, 2), dtype=np.float32),
        "indices": np.arange(4, dtype=np.int64),
    }
    _install_fake_method(
        monkeypatch,
        "_test_fake_bad_schema_method",
        extract_return,
        head_factory_args=_HEAD_FACTORY_ARGS,
    )
    run_dir = tmp_path / "run"
    _write_run_dir(
        run_dir,
        method_name="bad_method",
        method_module_name="_test_fake_bad_schema_method",
        output_schema="not_a_real_schema",
        n_classes=3,
    )
    feature_cache = tmp_path / "features"
    _make_feature_cache(feature_cache, n=4, dim=4)

    with pytest.raises(ValueError, match=r"unknown output_schema"):
        extract_uncertainties.main(
            ["--run", str(run_dir), "--feature-cache-dir", str(feature_cache)]
        )


def test_missing_schema_required_key_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the wrapper's extract() omits a schema-required key, the script raises.

    Regression guard for the wrapper-side contract: each wrapper's
    extract() MUST return the keys declared by its schema. Catching
    a mismatch here is much less painful than discovering it during
    a 12-hour SLURM run.
    """
    n, k = 4, 3
    # evidential_alpha requires both 'alpha' and 'evidence'. Omit evidence.
    extract_return = {
        "alpha": np.full((n, k), 2.0, dtype=np.float32),
        "indices": np.arange(n, dtype=np.int64),
    }
    _install_fake_method(
        monkeypatch,
        "_test_fake_missing_key_method",
        extract_return,
        head_factory_args=_HEAD_FACTORY_ARGS,
    )

    run_dir = tmp_path / "run"
    _write_run_dir(
        run_dir,
        method_name="bad_evidential_method",
        method_module_name="_test_fake_missing_key_method",
        output_schema="evidential_alpha",
        n_classes=k,
    )
    feature_cache = tmp_path / "features"
    _make_feature_cache(feature_cache, n=n, dim=4)

    with pytest.raises(ValueError, match=r"missing required \['evidence'\]"):
        extract_uncertainties.main(
            ["--run", str(run_dir), "--feature-cache-dir", str(feature_cache)]
        )
