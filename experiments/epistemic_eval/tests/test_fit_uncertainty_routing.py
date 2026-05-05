"""Routing regression tests for ``fit_uncertainty.main()``.

Pre-fix, the full-network dispatch only sent the ``ensemble`` method
to the real CIFAR-10 train provider. Evidential and DDU fell into
the load-pretrained branch and got the metadata-only stub provider,
so:

* ``evidential.fit()`` silently iterated zero batches (no training).
* ``ddu.fit()`` Phase B crashed at ``torch.cat([])`` because the
  feature-walk had no inputs.

These tests pin the post-fix behaviour: the
``_FROM_SCRATCH_FULL_NETWORK_METHODS`` set must include exactly
``{ensemble, evidential, ddu}``, and the router must invoke
``_make_full_network_cifar10h_train_provider`` for any of those
three methods on CIFAR-10H without ``ensemble_classifier_paths``.

We don't run real CIFAR-10 training in this test (that would require
the full canonical pickles). Instead we install a fake method module
that records the data_provider it received, then assert the provider
is the real CIFAR-10 train provider (sentinel: it has > 0 batches
and ``mode == "full_network"``) for the three from-scratch methods.
We monkeypatch ``_make_full_network_cifar10h_train_provider`` to
return a small synthetic provider so the test runs in milliseconds.
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
from experiments.epistemic_eval.scripts import fit_uncertainty  # noqa: E402


def _build_synthetic_provider(n_batches: int) -> FeatureProvider:
    """Build a small synthetic FeatureProvider with ``n_batches`` batches."""
    batch_x = torch.zeros(2, 16, dtype=torch.float32)
    batch_y = torch.zeros(2, 3, dtype=torch.float32)
    batch_y[:, 0] = 1.0
    return FeatureProvider(
        batches=[(batch_x, batch_y)] * n_batches,
        mode="full_network",
        feature_dim=None,
        n_classes=3,
        indices=np.arange(2, dtype=np.int64),
    )


def _synthetic_train_provider(_dataset_config: dict[str, Any]) -> FeatureProvider:
    """Stand-in for ``_make_full_network_cifar10h_train_provider``.

    Yields a single batch of synthetic 16-feature inputs + 3-class
    soft labels so the fake method's fit() can detect "non-empty
    batches" without paying the cost of loading 50k CIFAR-10
    images.
    """
    return _build_synthetic_provider(n_batches=1)


def _synthetic_train_val_providers(
    _dataset_config: dict[str, Any], **_kwargs: Any
) -> tuple[FeatureProvider, FeatureProvider]:
    """Stand-in for ``_make_full_network_cifar10h_train_val_providers``.

    Returns ``(train, val)`` providers; the train side has 1 batch
    (the regression sentinel from the original routing test) and the
    val side has 1 small batch so val-tracking does not silently
    short-circuit on zero-length val sets.
    """
    return _build_synthetic_provider(n_batches=1), _build_synthetic_provider(n_batches=1)


def _install_recording_method(
    monkeypatch: pytest.MonkeyPatch,
    module_name: str,
    received: dict[str, Any],
) -> None:
    """Install a fake method module that records the provider passed to fit().

    ``received`` is mutated in place: it gets ``"n_batches"`` and
    ``"provider_mode"`` keys after ``fit()`` runs.
    """

    @dataclass
    class _FakeHandle:
        method_config: dict[str, Any] = field(default_factory=dict)
        dataset_config: dict[str, Any] = field(default_factory=dict)
        seed: int = 0
        head_factory_args: dict[str, Any] | None = None

    def _save(handle: _FakeHandle, path: Path) -> None:  # noqa: ARG001
        path.write_bytes(b"fake")

    def _fit(
        method_config: dict[str, Any],
        dataset_config: dict[str, Any],
        data_provider: FeatureProvider,
        model_factory: Any = None,  # noqa: ANN401, ARG001
        seed: int = 0,
        *,
        val_data_provider: FeatureProvider | None = None,
    ) -> _FakeHandle:
        # Materialise the iterator so we can count batches without
        # mutating the FeatureProvider's internal list.
        batches = list(data_provider)
        received["n_batches"] = len(batches)
        received["provider_mode"] = data_provider.mode
        received["val_provider_present"] = val_data_provider is not None
        if val_data_provider is not None:
            received["n_val_batches"] = len(list(val_data_provider))
        return _FakeHandle(
            method_config=method_config,
            dataset_config=dataset_config,
            seed=seed,
        )

    module = types.ModuleType(module_name)
    module.fit = _fit  # ty: ignore[unresolved-attribute]
    module.save = _save  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, module_name, module)


def _write_configs(
    tmp_path: Path,
    method_name: str,
    method_module_name: str,
    output_schema: str,
) -> tuple[Path, Path]:
    method_cfg = {
        "name": method_name,
        "method_module": method_module_name,
        "output_schema": output_schema,
    }
    method_path = tmp_path / "method.yaml"
    method_path.write_text(yaml.safe_dump(method_cfg))

    dataset_cfg = {
        "name": "cifar10h",
        "extraction_mode": "full_network",
        "loader_kwargs": {"root": "data/cifar10h"},
        "supported_losses": ["cross_entropy"],
        "classifier": {
            "architecture": "torch.nn.Linear",
            "num_classes": 3,
            "pretrained_classifier_path": None,
        },
        "metadata": {"num_classes": 3},
    }
    dataset_path = tmp_path / "dataset.yaml"
    dataset_path.write_text(yaml.safe_dump(dataset_cfg))
    return method_path, dataset_path


@pytest.mark.parametrize(
    "method_name",
    sorted(fit_uncertainty._FROM_SCRATCH_FULL_NETWORK_METHODS),
)
def test_from_scratch_methods_get_real_train_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    """ensemble / evidential / ddu must all see a non-empty train provider.

    Pre-fix, only ensemble took the real-provider branch; evidential
    and ddu silently got the stub (zero batches), causing silent
    miscompiles and a torch.cat crash respectively. This is the
    direct routing pin.
    """
    received: dict[str, Any] = {}
    module_name = f"_test_routing_{method_name}"
    _install_recording_method(monkeypatch, module_name, received)

    # Stub the train-provider builder: the production builder reads
    # 50k canonical CIFAR-10 pickles, which we don't want to pay for
    # in a unit test.
    monkeypatch.setattr(
        fit_uncertainty,
        "_make_full_network_cifar10h_train_val_providers",
        _synthetic_train_val_providers,
    )

    method_path, dataset_path = _write_configs(
        tmp_path,
        method_name=method_name,
        method_module_name=module_name,
        output_schema="logits_nks",
    )

    rc = fit_uncertainty.main(
        [
            "--method-config",
            str(method_path),
            "--dataset-config",
            str(dataset_path),
            "--seed",
            "0",
            "--run-dir",
            str(tmp_path / "run"),
        ]
    )
    assert rc == 0

    assert received["provider_mode"] == "full_network"
    # The synthetic provider has exactly one batch; if the stub
    # provider had been routed through (zero batches) the assertion
    # would fail with n_batches=0, which is the regression we're
    # guarding against.
    assert received["n_batches"] == 1, (
        f"method {method_name!r} got {received['n_batches']} batches, "
        f"expected 1 from the real CIFAR-10 train provider stub. "
        f"Pre-fix bug: the routing fell through to the stub provider."
    )


def test_mc_dropout_still_uses_stub_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """mc_dropout must NOT iterate train data at fit time.

    The full-network mc_dropout flow loads a pretrained classifier
    and applies ``probly.method.dropout`` -- it short-circuits
    training via ``epochs = 0``. The provider it receives should be
    the stub (zero batches). Inverse of the parametrised test above.
    """
    received: dict[str, Any] = {}
    module_name = "_test_routing_mc_dropout"
    _install_recording_method(monkeypatch, module_name, received)

    # If routing falls through to mc_dropout's branch correctly the
    # train-provider builder is never called; if the stub providers
    # the wrong path, this hot-fail would surface a regression.
    def _should_not_be_called(
        _dataset_config: dict[str, Any], **_kwargs: Any
    ) -> tuple[FeatureProvider, FeatureProvider]:
        msg = "mc_dropout must not invoke _make_full_network_cifar10h_train_val_providers"
        raise AssertionError(msg)

    monkeypatch.setattr(
        fit_uncertainty,
        "_make_full_network_cifar10h_train_val_providers",
        _should_not_be_called,
    )

    method_path, dataset_path = _write_configs(
        tmp_path,
        method_name="mc_dropout",
        method_module_name=module_name,
        output_schema="logits_nks",
    )
    # mc_dropout's full-network branch wants a stage-1 classifier
    # state_dict on disk; provide a one-tensor placeholder.
    classifier_path = tmp_path / "classifier.pth"
    torch.save({"weight": torch.zeros(3, 16)}, classifier_path)

    rc = fit_uncertainty.main(
        [
            "--method-config",
            str(method_path),
            "--dataset-config",
            str(dataset_path),
            "--seed",
            "0",
            "--run-dir",
            str(tmp_path / "run"),
            "--classifier-state-dict-path",
            str(classifier_path),
        ]
    )
    assert rc == 0
    assert received["n_batches"] == 0


def test_from_scratch_set_contains_expected_methods() -> None:
    """The locked from-scratch set is exactly {ensemble, evidential, ddu}.

    A documentation-pin: when the locked method set changes, this
    test must change too. That's intentional; growing the set
    silently (e.g. via a typo or a refactor) is exactly what this
    test catches.
    """
    assert fit_uncertainty._FROM_SCRATCH_FULL_NETWORK_METHODS == frozenset(
        {"ensemble", "evidential", "ddu"}
    )
