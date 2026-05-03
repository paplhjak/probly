"""Full-network Evidential Classification fit -> save -> load -> extract.

Mirrors the layout of
:mod:`tests.test_full_network_mc_dropout`: a synthetic 64-sample,
3-class fixture exercises the wrapper's full lifecycle in-process,
without any cluster compute. The schema check
(``predictions["alpha"]`` and ``predictions["evidence"]`` are present
and have the expected shapes) is the load-bearing assertion --
without it the downstream decomposition layer cannot dispatch.

Note that we use a tiny 16-feature MLP rather than ResNet18 here:
evidential is single-pass and architecture-agnostic, so the smoke
test does not need a CIFAR-style backbone.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from experiments.epistemic_eval.methods import evidential_classification  # noqa: E402
from experiments.epistemic_eval.methods._base import FeatureProvider  # noqa: E402


_NUM_SAMPLES = 64
_NUM_CLASSES = 3
_BATCH_SIZE = 16
_IN_FEATURES = 16


def _make_provider(seed: int = 0) -> FeatureProvider:
    g = torch.Generator().manual_seed(seed)
    features = torch.randn(_NUM_SAMPLES, _IN_FEATURES, generator=g)
    classes = torch.randint(0, _NUM_CLASSES, (_NUM_SAMPLES,), generator=g)
    labels = torch.zeros(_NUM_SAMPLES, _NUM_CLASSES)
    labels[torch.arange(_NUM_SAMPLES), classes] = 1.0
    batches = [
        (features[i : i + _BATCH_SIZE], labels[i : i + _BATCH_SIZE])
        for i in range(0, _NUM_SAMPLES, _BATCH_SIZE)
    ]
    return FeatureProvider(
        batches=batches,
        mode="full_network",
        feature_dim=None,
        n_classes=_NUM_CLASSES,
        indices=np.arange(_NUM_SAMPLES, dtype=np.int64),
    )


def _factory() -> nn.Module:
    """Tiny MLP factory used as the base classifier."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(_IN_FEATURES, 32),
        nn.ReLU(),
        nn.Linear(32, _NUM_CLASSES),
    )


def test_evidential_classification_fit_extract_roundtrip(tmp_path):
    """fit -> save -> load -> extract produces a valid Dirichlet cache.

    Asserts:
    - ``predictions["alpha"]`` has shape ``(N, K)`` and dtype
      ``float32`` (NOT a sample-axis schema).
    - All ``alpha`` values are strictly greater than 1.0 (the
      ``Softplus + 1`` floor).
    - ``predictions["evidence"]`` has shape ``(N,)`` and dtype
      ``float32``.
    - ``evidence == alpha.sum(axis=1) - K`` (definition).
    - ``predictions["indices"]`` is contiguous ``[0, N)``.
    - The handle round-trips through ``save / load`` without dropping
      state (compare a known weight buffer).
    """
    provider = _make_provider(seed=0)

    method_config = {
        "name": "evidential",
        "epochs": 1,
        "lr": 1.0e-3,
        "weight_decay": 0.0,
    }
    dataset_config = {
        "name": "synthetic_evidential_smoke",
        "extraction_mode": "full_network",
        "metadata": {"num_classes": _NUM_CLASSES},
    }

    handle = evidential_classification.fit(
        method_config=method_config,
        dataset_config=dataset_config,
        data_provider=provider,
        model_factory=_factory,
        seed=0,
    )

    handle_path = tmp_path / "method.pth"
    evidential_classification.save(handle, handle_path)
    reloaded = evidential_classification.load(handle_path)

    # Round-trip: same state_dict keys, same first-layer-weight values.
    assert set(reloaded.state_dict.keys()) == set(handle.state_dict.keys())
    a_key = sorted(handle.state_dict.keys())[0]
    np.testing.assert_array_equal(
        reloaded.state_dict[a_key].detach().cpu().numpy(),
        handle.state_dict[a_key].detach().cpu().numpy(),
    )

    out = evidential_classification.extract(
        reloaded,
        provider,
        n_samples=None,
        model_factory=_factory,
    )

    # Schema assertions.
    assert "alpha" in out
    assert "evidence" in out
    assert "indices" in out
    assert "logits" not in out  # NOT the (N, K, S) schema.

    assert out["alpha"].shape == (_NUM_SAMPLES, _NUM_CLASSES)
    assert out["alpha"].dtype == np.float32
    assert out["evidence"].shape == (_NUM_SAMPLES,)
    assert out["evidence"].dtype == np.float32
    assert out["indices"].shape == (_NUM_SAMPLES,)
    assert out["indices"].dtype == np.int64
    assert np.array_equal(out["indices"], np.arange(_NUM_SAMPLES))

    # Math invariants.
    assert np.all(out["alpha"] > 1.0)  # Softplus + 1 floor.
    assert np.all(np.isfinite(out["alpha"]))
    assert np.all(np.isfinite(out["evidence"]))
    np.testing.assert_allclose(
        out["evidence"],
        out["alpha"].sum(axis=1) - float(_NUM_CLASSES),
        atol=1e-5,
        rtol=1e-5,
    )
