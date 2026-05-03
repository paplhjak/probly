"""Full-network DDU fit -> save -> load -> extract.

Synthetic 64-sample, 3-class, 16-feature fixture exercises the
wrapper's two-phase pipeline (Phase A: spectral-norm classifier
training; Phase B: GMM density-head fit) end to end in-process.

probly's DDU emits a UserWarning when no residual connections are
found in the model. We use a small MLP for speed, accepting the
warning -- the smoke test verifies the plumbing, not the OOD-
detection efficacy that requires a ResNet-class architecture.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from experiments.epistemic_eval.methods import ddu as ddu_wrapper  # noqa: E402
from experiments.epistemic_eval.methods._base import FeatureProvider  # noqa: E402


_NUM_SAMPLES = 64
_NUM_CLASSES = 3
_BATCH_SIZE = 16
_IN_FEATURES = 16
_HIDDEN = 32


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
    """3-Linear MLP. probly's DDU walks Linear layers reversed to
    identify the head; the last Linear is treated as the
    classification head."""
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(_IN_FEATURES, _HIDDEN),
        nn.ReLU(),
        nn.Linear(_HIDDEN, _HIDDEN),
        nn.ReLU(),
        nn.Linear(_HIDDEN, _NUM_CLASSES),
    )


def test_ddu_fit_extract_roundtrip(tmp_path):
    """fit -> save -> load -> extract produces a valid (probs, density) cache.

    Asserts:
    - ``predictions["probs"]`` has shape ``(N, K)`` and is row-
      stochastic (rows sum to 1.0 within atol=1e-5).
    - ``predictions["density"]`` has shape ``(N,)`` and contains
      finite floats.
    - ``predictions["logits"]`` is NOT in the cache (this verifies
      the schema choice).
    - The handle round-trips through ``save / load`` *including the
      GMM buffers* (means, scale_tril, log_pi). We check by
      comparing the GMM ``means`` buffer pre- and post-roundtrip.
    """
    provider = _make_provider(seed=0)

    method_config = {
        "name": "ddu",
        "epochs": 1,
        "lr": 1.0e-3,
        "weight_decay": 0.0,
        "sn_coeff": 3.0,
    }
    dataset_config = {
        "name": "synthetic_ddu_smoke",
        "extraction_mode": "full_network",
        "metadata": {"num_classes": _NUM_CLASSES},
    }

    # probly's DDU warns about non-residual networks; the warning is
    # benign for plumbing tests. Suppress to keep test output clean.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        handle = ddu_wrapper.fit(
            method_config=method_config,
            dataset_config=dataset_config,
            data_provider=provider,
            model_factory=_factory,
            seed=0,
        )

    # The GMM buffers (means / scale_tril / log_pi) must be in the
    # state_dict so they survive save/load.
    assert any("density_head.means" in k for k in handle.state_dict)
    assert any("density_head.scale_tril" in k for k in handle.state_dict)
    assert any("density_head.log_pi" in k for k in handle.state_dict)

    means_before = next(
        v.detach().cpu().clone()
        for k, v in handle.state_dict.items()
        if k.endswith("density_head.means")
    )

    handle_path = tmp_path / "method.pth"
    ddu_wrapper.save(handle, handle_path)
    reloaded = ddu_wrapper.load(handle_path)

    means_after = next(
        v.detach().cpu().clone()
        for k, v in reloaded.state_dict.items()
        if k.endswith("density_head.means")
    )
    np.testing.assert_array_equal(means_before.numpy(), means_after.numpy())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        out = ddu_wrapper.extract(
            reloaded,
            provider,
            n_samples=None,
            model_factory=_factory,
        )

    # Schema assertions.
    assert "probs" in out
    assert "density" in out
    assert "indices" in out
    assert "logits" not in out  # NOT the (N, K, S) schema.

    assert out["probs"].shape == (_NUM_SAMPLES, _NUM_CLASSES)
    assert out["probs"].dtype == np.float32
    assert out["density"].shape == (_NUM_SAMPLES,)
    assert out["density"].dtype == np.float32
    assert out["indices"].shape == (_NUM_SAMPLES,)
    assert out["indices"].dtype == np.int64
    assert np.array_equal(out["indices"], np.arange(_NUM_SAMPLES))

    # Math invariants.
    assert np.all(np.isfinite(out["probs"]))
    assert np.all(np.isfinite(out["density"]))
    np.testing.assert_allclose(
        out["probs"].sum(axis=1),
        np.ones(_NUM_SAMPLES, dtype=np.float32),
        atol=1e-5,
        rtol=1e-5,
    )
