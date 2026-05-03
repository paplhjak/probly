"""Full-network MC-Dropout: load classifier from disk, apply dropout, extract.

Exercises the ``full_network`` code path of
:mod:`experiments.epistemic_eval.methods.mc_dropout` end to end with a
synthetic ResNet18 fixture. The fixture mirrors what stage 1
(``train_classifier.py``) would write under
``runs/<basecls_run_id>/classifier.pth``: a CIFAR-style ResNet18
state_dict with random weights. The method module loads it via the
factory, calls :func:`probly.method.dropout.dropout` to inject the
``nn.Dropout`` module, and runs ``S`` forward passes at extract time
with dropout active.

Real-data wiring (constructing a test :class:`FeatureProvider` over
CIFAR-10 / ImageNet test images) is a Task-9 cluster concern; this
test bypasses the script and supplies a synthetic in-process provider
directly so the full-network code path is covered now rather than
discovered to be broken on the cluster later.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.epistemic_eval.methods import mc_dropout  # noqa: E402
from experiments.epistemic_eval.methods._base import FeatureProvider  # noqa: E402
from probly_benchmark.resnet import ResNet18  # noqa: E402


_NUM_SAMPLES = 64
_NUM_CLASSES = 10
_BATCH_SIZE = 16
_S = 4


def _make_provider(seed: int = 0) -> FeatureProvider:
    g = torch.Generator().manual_seed(seed)
    images = torch.randn(_NUM_SAMPLES, 3, 32, 32, generator=g)
    classes = torch.randint(0, _NUM_CLASSES, (_NUM_SAMPLES,), generator=g)
    labels = torch.zeros(_NUM_SAMPLES, _NUM_CLASSES)
    labels[torch.arange(_NUM_SAMPLES), classes] = 1.0
    batches = [
        (images[i : i + _BATCH_SIZE], labels[i : i + _BATCH_SIZE])
        for i in range(0, _NUM_SAMPLES, _BATCH_SIZE)
    ]
    return FeatureProvider(
        batches=batches,
        mode="full_network",
        feature_dim=None,
        n_classes=_NUM_CLASSES,
        indices=np.arange(_NUM_SAMPLES, dtype=np.int64),
    )


def _classifier_factory_with_loaded_state_dict(
    state_dict: dict[str, torch.Tensor],
):
    """Mirror what ``fit_uncertainty.py`` does: a factory whose output
    has the saved classifier weights pre-loaded. Each call returns a
    fresh module so ensemble-style callers don't get aliased params.
    """

    def factory():
        model = ResNet18()
        model.load_state_dict(state_dict)
        return model

    return factory


def test_full_network_mc_dropout_fit_extract_roundtrip(tmp_path):
    """fit -> save -> load -> extract on a synthetic CIFAR-style ResNet18.

    Asserts:
    - ``predictions["logits"]`` has shape ``(N, K, S)`` and dtype
      ``float32``.
    - ``predictions["indices"]`` is contiguous ``[0, N)``.
    - The ``S`` samples are not all identical: dropout actually
      activates at inference time and adds visible per-sample variance.
    """
    torch.manual_seed(0)
    base = ResNet18()
    classifier_path = tmp_path / "classifier.pth"
    torch.save(base.state_dict(), classifier_path)
    state_dict = torch.load(classifier_path, map_location="cpu", weights_only=False)

    provider = _make_provider(seed=0)
    factory = _classifier_factory_with_loaded_state_dict(state_dict)

    method_config = {
        "name": "mc_dropout",
        "n_samples": _S,
        "classifier_dropout_p": 0.5,
        # epochs=0 short-circuits training: load + apply dropout only.
        "epochs": 0,
    }
    dataset_config = {
        "name": "cifar10h_synthetic",
        "extraction_mode": "full_network",
        "metadata": {"num_classes": _NUM_CLASSES},
    }

    handle = mc_dropout.fit(
        method_config=method_config,
        dataset_config=dataset_config,
        data_provider=provider,
        model_factory=factory,
        seed=0,
    )

    handle_path = tmp_path / "method.pth"
    mc_dropout.save(handle, handle_path)
    reloaded = mc_dropout.load(handle_path)

    out = mc_dropout.extract(
        reloaded,
        provider,
        n_samples=_S,
        model_factory=factory,
    )

    assert out["logits"].shape == (_NUM_SAMPLES, _NUM_CLASSES, _S)
    assert out["logits"].dtype == np.float32
    assert out["indices"].shape == (_NUM_SAMPLES,)
    assert out["indices"].dtype == np.int64
    assert np.array_equal(out["indices"], np.arange(_NUM_SAMPLES))

    # Dropout-induced variability across samples: at p=0.5, drawing S
    # different forward passes through the same loaded model should
    # produce non-trivially different logits. We compare any two
    # samples.
    assert not np.allclose(out["logits"][:, :, 0], out["logits"][:, :, 1])
