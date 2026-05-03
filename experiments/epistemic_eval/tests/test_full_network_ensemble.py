"""Full-network Deep Ensembles: load N classifiers from disk, extract.

Mirrors :mod:`test_full_network_mc_dropout` but exercises the
``full_network`` ensemble path that ``decisions.md`` "ImageNet-ReaL
ensemble construction" calls "option (a)": ``N`` independently
pretrained classifiers loaded from disk, no per-member training.

The fixture saves ``N=3`` ResNet18 state_dicts (each with a different
torch seed so weights genuinely differ between members) under
``ensemble_classifier_paths`` and asserts that:

- :func:`ensemble.fit` accepts the path list and constructs an
  :class:`EnsembleHandle` with three captured ``state_dicts`` (no
  per-member training, mirroring how ImageNet-ReaL is meant to run on
  the cluster).
- :func:`ensemble.extract` produces a ``(N, K, n_members)`` logits
  array, ``float32``, with members yielding distinct outputs (so the
  ensemble actually carries diversity).
- The ``n_members > len(paths)`` failure mode raises with the locked
  remediation message.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from experiments.epistemic_eval.methods import ensemble  # noqa: E402
from experiments.epistemic_eval.methods._base import FeatureProvider  # noqa: E402
from probly_benchmark.resnet import ResNet18  # noqa: E402


_NUM_SAMPLES = 64
_NUM_CLASSES = 10
_BATCH_SIZE = 16
_N_MEMBERS = 3


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


def _save_random_resnets(tmp_path, n_members: int) -> list[str]:
    """Save ``n_members`` ResNet18 state_dicts with different seeds.

    Returns the absolute path strings of each ``classifier_<i>.pth``,
    in seed order.
    """
    paths: list[str] = []
    for i in range(n_members):
        torch.manual_seed(1000 + i)
        model = ResNet18()
        path = tmp_path / f"classifier_{i}.pth"
        torch.save(model.state_dict(), path)
        paths.append(str(path))
    return paths


def _factory():
    """Plain ResNet18 factory; ensemble.fit (use_pretrained branch)
    doesn't invoke this, but ensemble.extract does to load each
    member's state_dict into a fresh module.
    """
    return ResNet18()


def test_full_network_ensemble_loads_pretrained_members(tmp_path):
    """``ensemble_classifier_paths`` -> handle with N captured state_dicts."""
    paths = _save_random_resnets(tmp_path, _N_MEMBERS)
    provider = _make_provider(seed=0)

    method_config = {"name": "ensemble", "n_members": _N_MEMBERS}
    dataset_config = {
        "name": "cifar10h_synthetic",
        "extraction_mode": "full_network",
        "ensemble_classifier_paths": paths,
        "metadata": {"num_classes": _NUM_CLASSES},
    }

    handle = ensemble.fit(
        method_config=method_config,
        dataset_config=dataset_config,
        data_provider=provider,
        model_factory=_factory,
        seed=0,
    )

    assert len(handle.state_dicts) == _N_MEMBERS
    # Members really differ: pick the same parameter from any two
    # state_dicts and check it's distinct.
    a = handle.state_dicts[0]["linear.weight"]
    b = handle.state_dicts[1]["linear.weight"]
    assert not torch.equal(a, b)

    handle_path = tmp_path / "method.pth"
    ensemble.save(handle, handle_path)
    reloaded = ensemble.load(handle_path)
    assert len(reloaded.state_dicts) == _N_MEMBERS

    out = ensemble.extract(
        reloaded,
        provider,
        n_samples=None,
        model_factory=_factory,
    )

    assert out["logits"].shape == (_NUM_SAMPLES, _NUM_CLASSES, _N_MEMBERS)
    assert out["logits"].dtype == np.float32
    assert out["indices"].shape == (_NUM_SAMPLES,)
    assert np.array_equal(out["indices"], np.arange(_NUM_SAMPLES))

    # Members are independently initialised, so any two produce
    # different logits at the same input.
    assert not np.allclose(out["logits"][:, :, 0], out["logits"][:, :, 1])
    assert not np.allclose(out["logits"][:, :, 0], out["logits"][:, :, 2])


def test_full_network_ensemble_too_few_paths_raises(tmp_path):
    """``n_members > len(ensemble_classifier_paths)`` -> ValueError.

    Locked remediation message per ``decisions.md`` -> "Methods to
    evaluate" / brief.
    """
    paths = _save_random_resnets(tmp_path, n_members=2)  # but ask for 3
    provider = _make_provider(seed=0)

    method_config = {"name": "ensemble", "n_members": 3}
    dataset_config = {
        "name": "cifar10h_synthetic",
        "extraction_mode": "full_network",
        "ensemble_classifier_paths": paths,
        "metadata": {"num_classes": _NUM_CLASSES},
    }

    with pytest.raises(ValueError, match="ensemble_classifier_paths has only 2 entries"):
        ensemble.fit(
            method_config=method_config,
            dataset_config=dataset_config,
            data_provider=provider,
            model_factory=_factory,
            seed=0,
        )
