"""MC-Dropout wrapper for the epistemic-eval pipeline.

For paper reproducibility, all results are reported as mean +/- std
across 5 seeds. CUDA non-determinism (atomic ops on GPU) may cause
sub-1e-4 logit drift across runs; this is well below inter-seed
variance and does not affect any reported metric. Bit-identical
reproducibility holds only on CPU with deterministic algorithms
enabled.

This wrapper covers two modes:

* ``linear_probe`` (APPA-REAL): ``model_factory`` returns a fresh
  :class:`probly.method.head.MlpHead` -- the dropout layer is part of
  the architecture and is activated at inference time via
  ``model.train()``.
* ``full_network`` (CIFAR-10H, ImageNet-ReaL): ``model_factory``
  returns a base classifier (e.g. CIFAR ``ResNet18``); we then call
  :func:`probly.method.dropout.dropout` to inject dropout layers
  before each ``nn.Linear``.

The wrapper bypasses probly's :class:`Sampler` because we want raw
pre-softmax logits cached on disk, not a
:class:`CategoricalDistribution` (see the Bypass-of-Sampler comment
in :func:`extract`).

Reproducibility note: per-forward-pass stochasticity comes from
PyTorch's global RNG (probly's
``_enforce_train_mode`` toggles ``model.train()``/``model.eval()``;
see ``src/probly/representer/sampler/torch.py:18-46``). We therefore
seed the global RNG once at the top of :func:`fit` and :func:`extract`
via :func:`_base.setup_determinism`; the entire ``S``-loop uses that
single root seed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from probly.method.dropout import dropout

from ._base import FeatureProvider, setup_determinism


@dataclass
class McDropoutHandle:
    """Per-run state produced by :func:`fit`.

    Attributes:
        state_dict: Trained weights (CPU tensors).
        head_factory_args: Args passed to the head factory in
            ``linear_probe`` mode; ``None`` in ``full_network`` mode
            (caller reconstructs the architecture from
            ``dataset_config['classifier']``).
        method_config: Resolved method config dict.
        dataset_config: Resolved dataset config dict.
        seed: Root seed for this run.
    """

    state_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    head_factory_args: dict[str, Any] | None = None
    method_config: dict[str, Any] = field(default_factory=dict)
    dataset_config: dict[str, Any] = field(default_factory=dict)
    seed: int = 0


def _required(config: dict[str, Any], key: str) -> Any:  # noqa: ANN401
    """Read ``key`` from ``config`` or raise loudly if missing.

    Project convention: silent defaults are forbidden. If the user
    forgets a key, the pipeline must fail with a clear message.
    """
    if key not in config:
        msg = f"missing required config key '{key}'"
        raise KeyError(msg)
    return config[key]


def _train_mc_dropout_model(
    model: nn.Module,
    data_provider: FeatureProvider,
    method_config: dict[str, Any],
    seed: int,
) -> nn.Module:
    """Train a single classifier with cross-entropy on soft labels.

    The optimizer config is overridden by
    ``dataset_config['training']`` if the caller wires that through
    (the test suite passes a small flat method config with ``epochs``
    and ``lr``). Reads ``method_config`` for ``epochs``, ``lr``, and
    optionally ``weight_decay``.

    Per-epoch progress is printed to stdout in the format
    ``epoch <e>/<E>: train_loss=<float>`` matching the convention the
    other UQ wrappers use, so SLURM logs surface progress for
    multi-epoch runs (APPA-REAL linear-probe, 50 epochs). The print
    is a no-op when ``epochs == 0`` (the full-network load-pretrained
    short-circuit used by CIFAR-10H/DCIC mc_dropout).
    """
    setup_determinism(seed)
    epochs = int(method_config.get("epochs", 1))
    lr = float(method_config.get("lr", 1.0e-3))
    weight_decay = float(method_config.get("weight_decay", 0.0))

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for x, y in data_provider:
            logits = model(x)
            # KL-equivalent cross-entropy on soft targets; matches
            # decisions.md "Loss handling in the codebase" for
            # cross-entropy datasets, and is the natural training
            # surrogate for the regression-style APPA-REAL head too.
            log_probs = torch.log_softmax(logits, dim=1)
            loss = -(y * log_probs).sum(dim=1).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        print(
            f"epoch {epoch + 1}/{epochs}: train_loss={train_loss:.4f}",
            flush=True,
        )
    return model


def fit(
    method_config: dict[str, Any],
    dataset_config: dict[str, Any],
    data_provider: FeatureProvider,
    model_factory: Callable[[], nn.Module],
    seed: int,
) -> McDropoutHandle:
    """Train a single classifier with dropout active.

    For ``linear_probe`` mode, ``model_factory`` returns an
    :class:`MlpHead` with ``dropout_p`` set per the method config; the
    Dropout module is part of the locked head architecture so it
    participates in MC-Dropout at inference without further surgery.

    For ``full_network`` mode, ``model_factory`` returns a base
    classifier (e.g. CIFAR ``ResNet18``). We then call
    :func:`probly.method.dropout.dropout` to inject ``nn.Dropout``
    before each ``nn.Linear`` (see
    ``src/probly/method/dropout/torch.py:25``).

    Sets ``torch.manual_seed`` and
    ``torch.use_deterministic_algorithms`` via
    :func:`_base.setup_determinism`. Returns a handle suitable for
    :func:`save` / :func:`load` and :func:`extract`.

    Args:
        method_config: Resolved method config dict.
        dataset_config: Resolved dataset config dict.
        data_provider: Training data provider (image or feature mode).
        model_factory: Zero-argument callable returning a fresh
            :class:`torch.nn.Module`. The factory is responsible for
            architecture; ``fit`` only handles training.
        seed: Root seed for this run.

    Returns:
        The trained :class:`McDropoutHandle`.
    """
    base_model = model_factory()
    if data_provider.mode == "full_network":
        # probly's dropout transformation prepends nn.Dropout before
        # every nn.Linear in the model (with is_first_layer skipping
        # the first encountered Linear). See
        # src/probly/method/dropout/torch.py:25.
        p = float(_required(method_config, "classifier_dropout_p"))
        model: nn.Module = dropout(base_model, p=p)  # ty:ignore[invalid-argument-type]
    else:
        # linear-probe: dropout is already part of MlpHead's architecture.
        model = base_model
    trained = _train_mc_dropout_model(model, data_provider, method_config, seed)
    head_factory_args = method_config.get("head_factory_args")
    return McDropoutHandle(
        state_dict={k: v.detach().cpu() for k, v in trained.state_dict().items()},
        head_factory_args=dict(head_factory_args) if head_factory_args is not None else None,
        method_config=dict(method_config),
        dataset_config=dict(dataset_config),
        seed=int(seed),
    )


def save(handle: McDropoutHandle, path: Path) -> None:
    """Persist a handle to ``path`` via ``torch.save``.

    Args:
        handle: The handle to save.
        path: Destination file (parent directory must exist).
    """
    torch.save(asdict(handle), path)


def load(path: Path) -> McDropoutHandle:
    """Reload a handle previously written by :func:`save`.

    Args:
        path: Source file produced by :func:`save`.

    Returns:
        The reconstructed :class:`McDropoutHandle`.
    """
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return McDropoutHandle(**blob)


def _set_dropout_train_mode(model: nn.Module) -> None:
    """Activate dropout at inference; freeze BatchNorm running stats.

    We want stochasticity from dropout at inference, but not from
    batch-norm running-mean drift. Iterate the module tree and put
    only Dropout-family modules in ``train`` mode; everything else
    stays in ``eval``.
    """
    model.eval()
    dropout_types = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    for module in model.modules():
        if isinstance(module, dropout_types):
            module.train()


def extract(
    handle: McDropoutHandle,
    data_provider: FeatureProvider,
    n_samples: int,
    *,
    model_factory: Callable[[], nn.Module],
) -> dict[str, np.ndarray]:
    """Run ``S = n_samples`` forward passes with dropout active.

    Returns a dict with::

        {"logits": np.ndarray of shape (N, K, S) float32,
         "indices": np.ndarray of shape (N,) int64}

    Bypass-of-Sampler note:

    .. code-block::

        # probly's Sampler converts to CategoricalDistribution via
        # create_categorical_distribution_from_logits
        # (src/probly/predictor/_common.py:195-200). We cache raw
        # pre-softmax logits so the loss-specific decompositions
        # (Task 6.5) can produce a (N, K, S) tensor that downstream
        # code is free to interpret. We therefore run forward passes
        # manually instead of using Sampler.predict().

    Args:
        handle: A trained handle produced by :func:`fit`.
        data_provider: Data provider for the test split.
        n_samples: ``S`` -- number of stochastic forward passes.
        model_factory: Zero-argument callable returning the same
            architecture as ``fit`` used. For ``full_network`` mode,
            we re-apply :func:`probly.method.dropout.dropout` so the
            architecture matches the saved ``state_dict``.

    Returns:
        Dict with ``"logits"`` (``(N, K, S)`` float32) and
        ``"indices"`` (``(N,)`` int64).
    """
    setup_determinism(handle.seed)
    base_model = model_factory()
    if handle.dataset_config.get("extraction_mode") == "full_network":
        # Same comment as in fit(): see
        # src/probly/method/dropout/torch.py:25.
        p = float(_required(handle.method_config, "classifier_dropout_p"))
        model: nn.Module = dropout(base_model, p=p)  # ty:ignore[invalid-argument-type]
    else:
        model = base_model
    model.load_state_dict(handle.state_dict)
    _set_dropout_train_mode(model)

    # Run inference on GPU when one is available; falls back to CPU
    # otherwise. The MC-Dropout forward-pass loop is the dominant
    # cost on CIFAR-10 / ImageNet test sets, and CPU is ~50x slower
    # than a consumer GPU for ResNet-18-class models.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    _set_dropout_train_mode(model)  # to(device) re-runs train(False), so re-enable dropout

    n = data_provider.n_samples
    k = data_provider.n_classes
    logits_out = np.zeros((n, k, n_samples), dtype=np.float32)

    # probly's Sampler converts to CategoricalDistribution via
    # create_categorical_distribution_from_logits
    # (src/probly/predictor/_common.py:195-200). We cache raw
    # pre-softmax logits so the loss-specific decompositions
    # (Task 6.5) can produce a (N, K, S) tensor that downstream
    # code is free to interpret. We therefore run forward passes
    # manually instead of using Sampler.predict().
    with torch.no_grad():
        for s in range(n_samples):
            offset = 0
            for x, _ in data_provider:
                x = x.to(device, non_blocking=True)
                out = model(x).detach().cpu().to(torch.float32).numpy()
                bsz = out.shape[0]
                logits_out[offset : offset + bsz, :, s] = out
                offset += bsz
    return {
        "logits": logits_out,
        "indices": data_provider.indices.astype(np.int64, copy=False),
    }


__all__ = ["McDropoutHandle", "extract", "fit", "load", "save"]
