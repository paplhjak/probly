"""Evidential-Classification wrapper for the epistemic-eval pipeline.

Mirrors :mod:`mc_dropout` and :mod:`ensemble`, with two differences:

* The base classifier is wrapped with
  :func:`probly.method.evidential.classification.evidential_classification`
  (file:line ``src/probly/method/evidential/classification/common.py:42``)
  so the model outputs Dirichlet concentration parameters
  ``alpha = softplus(logit) + 1`` instead of raw logits.
* Training uses a soft-label-aware evidential cross-entropy loss
  (see :func:`_evidential_ce_soft`); probly's own
  ``evidential_ce_loss`` (file:line
  ``src/probly/train/evidential/torch.py:332``) only takes integer
  targets and the experiment's data providers yield ``(B, K)``
  soft-label rows.

Cache schema (``output_schema: "evidential_alpha"``):

* ``alpha`` of shape ``(N, K)``  -- Dirichlet concentration parameters.
* ``evidence`` of shape ``(N,)``  -- per-point ``alpha_0 - K`` scalar
  (the ``softplus(logit)`` sum), commonly reported alongside Dirichlet
  methods.

Reproducibility: as with :mod:`mc_dropout`, all stochasticity flows
through ``torch.manual_seed`` set in :func:`_base.setup_determinism`
at the top of :func:`fit` and :func:`extract`. CUDA non-determinism
(atomic ops on GPU) may cause sub-1e-4 alpha drift across runs; this
is well below inter-seed variance and does not affect reported
metrics. Bit-identical reproducibility holds only on CPU.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from probly.method.evidential.classification import evidential_classification
from probly.predictor import LogitClassifier

from ._base import FeatureProvider, setup_determinism


@dataclass
class EvidentialClassificationHandle:
    """Per-run state produced by :func:`fit`.

    Attributes:
        state_dict: Trained weights of the evidential-wrapped model
            (CPU tensors). Includes the ``Softplus`` and ``+1``
            modules' (parameterless) state, which round-trips
            cleanly.
        head_factory_args: Args for the head factory in
            ``linear_probe`` mode; ``None`` in ``full_network``
            mode.
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
    """Read ``key`` from ``config`` or raise loudly if missing."""
    if key not in config:
        msg = f"missing required config key '{key}'"
        raise KeyError(msg)
    return config[key]


def _evidential_ce_soft(alphas: torch.Tensor, y_soft: torch.Tensor) -> torch.Tensor:
    """Soft-label evidential CE (Sensoy 2018, Bayes-risk extension).

    Computes the per-batch loss

        loss = E_{y ~ y_soft}[ digamma(alpha_0) - digamma(alpha_y) ]
             = digamma(alpha_0) - sum_k y_soft[k] * digamma(alpha[k])

    where ``alpha_0 = sum_k alpha[k]``. This is the Bayes-risk
    extension of Sensoy 2018's hard-label evidential cross-entropy
    (probly's ``evidential_ce_loss`` at
    ``src/probly/train/evidential/torch.py:332``) and reduces to it
    when ``y_soft`` is one-hot. Alternative KL-style generalisations
    (e.g. ``KL(y_soft || p_hat)`` where ``p_hat = alpha / alpha_0``)
    exist but are NOT used here.

    Reference: Sensoy, Kaplan, Kandemir, "Evidential Deep Learning to
    Quantify Classification Uncertainty," NeurIPS 2018,
    https://arxiv.org/abs/1806.01768.

    Args:
        alphas: ``(B, K)`` Dirichlet concentration parameters
            (strictly positive). Probly's evidential head produces
            ``softplus(logit) + 1`` so this holds by construction.
        y_soft: ``(B, K)`` soft labels in ``[0, 1]`` summing to 1
            along the class axis. The experiment's full-network and
            linear-probe providers both emit this format.

    Returns:
        Scalar loss (mean over the batch).
    """
    alpha0 = alphas.sum(dim=1, keepdim=True)
    per_sample = (
        torch.digamma(alpha0).squeeze(1) - (y_soft * torch.digamma(alphas)).sum(dim=1)
    )
    return per_sample.mean()


def _train_evidential_model(
    model: nn.Module,
    data_provider: FeatureProvider,
    method_config: dict[str, Any],
    seed: int,
) -> nn.Module:
    """Train an evidential-classification model with soft-label evidential CE.

    Reads ``epochs``, ``lr``, ``weight_decay``, ``momentum``,
    ``nesterov`` from ``method_config``.

    Runs on GPU when one is available; falls back to CPU otherwise.

    Optimisation recipe (locked across all four UQ methods, mirrors
    :mod:`scripts.train_classifier`'s basecls path): SGD with momentum
    and cosine schedule. Sensoy 2018's original implementation uses
    Adam, but the soft-label evidential CE in :func:`_evidential_ce_soft`
    is well-posed under SGD; this codebase locks SGD-cosine to keep
    the cross-method comparison recipe-uniform. See
    ``decisions.md`` -> "Methods to evaluate" for the rationale.

    Per-epoch progress is printed to stdout in the format
    ``epoch <e>/<E>: train_loss=<float> lr=<float>`` so SLURM logs
    surface training progress. The accuracy columns from
    :mod:`scripts.train_classifier` are omitted because this loop's
    loss is on Dirichlet concentration parameters rather than class
    probabilities, so a top-1 accuracy is not directly defined.
    """
    setup_determinism(seed)
    epochs = int(method_config.get("epochs", 1))
    lr = float(method_config.get("lr", 1.0e-3))
    weight_decay = float(method_config.get("weight_decay", 0.0))
    momentum = float(method_config.get("momentum", 0.9))
    nesterov = bool(method_config.get("nesterov", False))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )
    model.train()
    for epoch in range(epochs):
        lr_now = float(scheduler.get_last_lr()[0])
        epoch_loss = 0.0
        n_batches = 0
        for x, y in data_provider:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            alphas = model(x)
            loss = _evidential_ce_soft(alphas, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach().item())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)
        print(
            f"epoch {epoch + 1}/{epochs}: "
            f"train_loss={train_loss:.4f} lr={lr_now:.4f}",
            flush=True,
        )
        scheduler.step()
    return model


def fit(
    method_config: dict[str, Any],
    dataset_config: dict[str, Any],
    data_provider: FeatureProvider,
    model_factory: Callable[[], nn.Module],
    seed: int,
) -> EvidentialClassificationHandle:
    """Train an evidential-classification model and return its handle.

    Steps:

    1. Build a fresh base model from ``model_factory``.
    2. Register it as a :class:`probly.predictor.LogitClassifier` so
       probly's transform dispatcher accepts it.
    3. Wrap it with :func:`probly.method.evidential.classification.evidential_classification`
       which appends ``Softplus`` and ``+ 1`` so the model now emits
       Dirichlet ``alpha``.
    4. Train with :func:`_train_evidential_model`.

    Args:
        method_config: Resolved method config dict. Reads
            ``epochs``, ``lr``, ``weight_decay``, and (optionally)
            ``head_factory_args``.
        dataset_config: Resolved dataset config dict.
        data_provider: Training data provider yielding
            ``(images_or_features, soft_labels)`` batches.
        model_factory: Zero-argument callable returning a fresh
            base model.
        seed: Root seed for this run.

    Returns:
        The trained :class:`EvidentialClassificationHandle`.
    """
    base_model = model_factory()
    LogitClassifier.register_instance(base_model)
    model: nn.Module = evidential_classification(base_model)  # ty:ignore[invalid-argument-type]
    trained = _train_evidential_model(model, data_provider, method_config, seed)
    head_factory_args = method_config.get("head_factory_args")
    return EvidentialClassificationHandle(
        state_dict={k: v.detach().cpu() for k, v in trained.state_dict().items()},
        head_factory_args=dict(head_factory_args) if head_factory_args is not None else None,
        method_config=dict(method_config),
        dataset_config=dict(dataset_config),
        seed=int(seed),
    )


def save(handle: EvidentialClassificationHandle, path: Path) -> None:
    """Persist a handle to ``path`` via ``torch.save``.

    Args:
        handle: The handle to save.
        path: Destination file (parent directory must exist).
    """
    torch.save(asdict(handle), path)


def load(path: Path) -> EvidentialClassificationHandle:
    """Reload a handle previously written by :func:`save`."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return EvidentialClassificationHandle(**blob)


def extract(
    handle: EvidentialClassificationHandle,
    data_provider: FeatureProvider,
    n_samples: int | None = None,  # noqa: ARG001
    *,
    model_factory: Callable[[], nn.Module],
) -> dict[str, np.ndarray]:
    """Forward-pass test data, return cached arrays.

    Returns:
        Dict with::

            "alpha":    (N, K) float32 Dirichlet concentration parameters.
            "evidence": (N,)   float32 per-point alpha_0 - K scalar.
            "indices":  (N,)   int64   per-row dataset index for traceability.

    The ``n_samples`` argument is ignored (Evidential is single-pass);
    it is part of the signature for symmetry with :mod:`mc_dropout`.
    """
    setup_determinism(handle.seed)
    base_model = model_factory()
    LogitClassifier.register_instance(base_model)
    model: nn.Module = evidential_classification(base_model)  # ty:ignore[invalid-argument-type]
    model.load_state_dict(handle.state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    n = data_provider.n_samples
    k = data_provider.n_classes
    alpha_out = np.zeros((n, k), dtype=np.float32)

    with torch.no_grad():
        offset = 0
        for x, _ in data_provider:
            x = x.to(device, non_blocking=True)
            alphas = model(x).detach().cpu().to(torch.float32).numpy()
            bsz = alphas.shape[0]
            alpha_out[offset : offset + bsz, :] = alphas
            offset += bsz
    evidence_out = alpha_out.sum(axis=1) - float(k)
    return {
        "alpha": alpha_out,
        "evidence": evidence_out.astype(np.float32, copy=False),
        "indices": data_provider.indices.astype(np.int64, copy=False),
    }


__all__ = ["EvidentialClassificationHandle", "extract", "fit", "load", "save"]
