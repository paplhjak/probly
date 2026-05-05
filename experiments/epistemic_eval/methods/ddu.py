"""DDU (Deep Deterministic Uncertainty) wrapper for the epistemic-eval pipeline.

Two-phase pipeline (Mukhoti et al. 2023):

* Phase A: train a base classifier with spectral-norm-restricted
  hidden Linear/Conv2d layers and LeakyReLU activations. probly's
  :func:`probly.method.ddu.ddu` (file:line
  ``src/probly/method/ddu/_common.py:34``) installs the spectral-norm
  parametrisations + activation swaps; we train against the
  resulting model with the same soft-label CE used for MC-Dropout
  and Ensembles.

* Phase B: walk the training provider once with no-grad, collect
  the encoder's penultimate features and integer hard labels
  (``argmax`` of the soft-label rows), then fit the per-class
  Gaussian mixture via
  :meth:`probly.method.ddu.torch.TorchDDUPredictor.fit_density_head`
  (file:line ``src/probly/method/ddu/torch.py:331``).

probly's DDU forward returns ``(logits, per_class_log_density)`` of
shape ``(N, K)`` each. We cache:

* ``probs`` of shape ``(N, K)``  -- ``softmax(logits)``.
* ``density`` of shape ``(N,)``  -- ``logsumexp(per_class_log_density,
  dim=-1)``, the marginal log-density ``log q(z) = log sum_c pi_c
  N(z; mu_c, Sigma_c)``.

Cache schema: ``output_schema: "ddu_probs_density"``.

Caveat: probly's DDU is "designed for ResNet-class architectures with
residual connections" and emits a UserWarning otherwise. The smoke
test exercises a small MLP without residuals; the warning is benign
for plumbing verification. Production runs use ResNet18 / ResNet50
backbones where DDU is fully defined.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import nn

from probly.method.ddu import ddu
from probly.predictor import LogitClassifier

from ._base import (
    FeatureProvider,
    build_optimizer,
    setup_determinism,
    train_with_val_tracking,
)


@dataclass
class DDUHandle:
    """Per-run state produced by :func:`fit`.

    Attributes:
        state_dict: Trained DDU predictor weights (CPU tensors). The
            spectral-norm parametrisations + LeakyReLU activations +
            post-fit GMM buffers (``means``, ``scale_tril``,
            ``log_pi``) all round-trip through here because they are
            registered as buffers on
            :class:`probly.method.ddu.torch.TorchDDUPredictor`.
        head_factory_args: Args for the head factory in
            ``linear_probe`` mode; ``None`` in ``full_network`` mode.
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


def _train_ddu_classifier(
    ddu_predictor: nn.Module,
    data_provider: FeatureProvider,
    method_config: dict[str, Any],
    seed: int,
    *,
    val_provider: FeatureProvider | None = None,
) -> nn.Module:
    """Train the DDU predictor's classifier on soft-label CE.

    The forward pass returns ``(logits, densities)``; we discard the
    densities at training time and train only the
    encoder + classification head pair via the same
    soft-label-CE recipe used by :mod:`mc_dropout`.

    Reads ``epochs``, ``lr``, ``weight_decay``, ``momentum``,
    ``nesterov`` from ``method_config``.

    Optimisation recipe (locked across all four UQ methods, mirrors
    :mod:`scripts.train_classifier`'s basecls path): SGD with momentum
    and cosine schedule. Mukhoti 2023's published DDU recipe uses
    SGD-cosine for the spectral-norm classifier so this matches the
    literature directly.

    Per-epoch progress is printed to stdout in the format
    ``epoch <e>/<E>: train_loss=<float> lr=<float>`` so SLURM logs
    surface training progress. The val/acc columns from
    :mod:`scripts.train_classifier` are omitted because this loop has
    no held-out validation provider and the wrapper deliberately
    keeps Phase A architecture-agnostic.
    """
    setup_determinism(seed)
    epochs = int(method_config.get("epochs", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddu_predictor = ddu_predictor.to(device)
    optimizer = build_optimizer(
        ddu_predictor,
        optimizer_name=str(method_config.get("optimizer", "sgd")).lower(),
        lr=float(method_config.get("lr", 1.0e-3)),
        momentum=float(method_config.get("momentum", 0.9)),
        weight_decay=float(method_config.get("weight_decay", 0.0)),
        nesterov=bool(method_config.get("nesterov", False)),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1)
    )

    # DDU's forward returns ``(logits, per_class_log_density)``; we
    # train Phase A against the logits with the standard soft-label
    # CE. The density head is fit post-hoc in :func:`_fit_density_head`.
    def _ddu_compute_loss(
        model: nn.Module, x: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        logits, _ = model(x)
        log_probs = torch.log_softmax(logits, dim=1)
        return -(y * log_probs).sum(dim=1).mean()

    return train_with_val_tracking(
        ddu_predictor,
        data_provider,
        val_provider,
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=epochs,
        compute_loss=_ddu_compute_loss,
    )


def _fit_density_head(
    ddu_predictor: nn.Module,
    data_provider: FeatureProvider,
) -> None:
    """Walk the train provider once and fit the GMM on encoder features.

    The soft-label provider yields ``(B, K)`` rows; the GMM expects
    integer class indices, so we ``argmax`` per row to derive hard
    labels. This is faithful to DDU's published recipe (a
    class-conditional GMM whose components are indexed by class).
    """
    device = next(ddu_predictor.parameters()).device
    ddu_predictor.eval()
    # ``ddu_predictor`` is a probly ``TorchDDUPredictor`` with
    # ``encoder`` and ``density_head`` submodules; ty cannot narrow the
    # generic nn.Module accessor, so cast the attribute reads.
    encoder = cast("nn.Module", ddu_predictor.encoder)
    density_head = ddu_predictor.density_head
    feature_chunks: list[torch.Tensor] = []
    label_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in data_provider:
            x = x.to(device, non_blocking=True)
            features = encoder(x)
            feature_chunks.append(features.detach())
            label_chunks.append(y.argmax(dim=1).to(device, non_blocking=True))
    features_cat = torch.cat(feature_chunks, dim=0)
    labels_cat = torch.cat(label_chunks, dim=0)
    density_head.fit(features_cat, labels_cat)  # ty:ignore[call-non-callable,unresolved-attribute]
    _sanitise_density_head(density_head)


def _sanitise_density_head(density_head: nn.Module) -> None:
    """Zero out classes whose Cholesky factor contains NaN entries.

    probly's :class:`probly.method.ddu.torch.GaussianMixtureHead` walks
    a jitter ladder until ``torch.linalg.cholesky`` stops raising
    ``LinAlgError``. On near-singular sample covariances Cholesky can
    return *without raising* but with NaN entries in the lower-
    triangular factor (numerical underflow during the back-substitute
    pass on a marginally PSD matrix). When that happens the affected
    class's log-density is NaN for every test point, and the marginal
    ``logsumexp(per_class_log_density)`` is NaN-poisoned for every row
    of the cache.

    APPA-REAL surfaces this on the linear-probe head: with 101 age
    classes and ~75 train samples per populated class against a
    128-dim feature space, several class covariances are
    rank-deficient, and one (age 45 on seed 0) fell into the
    Cholesky-without-error-but-with-NaN failure mode.

    The fix is downstream of probly: any class whose ``scale_tril``
    contains a NaN gets its prior mass redirected (``log_pi = -inf``)
    and its factor reset to identity. It then contributes nothing to
    the mixture density, leaving the well-fit classes intact. This is
    benign for OOD detection (ill-fit classes shouldn't be scoring
    points anyway) and avoids a probly fork.

    Args:
        density_head: A fitted GaussianMixtureHead (probly buffer
            layout: ``means``, ``scale_tril``, ``log_pi``).
    """
    scale_tril = cast("torch.Tensor", density_head.scale_tril)
    log_pi = cast("torch.Tensor", density_head.log_pi)
    bad = torch.isnan(scale_tril).any(dim=(1, 2))
    if not bad.any():
        return
    feature_dim = scale_tril.shape[-1]
    eye = torch.eye(feature_dim, device=scale_tril.device, dtype=scale_tril.dtype)
    scale_tril[bad] = eye
    log_pi[bad] = float("-inf")


def fit(
    method_config: dict[str, Any],
    dataset_config: dict[str, Any],
    data_provider: FeatureProvider,
    model_factory: Callable[[], nn.Module],
    seed: int,
    *,
    val_data_provider: FeatureProvider | None = None,
) -> DDUHandle:
    """Train a DDU predictor end-to-end and return its handle.

    Phase A: build a fresh base model, register it as a
    :class:`probly.predictor.LogitClassifier`, call
    :func:`probly.method.ddu.ddu` to install spectral-norm and
    activation swaps, and train via :func:`_train_ddu_classifier`.

    Phase B: walk the training provider once and call
    :func:`_fit_density_head` to populate the GMM buffers.

    Args:
        method_config: Resolved method config dict. Reads
            ``sn_coeff`` (default 3.0; passed to probly's
            :func:`ddu`), ``epochs``, ``lr``, ``weight_decay``.
        dataset_config: Resolved dataset config dict.
        data_provider: Training data provider yielding
            ``(images_or_features, soft_labels)`` batches.
        model_factory: Zero-argument callable returning a fresh
            base model.
        seed: Root seed for this run.

    Returns:
        The trained :class:`DDUHandle`.
    """
    base_model = model_factory()
    LogitClassifier.register_instance(base_model)
    sn_coeff = float(method_config.get("sn_coeff", 3.0))
    ddu_predictor = ddu(base_model, sn_coeff=sn_coeff)
    trained = _train_ddu_classifier(
        ddu_predictor,
        data_provider,
        method_config,
        seed,
        val_provider=val_data_provider,
    )
    _fit_density_head(trained, data_provider)
    head_factory_args = method_config.get("head_factory_args")
    return DDUHandle(
        state_dict={k: v.detach().cpu() for k, v in trained.state_dict().items()},
        head_factory_args=dict(head_factory_args) if head_factory_args is not None else None,
        method_config=dict(method_config),
        dataset_config=dict(dataset_config),
        seed=int(seed),
    )


def save(handle: DDUHandle, path: Path) -> None:
    """Persist a handle to ``path`` via ``torch.save``."""
    torch.save(asdict(handle), path)


def load(path: Path) -> DDUHandle:
    """Reload a handle previously written by :func:`save`."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return DDUHandle(**blob)


def extract(
    handle: DDUHandle,
    data_provider: FeatureProvider,
    n_samples: int | None = None,  # noqa: ARG001
    *,
    model_factory: Callable[[], nn.Module],
) -> dict[str, np.ndarray]:
    """Forward-pass test data, return cached arrays.

    Returns:
        Dict with::

            "probs":   (N, K) float32 softmax of the classifier logits.
            "density": (N,)   float32 marginal log-density log q(z).
            "indices": (N,)   int64   per-row dataset index.

    The ``n_samples`` argument is ignored (DDU is single-pass);
    accepted for API symmetry with :mod:`mc_dropout`.
    """
    setup_determinism(handle.seed)
    base_model = model_factory()
    LogitClassifier.register_instance(base_model)
    sn_coeff = float(handle.method_config.get("sn_coeff", 3.0))
    ddu_predictor = ddu(base_model, sn_coeff=sn_coeff)
    ddu_predictor.load_state_dict(handle.state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ddu_predictor = ddu_predictor.to(device)
    ddu_predictor.eval()

    n = data_provider.n_samples
    k = data_provider.n_classes
    probs_out = np.zeros((n, k), dtype=np.float32)
    density_out = np.zeros((n,), dtype=np.float32)

    with torch.no_grad():
        offset = 0
        for x, _ in data_provider:
            x = x.to(device, non_blocking=True)
            logits, per_class_log_density = ddu_predictor(x)
            probs = torch.softmax(logits, dim=1)
            density = torch.logsumexp(per_class_log_density, dim=1)
            bsz = probs.shape[0]
            probs_out[offset : offset + bsz, :] = probs.detach().cpu().to(torch.float32).numpy()
            density_out[offset : offset + bsz] = density.detach().cpu().to(torch.float32).numpy()
            offset += bsz
    return {
        "probs": probs_out,
        "density": density_out,
        "indices": data_provider.indices.astype(np.int64, copy=False),
    }


__all__ = ["DDUHandle", "extract", "fit", "load", "save"]
